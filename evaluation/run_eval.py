"""
RAG 评估运行入口

支持三种评估模式：
    - retrieval:  仅评估检索阶段（Hit Rate / MRR / Context Recall）
    - generation: 仅评估生成阶段（需提供已有的 query+answer+contexts）
    - full:       全链路评估（检索 + 生成 + 7 个指标）

全链路评估流程：
    1. 加载评估数据集（JSON 文件或内置示例）
    2. 对每个 query 调用 HybridRetriever.aretrieve 获取检索结果
    3. 用检索结果作为 context，调用 LLM（qwen-turbo）生成答案
    4. 对检索结果运行 3 个检索评估器
    5. 对生成答案运行 4 个生成评估器
    6. 汇总结果，输出格式化评估报告
    7. 同时上传到 LangSmith（使用 @traceable 包装）

使用示例：
    # 列出 LangSmith 账号下所有数据集（便于查找名称）
    python -m evaluation.run_eval --list-datasets

    # 从 LangSmith 拉取数据集进行全链路评估
    python -m evaluation.run_eval --mode full --langsmith-dataset my-rag-dataset

    # 通过数据集 ID 拉取（用于重名场景）
    python -m evaluation.run_eval --mode full --langsmith-dataset-id <uuid>

    # 使用本地 JSON 数据集
    python -m evaluation.run_eval --mode full --dataset ./eval_data.json

    # 内置示例数据（不指定数据源时默认）
    python -m evaluation.run_eval --mode full

    # 仅评估检索阶段
    python -m evaluation.run_eval --mode retrieval --langsmith-dataset my-rag-dataset
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langsmith import traceable

# 确保项目根目录在 sys.path 中（支持 python -m 与直接运行两种方式）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.config import settings
from evaluation.dataset import (
    EvalExample,
    SAMPLE_EXAMPLES,
    fetch_dataset_from_langsmith,
    fetch_dataset_from_langsmith_by_id,
    list_langsmith_datasets,
    load_dataset,
)
from evaluation.evaluators import (
    DASHSCOPE_BASE_URL,
    GENERATION_EVALUATORS,
    RETRIEVAL_EVALUATORS,
    ALL_EVALUATORS,
    EvaluationResult,
    LLMJudge,
    RunEvaluator,
    docs_to_texts,
    run_evaluators_async,
)

load_dotenv(".env.dev")
logger = logging.getLogger(__name__)


# =============================================================================
# 常量
# =============================================================================

GENERATION_MODEL: str = "qwen-turbo"
"""生成答案使用的模型（速度优先）。"""

RETRIEVAL_TOP_K: int = 5
"""全链路评估时检索返回的文档数。"""

GENERATION_MAX_TOKENS: int = 1024
"""生成答案最大 token 数。"""

GENERATION_PROMPT_TEMPLATE: str = """你是一个严谨的知识库问答助手。请仅根据下方检索到的上下文回答用户问题。
如果上下文中没有相关信息，请直接回答"根据已知信息无法回答该问题"，不要编造内容。

【检索到的上下文】
{context}

【用户问题】
{query}

【要求】
1. 答案必须完全基于上下文，不得引入外部知识。
2. 回答简洁、准确、直接针对问题。
3. 如实反映上下文中的信息，不要过度推断。
"""


# =============================================================================
# 评估结果汇总数据结构
# =============================================================================

@dataclass
class ExampleResult:
    """单条样例的评估结果。"""

    query: str
    ground_truth_answer: str
    retrieved_docs: List[str] = field(default_factory=list)
    generated_answer: str = ""
    retrieval_results: List[EvaluationResult] = field(default_factory=list)
    generation_results: List[EvaluationResult] = field(default_factory=list)

    @property
    def all_results(self) -> List[EvaluationResult]:
        return self.retrieval_results + self.generation_results


@dataclass
class EvalReport:
    """整体评估报告。"""

    mode: str
    total_examples: int
    example_results: List[ExampleResult] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def metric_averages(self) -> Dict[str, float]:
        """计算每个指标在所有样例上的平均分。"""
        sums: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        for ex in self.example_results:
            for res in ex.all_results:
                sums[res.key] = sums.get(res.key, 0.0) + (res.score or 0.0)
                counts[res.key] = counts.get(res.key, 0) + 1
        return {k: sums[k] / counts[k] for k in sums if counts[k] > 0}


# =============================================================================
# LLM 答案生成
# =============================================================================

@traceable(name="eval_generate_answer", tags=["evaluation", "generation"])
async def generate_answer(query: str, contexts: List[str]) -> str:
    """
    使用 qwen-turbo 根据检索上下文生成答案。

    与生产 RAG 服务解耦：评估时使用独立的轻量 prompt，
    避免引入对话历史、Memory 等额外变量，保证评估可复现。

    Args:
        query:    用户问题
        contexts: 检索到的上下文列表

    Returns:
        LLM 生成的答案文本
    """
    if not contexts:
        return "根据已知信息无法回答该问题"

    context_text = "\n---\n".join(contexts)
    prompt = GENERATION_PROMPT_TEMPLATE.format(context=context_text, query=query)

    llm = ChatOpenAI(
        model=GENERATION_MODEL,
        openai_api_key=settings.api_key,
        openai_api_base=DASHSCOPE_BASE_URL,
        temperature=0.3,
        max_tokens=GENERATION_MAX_TOKENS,
    )
    response = await llm.ainvoke(prompt)
    return (response.content if hasattr(response, "content") else str(response)).strip()


# =============================================================================
# 全链路评估核心
# =============================================================================

@traceable(name="eval_full_pipeline", tags=["evaluation", "full"])
async def evaluate_example_full(
    example: EvalExample,
    retriever: Any,
    evaluators: Optional[List[RunEvaluator]] = None,
) -> ExampleResult:
    """
    对单条样例执行全链路评估。

    流程：
        1. 异步检索（HybridRetriever.aretrieve）
        2. LLM 生成答案
        3. 并发运行检索评估器
        4. 并发运行生成评估器

    Args:
        example:   评估样例
        retriever: HybridRetriever 实例
        evaluators: 可选的评估器列表，默认使用 ALL_EVALUATORS
    """
    evaluators = evaluators or ALL_EVALUATORS
    retrieval_evals = [e for e in evaluators if e in RETRIEVAL_EVALUATORS]
    generation_evals = [e for e in evaluators if e in GENERATION_EVALUATORS]

    # 1. 检索
    logger.info(f"[full] 检索: {example.query[:60]}...")
    docs = await retriever.aretrieve(query=example.query, top_k=RETRIEVAL_TOP_K)
    retrieved_texts = docs_to_texts(docs)

    # 2. 生成答案
    logger.info(f"[full] 生成答案: {example.query[:60]}...")
    answer = await generate_answer(example.query, retrieved_texts)

    # 3. 检索阶段评估（并发）
    retrieval_inputs = {
        "query": example.query,
        "retrieved_docs": retrieved_texts,
        "ground_truth_contexts": example.ground_truth_contexts,
    }
    retrieval_results = await run_evaluators_async(retrieval_evals, retrieval_inputs)

    # 4. 生成阶段评估（并发）
    generation_inputs = {
        "query": example.query,
        "answer": answer,
        "contexts": retrieved_texts,
        "ground_truth": example.ground_truth_answer,
    }
    generation_results = await run_evaluators_async(generation_evals, generation_inputs)

    return ExampleResult(
        query=example.query,
        ground_truth_answer=example.ground_truth_answer,
        retrieved_docs=retrieved_texts,
        generated_answer=answer,
        retrieval_results=retrieval_results,
        generation_results=generation_results,
    )


async def evaluate_example_retrieval_only(
    example: EvalExample,
    retriever: Any,
    evaluators: Optional[List[RunEvaluator]] = None,
) -> ExampleResult:
    """仅评估检索阶段。"""
    evaluators = evaluators or RETRIEVAL_EVALUATORS

    docs = await retriever.aretrieve(query=example.query, top_k=RETRIEVAL_TOP_K)
    retrieved_texts = docs_to_texts(docs)

    inputs = {
        "query": example.query,
        "retrieved_docs": retrieved_texts,
        "ground_truth_contexts": example.ground_truth_contexts,
    }
    results = await run_evaluators_async(evaluators, inputs)

    return ExampleResult(
        query=example.query,
        ground_truth_answer=example.ground_truth_answer,
        retrieved_docs=retrieved_texts,
        retrieval_results=results,
    )


async def evaluate_example_generation_only(
    example: EvalExample,
    retriever: Any,
    evaluators: Optional[List[RunEvaluator]] = None,
) -> ExampleResult:
    """
    仅评估生成阶段。

    仍需执行检索以获取 contexts，但不评估检索指标。
    """
    evaluators = evaluators or GENERATION_EVALUATORS

    docs = await retriever.aretrieve(query=example.query, top_k=RETRIEVAL_TOP_K)
    retrieved_texts = docs_to_texts(docs)
    answer = await generate_answer(example.query, retrieved_texts)

    inputs = {
        "query": example.query,
        "answer": answer,
        "contexts": retrieved_texts,
        "ground_truth": example.ground_truth_answer,
    }
    results = await run_evaluators_async(evaluators, inputs)

    return ExampleResult(
        query=example.query,
        ground_truth_answer=example.ground_truth_answer,
        retrieved_docs=retrieved_texts,
        generated_answer=answer,
        generation_results=results,
    )


# =============================================================================
# 评估器工厂（按模式选择评估器集合）
# =============================================================================

_MODE_DISPATCH = {
    "retrieval":  evaluate_example_retrieval_only,
    "generation": evaluate_example_generation_only,
    "full":       evaluate_example_full,
}


# =============================================================================
# 主流程
# =============================================================================

async def run_evaluation(
    examples: List[EvalExample],
    mode: str,
    retriever: Any,
) -> EvalReport:
    """
    执行评估主流程。

    Args:
        examples:  评估样例列表
        mode:      评估模式 (retrieval / generation / full)
        retriever: HybridRetriever 实例

    Returns:
        EvalReport 评估报告
    """
    if mode not in _MODE_DISPATCH:
        raise ValueError(f"不支持的评估模式: {mode}, 可选: {list(_MODE_DISPATCH.keys())}")

    eval_fn = _MODE_DISPATCH[mode]
    start_time = time.time()

    # 串行执行每条样例（避免并发检索打爆 Qdrant / BM25）
    # 样例内部的评估器已并发执行
    example_results: List[ExampleResult] = []
    for idx, ex in enumerate(examples, 1):
        logger.info(f"=== 评估样例 {idx}/{len(examples)}: {ex.query[:60]}... ===")
        try:
            result = await eval_fn(ex, retriever)
            example_results.append(result)
        except Exception as e:
            logger.error(f"样例评估失败: {ex.query[:60]}..., 错误: {e}")
            example_results.append(ExampleResult(
                query=ex.query,
                ground_truth_answer=ex.ground_truth_answer,
            ))

    elapsed = time.time() - start_time
    return EvalReport(
        mode=mode,
        total_examples=len(examples),
        example_results=example_results,
        elapsed_seconds=elapsed,
    )


def _init_retriever() -> Any:
    """
    初始化 HybridRetriever。

    复用项目已有的 init_hybrid_retriever()，避免重复实现检索器初始化逻辑。
    """
    from rag.rag_conversation_service import init_hybrid_retriever
    return init_hybrid_retriever()


# =============================================================================
# 报告格式化输出
# =============================================================================

def format_report(report: EvalReport) -> str:
    """
    将评估报告格式化为表格字符串。

    包含：
        - 总览（样例数、耗时、模式）
        - 每个指标的均分表
        - 每条样例的明细（可选）
    """
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("                    RAG 评估报告 (LangSmith)")
    lines.append("=" * 80)
    lines.append(f"评估模式: {report.mode}")
    lines.append(f"样例总数: {report.total_examples}")
    lines.append(f"总耗时:   {report.elapsed_seconds:.2f}s")
    lines.append(f"平均每条: {report.elapsed_seconds / max(1, report.total_examples):.2f}s")
    lines.append("-" * 80)

    # 指标均分表
    averages = report.metric_averages()
    if averages:
        lines.append(f"{'指标':<30} {'平均分':<10} {'等级':<10}")
        lines.append("-" * 80)
        for key, score in sorted(averages.items()):
            grade = _score_to_grade(score)
            lines.append(f"{key:<30} {score:<10.4f} {grade:<10}")
    else:
        lines.append("(无评估指标结果)")
    lines.append("-" * 80)

    # 每条样例明细
    for idx, ex in enumerate(report.example_results, 1):
        lines.append(f"[样例 {idx}] {ex.query[:70]}")
        if ex.generated_answer:
            lines.append(f"  生成答案: {ex.generated_answer[:100]}...")
        if ex.retrieval_results:
            for r in ex.retrieval_results:
                lines.append(f"  - {r.key:<28} {r.score or 0:.4f}  {r.comment or ''}")
        if ex.generation_results:
            for r in ex.generation_results:
                lines.append(f"  - {r.key:<28} {r.score or 0:.4f}  {r.comment or ''}")
        lines.append("")

    lines.append("=" * 80)
    return "\n".join(lines)


def _score_to_grade(score: float) -> str:
    """将分数转换为等级标签，便于快速判读。"""
    if score >= 0.9:
        return "优秀"
    if score >= 0.75:
        return "良好"
    if score >= 0.6:
        return "合格"
    if score >= 0.4:
        return "待改进"
    return "不合格"


# =============================================================================
# 命令行入口
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LangSmith RAG 评估入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode", choices=["retrieval", "generation", "full"],
        default="full",
        help="评估模式: retrieval=仅检索, generation=仅生成, full=全链路 (默认 full)",
    )
    # 数据源参数（三选一，优先级: --langsmith-dataset > --dataset > 内置示例）
    parser.add_argument(
        "--langsmith-dataset", type=str, default=None,
        help="从 LangSmith 拉取数据集，传入数据集名称。与 --dataset 互斥",
    )
    parser.add_argument(
        "--langsmith-dataset-id", type=str, default=None,
        help="从 LangSmith 拉取数据集，传入数据集 UUID（用于重名场景）。"
             "与 --langsmith-dataset 互斥",
    )
    parser.add_argument(
        "--list-datasets", action="store_true",
        help="列出 LangSmith 账号下所有数据集后退出（便于查找名称）",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="评估数据集 JSON 文件路径（本地数据源）。未指定时使用内置示例数据",
    )
    parser.add_argument(
        "--upload-langsmith", action="store_true", default=True,
        help="是否上传评估结果到 LangSmith (默认开启，需配置 LANGSMITH_API_KEY)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别 (默认 INFO)",
    )
    return parser.parse_args()


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


async def async_main(args: argparse.Namespace) -> EvalReport:
    """异步主流程，便于在测试中调用。"""
    # 1. 加载数据集（支持三种数据源，优先级: LangSmith > 本地 JSON > 内置示例）
    examples = _load_examples(args)

    logger.info(
        f"评估模式: {args.mode}, 样例数: {len(examples)}"
    )

    # 2. 初始化检索器
    logger.info("初始化 HybridRetriever...")
    retriever = _init_retriever()
    logger.info("HybridRetriever 初始化完成")

    # 3. 执行评估
    report = await run_evaluation(examples, args.mode, retriever)

    # 4. 输出报告
    print(format_report(report))

    # 5. 上传 LangSmith（可选）
    # 注：评估过程中的 @traceable 已自动上传 trace，
    # 此处补充上传汇总结果到 LangSmith 的反馈系统
    if args.upload_langsmith:
        try:
            _upload_summary_to_langsmith(report)
        except Exception as e:
            logger.warning(f"上传汇总结果到 LangSmith 失败（非致命）: {e}")

    return report


def _load_examples(args: argparse.Namespace) -> List[EvalExample]:
    """
    根据命令行参数加载评估样例。

    数据源优先级：
        1. --langsmith-dataset / --langsmith-dataset-id：从 LangSmith 平台拉取
        2. --dataset：从本地 JSON 文件加载
        3. 默认：使用内置示例数据

    Args:
        args: 命令行参数。

    Returns:
        EvalExample 列表。

    Raises:
        ValueError: 同时指定了互斥的数据源参数。
    """
    # 参数互斥校验
    ls_sources = [args.langsmith_dataset, args.langsmith_dataset_id]
    specified_ls = [s for s in ls_sources if s]
    if len(specified_ls) > 1:
        raise ValueError(
            "--langsmith-dataset 与 --langsmith-dataset-id 互斥，请只指定一个"
        )
    if specified_ls and args.dataset:
        raise ValueError(
            "--langsmith-dataset(-id) 与 --dataset 互斥，请只指定一个数据源"
        )

    # 1. LangSmith 数据集（按名称）
    if args.langsmith_dataset:
        logger.info(f"从 LangSmith 拉取数据集: {args.langsmith_dataset}")
        return fetch_dataset_from_langsmith(args.langsmith_dataset)

    # 2. LangSmith 数据集（按 ID，用于重名场景）
    if args.langsmith_dataset_id:
        logger.info(f"从 LangSmith 拉取数据集 (ID): {args.langsmith_dataset_id}")
        return fetch_dataset_from_langsmith_by_id(args.langsmith_dataset_id)

    # 3. 本地 JSON 文件
    if args.dataset:
        logger.info(f"从本地 JSON 加载数据集: {args.dataset}")
        return load_dataset(args.dataset)

    # 4. 默认：内置示例
    logger.info("未指定数据源，使用内置示例数据 (3 条)")
    return SAMPLE_EXAMPLES


def _upload_summary_to_langsmith(report: EvalReport) -> None:
    """
    将评估汇总结果上传到 LangSmith。

    通过 langsmith.Client.create_feedback 上传每个指标的均分，
    便于在 LangSmith 仪表盘中可视化追踪。
    """
    from langsmith import Client

    client = Client()
    averages = report.metric_averages()
    project_name = os.getenv("LANGSMITH_PROJECT", "lingxi-agent")

    for metric_name, score in averages.items():
        client.create_feedback(
            key=f"{metric_name}_avg",
            score=score,
            comment=f"{report.mode} 模式下 {metric_name} 在 {report.total_examples} "
                    f"条样例上的平均分",
            project_name=project_name,
        )
    logger.info(
        f"评估汇总已上传 LangSmith: {len(averages)} 个指标, 项目: {project_name}"
    )


def _print_langsmith_datasets() -> None:
    """列出 LangSmith 账号下所有数据集，以表格形式打印。"""
    try:
        datasets = list_langsmith_datasets()
    except Exception as e:
        print(f"获取 LangSmith 数据集列表失败: {e}")
        return

    if not datasets:
        print("LangSmith 账号下暂无数据集")
        return

    print("=" * 90)
    print(f"{'序号':<6}{'名称':<30}{'ID':<40}{'创建时间'}")
    print("-" * 90)
    for idx, ds in enumerate(datasets, 1):
        print(f"{idx:<6}{ds['name']:<30}{ds['id']:<40}{ds['created_at']}")
    print("=" * 90)
    print(f"共 {len(datasets)} 个数据集")
    print("\n使用方式:")
    print("  python -m evaluation.run_eval --langsmith-dataset <名称>")
    print("  python -m evaluation.run_eval --langsmith-dataset-id <ID>")


def main() -> None:
    """命令行入口。"""
    args = parse_args()
    _setup_logging(args.log_level)

    # --list-datasets: 仅列出 LangSmith 数据集后退出
    if args.list_datasets:
        _print_langsmith_datasets()
        return

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
