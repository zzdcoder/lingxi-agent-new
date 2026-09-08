"""
RAG 评估器实现

本模块实现 7 个核心评估指标，分为检索阶段（3 个）与生成阶段（4 个）：

检索阶段（输入：query, retrieved_docs, ground_truth_contexts）：
    - HitRateEvaluator:        Top-K 检索结果是否包含 ground truth 文档
    - MRREvaluator:            MRR，第一个命中 ground truth 文档的排名倒数
    - ContextRecallEvaluator:  检索上下文覆盖 ground truth 要点的比例

生成阶段（输入：query, answer, contexts, ground_truth）：
    - FaithfulnessEvaluator:        答案是否完全基于 context，不包含幻觉
    - AnswerCorrectnessEvaluator:   答案与 ground truth 答案的一致性
    - AnswerRelevancyEvaluator:     答案与用户问题的相关程度
    - HallucinationRateEvaluator:   幻觉内容占比（与 Faithfulness 互补）

设计原则：
    1. LLMJudge 基类封装 LLM 调用、Prompt 渲染、结果解析、异常兜底的通用逻辑，
       子类只需定义 prompt 模板与解析规则，最大化复用率。
    2. 所有评估器使用 DashScope (qwen-plus) 作为 LLM-as-Judge，通过
       langchain_openai.ChatOpenAI + DashScope 兼容模式调用。
    3. 全部 async 实现，支持 asyncio.gather 并发执行多个评估器。
    4. LLM 输出解析失败时默认返回 0 分 + 错误信息，保证评估流程不中断。
    5. 所有评估器继承 langsmith RunEvaluator，返回标准 EvaluationResult。
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from abc import abstractmethod
from typing import Any, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langsmith.evaluation import EvaluationResult, RunEvaluator
from langsmith.schemas import Example, Run

from core.config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# 常量
# =============================================================================

DASHSCOPE_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
"""DashScope OpenAI 兼容模式 endpoint。"""

JUDGE_MODEL: str = "qwen-plus"
"""LLM-as-Judge 模型，评估质量优先。"""

JUDGE_TEMPERATURE: float = 0.0
"""Judge 模型温度，0 保证评估结果可复现。"""

JUDGE_MAX_TOKENS: int = 1024
"""Judge 模型最大输出 token，评估结果通常简短。"""

DEFAULT_RETRIEVAL_K: int = 5
"""检索阶段评估默认 Top-K，影响 Hit Rate / MRR 计算。"""

_CONTEXT_HIT_THRESHOLD: float = 0.7
"""检索文档与 ground truth context 的字符重叠率阈值，超过则视为命中。

采用基于字符级 Jaccard 相似度的近似匹配，避免依赖外部 embedding 模型，
保证评估流程轻量、可独立运行。
"""


# =============================================================================
# 文档匹配工具
# =============================================================================

def _normalize_text(text: str) -> str:
    """文本归一化：去除空白与标点，便于字符级相似度计算。"""
    return re.sub(r"[\s，。、；：？！,.;:?!\"'()（）\[\]【】]+", "", text or "")


def _char_jaccard(a: str, b: str) -> float:
    """
    字符级 Jaccard 相似度。

    相比单词级 Jaccard，字符级对中文更友好（中文无空格分词）。
    时间复杂度 O(|a| + |b|)，适合短文本匹配。
    """
    sa, sb = set(_normalize_text(a)), set(_normalize_text(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _is_doc_hit(doc_text: str, ground_truth_contexts: List[str]) -> bool:
    """
    判断单篇检索文档是否命中任一 ground truth context。

    命中条件：与某条 ground truth context 的字符 Jaccard 相似度 >= 阈值。
    """
    for gt in ground_truth_contexts:
        if _char_jaccard(doc_text, gt) >= _CONTEXT_HIT_THRESHOLD:
            return True
    return False


def _find_first_hit_rank(
    retrieved_docs: List[str],
    ground_truth_contexts: List[str],
) -> Optional[int]:
    """
    返回第一个命中 ground truth 的检索文档排名（从 1 开始），未命中返回 None。
    """
    for rank, doc in enumerate(retrieved_docs, start=1):
        if _is_doc_hit(doc, ground_truth_contexts):
            return rank
    return None


# =============================================================================
# LLMJudge 基类
# =============================================================================

class LLMJudge(RunEvaluator):
    """
    LLM-as-Judge 基类。

    封装 LLM 调用、Prompt 渲染、结果解析、异常兜底的通用逻辑。
    子类只需实现：
        - evaluation_name: 评估指标名称（作为 EvaluationResult.key）
        - _build_prompt:   构造 LLM 评估 prompt
        - _parse_response: 解析 LLM 输出为 (score, comment)

    通用能力：
        - 异步调用 LLM（ChatOpenAI.ainvoke），支持并发
        - LLM 调用失败 / 输出解析失败时返回 0 分 + 错误信息
        - 自动构造 EvaluationResult，类型安全

    使用方式：
        class MyEvaluator(LLMJudge):
            def _build_prompt(self, **inputs) -> str: ...
            def _parse_response(self, text: str) -> Tuple[float, str]: ...

        evaluator = MyEvaluator()
        result = await evaluator.ajudge(**inputs)
    """

    def __init__(
        self,
        model: str = JUDGE_MODEL,
        temperature: float = JUDGE_TEMPERATURE,
        max_tokens: int = JUDGE_MAX_TOKENS,
        api_key: Optional[str] = None,
        base_url: str = DASHSCOPE_BASE_URL,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._api_key = api_key or settings.api_key
        self._base_url = base_url
        self._llm: Optional[ChatOpenAI] = None

    # -------------------------------------------------------------------------
    # 子类必须实现的抽象方法
    # -------------------------------------------------------------------------

    @property
    @abstractmethod
    def evaluation_name(self) -> str:
        """评估指标名称，作为 EvaluationResult.key。"""

    @abstractmethod
    def _build_prompt(self, **inputs: Any) -> str:
        """根据评估输入构造 LLM prompt。"""

    @abstractmethod
    def _parse_response(self, text: str) -> Tuple[float, str]:
        """
        解析 LLM 输出为 (score, comment)。

        Args:
            text: LLM 原始输出文本。

        Returns:
            (score, comment) 元组。score 范围 [0, 1]，comment 为评估说明。
        """

    # -------------------------------------------------------------------------
    # LLM 客户端惰性初始化
    # -------------------------------------------------------------------------

    def _get_llm(self) -> ChatOpenAI:
        """惰性创建 ChatOpenAI 实例，避免模块导入时即建立连接。"""
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=self.model,
                openai_api_key=self._api_key,
                openai_api_base=self._base_url,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        return self._llm

    # -------------------------------------------------------------------------
    # 核心评估入口
    # -------------------------------------------------------------------------

    async def ajudge(self, **inputs: Any) -> EvaluationResult:
        """
        异步执行 LLM 评估。

        流程：
            1. 构造 prompt
            2. 调用 LLM（失败兜底 0 分）
            3. 解析输出（失败兜底 0 分）
            4. 构造标准 EvaluationResult

        Args:
            **inputs: 评估输入，由子类的 _build_prompt 消费。

        Returns:
            标准 EvaluationResult。
        """
        try:
            prompt = self._build_prompt(**inputs)
            llm = self._get_llm()
            response = await llm.ainvoke(prompt)
            text = response.content if hasattr(response, "content") else str(response)

            try:
                score, comment = self._parse_response(text)
            except Exception as parse_err:
                logger.warning(
                    f"[{self.evaluation_name}] LLM 输出解析失败: {parse_err}, "
                    f"原始输出: {text[:200]}"
                )
                score, comment = 0.0, f"LLM 输出解析失败: {parse_err}; 原始输出: {text[:200]}"

            return EvaluationResult(
                key=self.evaluation_name,
                score=float(score),
                value=float(score),
                comment=comment,
            )
        except Exception as e:
            logger.error(f"[{self.evaluation_name}] 评估执行失败: {e}")
            return EvaluationResult(
                key=self.evaluation_name,
                score=0.0,
                value=0.0,
                comment=f"评估执行失败: {e}",
            )

    # -------------------------------------------------------------------------
    # RunEvaluator 协议实现
    # -------------------------------------------------------------------------

    def evaluate_run(
        self,
        run: Run,
        example: Optional[Example] = None,
        evaluator_run_id: Optional[uuid.UUID] = None,
    ) -> EvaluationResult:
        """
        同步评估入口（RunEvaluator 协议要求）。

        默认实现通过 asyncio.run 调用异步版本。
        在异步上下文中调用本方法会报错，请改用 aevaluate_run。
        """
        inputs = self._extract_inputs(run, example)
        return asyncio.run(self.ajudge(**inputs))

    async def aevaluate_run(
        self,
        run: Run,
        example: Optional[Example] = None,
        evaluator_run_id: Optional[uuid.UUID] = None,
    ) -> EvaluationResult:
        """
        异步评估入口（RunEvaluator 协议）。

        从 run.outputs 与 example.outputs 中提取评估所需输入，调用 ajudge。
        """
        inputs = self._extract_inputs(run, example)
        return await self.ajudge(**inputs)

    # -------------------------------------------------------------------------
    # 输入提取（子类可覆盖）
    # -------------------------------------------------------------------------

    def _extract_inputs(
        self, run: Run, example: Optional[Example]
    ) -> dict:
        """
        从 LangSmith Run / Example 中提取评估输入。

        约定 run.outputs 结构：
            - 检索类：{"query": str, "retrieved_docs": List[str], ...}
            - 生成类：{"query": str, "answer": str, "contexts": List[str], ...}
        约定 example.outputs 结构：
            {"ground_truth_answer": str, "ground_truth_contexts": List[str]}

        子类可覆盖此方法以适配自定义 Run 结构。
        """
        run_outputs = dict(run.outputs or {}) if run.outputs else {}
        example_outputs = dict(example.outputs or {}) if example and example.outputs else {}

        inputs: dict = {"query": run_outputs.get("query", "")}
        inputs.update(example_outputs)

        # 检索类评估器需要 retrieved_docs
        if "retrieved_docs" in run_outputs:
            inputs["retrieved_docs"] = run_outputs["retrieved_docs"]

        # 生成类评估器需要 answer / contexts
        if "answer" in run_outputs:
            inputs["answer"] = run_outputs["answer"]
        if "contexts" in run_outputs:
            inputs["contexts"] = run_outputs["contexts"]

        return inputs


# =============================================================================
# 检索阶段评估器
# =============================================================================

class HitRateEvaluator(LLMJudge):
    """
    Hit Rate@K 评估器。

    定义：Top-K 检索结果中是否包含至少一个 ground truth 文档。
    取值：命中为 1.0，未命中为 0.0。

    本评估器为纯规则计算，不调用 LLM，但仍继承 LLMJudge 以统一接口管理。
    """

    def __init__(self, k: int = DEFAULT_RETRIEVAL_K, **kwargs: Any):
        # Hit Rate 无需 LLM，跳过父类 LLM 初始化
        super().__init__(**kwargs)
        self.k = k

    @property
    def evaluation_name(self) -> str:
        return f"hit_rate@{self.k}"

    def _build_prompt(self, **inputs: Any) -> str:
        # 纯规则评估，不需要 prompt
        return ""

    def _parse_response(self, text: str) -> Tuple[float, str]:
        return 0.0, "Hit Rate 由规则计算，不解析 LLM 输出"

    async def ajudge(self, **inputs: Any) -> EvaluationResult:
        """重写为纯规则计算，跳过 LLM 调用。"""
        query: str = inputs.get("query", "")
        retrieved_docs: List[str] = inputs.get("retrieved_docs", [])
        ground_truth_contexts: List[str] = inputs.get("ground_truth_contexts", [])

        top_k_docs = retrieved_docs[: self.k]
        hit = any(
            _is_doc_hit(doc, ground_truth_contexts) for doc in top_k_docs
        )
        score = 1.0 if hit else 0.0
        comment = (
            f"Top-{self.k} 检索 {'命中' if hit else '未命中'} ground truth，"
            f"检索文档数={len(top_k_docs)}"
        )
        return EvaluationResult(
            key=self.evaluation_name, score=score, value=score, comment=comment
        )


class MRREvaluator(LLMJudge):
    """
    MRR (Mean Reciprocal Rank) 评估器。

    定义：第一个命中 ground truth 的检索文档排名的倒数。
    取值：1/rank（rank 从 1 开始），未命中为 0.0。

    本评估器为纯规则计算，不调用 LLM。
    """

    def __init__(self, k: int = DEFAULT_RETRIEVAL_K, **kwargs: Any):
        super().__init__(**kwargs)
        self.k = k

    @property
    def evaluation_name(self) -> str:
        return f"mrr@{self.k}"

    def _build_prompt(self, **inputs: Any) -> str:
        return ""

    def _parse_response(self, text: str) -> Tuple[float, str]:
        return 0.0, "MRR 由规则计算，不解析 LLM 输出"

    async def ajudge(self, **inputs: Any) -> EvaluationResult:
        query: str = inputs.get("query", "")
        retrieved_docs: List[str] = inputs.get("retrieved_docs", [])
        ground_truth_contexts: List[str] = inputs.get("ground_truth_contexts", [])

        top_k_docs = retrieved_docs[: self.k]
        rank = _find_first_hit_rank(top_k_docs, ground_truth_contexts)
        if rank is None:
            score, comment = 0.0, f"Top-{self.k} 未命中 ground truth"
        else:
            score, comment = 1.0 / rank, f"第一个命中排名={rank}, MRR={1.0 / rank:.4f}"

        return EvaluationResult(
            key=self.evaluation_name, score=score, value=score, comment=comment
        )


class ContextRecallEvaluator(LLMJudge):
    """
    Context Recall 评估器。

    定义：检索到的上下文覆盖 ground truth 答案要点的比例。
    取值：[0, 1]，1 表示全部要点被检索上下文覆盖。

    实现：让 LLM 从 ground truth 答案中抽取若干独立要点，
    逐条判断是否可由检索上下文推导，最终返回被覆盖要点占比。
    """

    @property
    def evaluation_name(self) -> str:
        return "context_recall"

    def _build_prompt(
        self,
        query: str,
        retrieved_docs: List[str],
        ground_truth_contexts: List[str],
        **_: Any,
    ) -> str:
        # 合并所有检索上下文为一段文本，供 LLM 判断
        context_text = "\n---\n".join(retrieved_docs) if retrieved_docs else "(无检索上下文)"
        gt_text = "\n".join(
            f"[{i}] {c}" for i, c in enumerate(ground_truth_contexts, 1)
        ) if ground_truth_contexts else "(无 ground truth context)"

        return f"""你是一个严谨的 RAG 评估专家。请评估【检索到的上下文】对【ground truth 要点】的覆盖程度。

【用户问题】
{query}

【ground truth 上下文要点】
{gt_text}

【检索到的上下文】
{context_text}

评估步骤：
1. 从 ground truth 上下文要点中抽取所有独立的事实陈述（每个要点一行）。
2. 逐条判断该要点是否能由【检索到的上下文】完全推导出来（可推导=1，不可推导=0）。
3. 计算 context_recall = 可推导要点数 / 总要点数。

请严格按以下格式输出（不要输出任何其他内容）：
SCORE: <0到1之间的浮点数>
REASON: <一句话说明覆盖情况，包含 可推导数/总数>
"""

    def _parse_response(self, text: str) -> Tuple[float, str]:
        score_match = re.search(r"SCORE:\s*([0-9]*\.?[0-9]+)", text, re.IGNORECASE)
        reason_match = re.search(r"REASON:\s*(.+)", text, re.IGNORECASE | re.DOTALL)

        if not score_match:
            raise ValueError(f"无法从 LLM 输出中解析 SCORE: {text[:200]}")

        score = max(0.0, min(1.0, float(score_match.group(1))))
        comment = reason_match.group(1).strip() if reason_match else "未提供评估理由"
        return score, comment


# =============================================================================
# 生成阶段评估器
# =============================================================================

class FaithfulnessEvaluator(LLMJudge):
    """
    Faithfulness（忠实度）评估器。

    定义：答案是否完全基于检索 context，不包含幻觉。
    取值：[0, 1]，1 表示答案完全可由 context 推导，0 表示答案与 context 完全无关。

    实现：将答案拆分为若干事实陈述，逐条判断是否可由 context 推导，
    返回可推导陈述占比。
    """

    @property
    def evaluation_name(self) -> str:
        return "faithfulness"

    def _build_prompt(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        **_: Any,
    ) -> str:
        context_text = "\n---\n".join(contexts) if contexts else "(无检索上下文)"

        return f"""你是一个严谨的 RAG 评估专家。请评估【答案】对【检索上下文】的忠实度（Faithfulness）。

忠实度定义：答案中的每一句事实陈述是否都能由检索上下文直接推导出来。
- 完全可推导：1.0（无幻觉）
- 部分可推导：0~1 之间
- 完全不可推导：0.0（纯幻觉）

【用户问题】
{query}

【检索上下文】
{context_text}

【待评估答案】
{answer}

评估步骤：
1. 将答案拆分为若干独立的事实陈述。
2. 逐条判断该陈述是否能由【检索上下文】直接推导（可推导=1，不可推导=0）。
3. 计算 faithfulness = 可推导陈述数 / 总陈述数。

请严格按以下格式输出（不要输出任何其他内容）：
SCORE: <0到1之间的浮点数>
REASON: <一句话说明，包含 可推导数/总数>
"""

    def _parse_response(self, text: str) -> Tuple[float, str]:
        score_match = re.search(r"SCORE:\s*([0-9]*\.?[0-9]+)", text, re.IGNORECASE)
        reason_match = re.search(r"REASON:\s*(.+)", text, re.IGNORECASE | re.DOTALL)

        if not score_match:
            raise ValueError(f"无法从 LLM 输出中解析 SCORE: {text[:200]}")

        score = max(0.0, min(1.0, float(score_match.group(1))))
        comment = reason_match.group(1).strip() if reason_match else "未提供评估理由"
        return score, comment


class AnswerCorrectnessEvaluator(LLMJudge):
    """
    Answer Correctness（答案正确性）评估器。

    定义：答案与 ground truth 答案的事实一致性。
    取值：[0, 1]，1 表示与 ground truth 完全一致，0 表示完全错误。

    实现：LLM 对比答案与 ground truth，从事实正确性、完整性、无矛盾三维度打分。
    """

    @property
    def evaluation_name(self) -> str:
        return "answer_correctness"

    def _build_prompt(
        self,
        query: str,
        answer: str,
        ground_truth: str,
        **_: Any,
    ) -> str:
        return f"""你是一个严谨的 RAG 评估专家。请评估【待评估答案】与【ground truth 答案】的事实一致性。

评估维度：
1. 事实正确性：待评估答案中的事实是否与 ground truth 一致。
2. 完整性：待评估答案是否覆盖 ground truth 的关键信息。
3. 无矛盾：待评估答案是否与 ground truth 存在矛盾。

打分规则：
- 1.0：与 ground truth 事实完全一致，覆盖全部关键信息，无矛盾
- 0.7~0.9：主要事实一致，遗漏少量次要信息
- 0.4~0.6：部分事实一致，存在遗漏或轻微矛盾
- 0.1~0.3：大部分事实不一致或严重遗漏
- 0.0：完全错误或与 ground truth 矛盾

【用户问题】
{query}

【ground truth 答案】
{ground_truth}

【待评估答案】
{answer}

请严格按以下格式输出（不要输出任何其他内容）：
SCORE: <0到1之间的浮点数>
REASON: <一句话说明扣分原因>
"""

    def _parse_response(self, text: str) -> Tuple[float, str]:
        score_match = re.search(r"SCORE:\s*([0-9]*\.?[0-9]+)", text, re.IGNORECASE)
        reason_match = re.search(r"REASON:\s*(.+)", text, re.IGNORECASE | re.DOTALL)

        if not score_match:
            raise ValueError(f"无法从 LLM 输出中解析 SCORE: {text[:200]}")

        score = max(0.0, min(1.0, float(score_match.group(1))))
        comment = reason_match.group(1).strip() if reason_match else "未提供评估理由"
        return score, comment


class AnswerRelevancyEvaluator(LLMJudge):
    """
    Answer Relevancy（答案相关性）评估器。

    定义：答案与用户问题的相关程度。
    取值：[0, 1]，1 表示答案完全针对问题，0 表示答案与问题无关。

    实现：LLM 从问题覆盖度、答案聚焦度、无冗余三维度打分。
    """

    @property
    def evaluation_name(self) -> str:
        return "answer_relevancy"

    def _build_prompt(
        self,
        query: str,
        answer: str,
        **_: Any,
    ) -> str:
        return f"""你是一个严谨的 RAG 评估专家。请评估【答案】与【用户问题】的相关程度。

评估维度：
1. 问题覆盖度：答案是否直接回答了用户问题。
2. 答案聚焦度：答案是否围绕问题展开，不偏题。
3. 无冗余：答案是否避免与问题无关的冗余信息。

打分规则：
- 1.0：完全针对问题，聚焦无冗余
- 0.7~0.9：基本回答了问题，少量偏题
- 0.4~0.6：部分回答了问题，存在明显偏题
- 0.1~0.3：大部分内容与问题无关
- 0.0：完全未回答问题或答非所问

【用户问题】
{query}

【待评估答案】
{answer}

请严格按以下格式输出（不要输出任何其他内容）：
SCORE: <0到1之间的浮点数>
REASON: <一句话说明扣分原因>
"""

    def _parse_response(self, text: str) -> Tuple[float, str]:
        score_match = re.search(r"SCORE:\s*([0-9]*\.?[0-9]+)", text, re.IGNORECASE)
        reason_match = re.search(r"REASON:\s*(.+)", text, re.IGNORECASE | re.DOTALL)

        if not score_match:
            raise ValueError(f"无法从 LLM 输出中解析 SCORE: {text[:200]}")

        score = max(0.0, min(1.0, float(score_match.group(1))))
        comment = reason_match.group(1).strip() if reason_match else "未提供评估理由"
        return score, comment


class HallucinationRateEvaluator(LLMJudge):
    """
    Hallucination Rate（幻觉率）评估器。

    定义：答案中幻觉内容占比，与 Faithfulness 互补。
    取值：[0, 1]，0 表示无幻觉，1 表示完全幻觉。

    实现：LLM 将答案拆分为事实陈述，逐条判断是否可由 context 推导，
    返回不可推导陈述占比（即幻觉率 = 1 - faithfulness 的细化版本）。
    """

    @property
    def evaluation_name(self) -> str:
        return "hallucination_rate"

    def _build_prompt(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        **_: Any,
    ) -> str:
        context_text = "\n---\n".join(contexts) if contexts else "(无检索上下文)"

        return f"""你是一个严谨的 RAG 评估专家。请评估【答案】中幻觉内容的占比（Hallucination Rate）。

幻觉定义：答案中无法由检索上下文直接推导的事实陈述。
- 0.0：无幻觉（所有陈述均可由 context 推导）
- 1.0：完全幻觉（所有陈述均无法由 context 推导）

【用户问题】
{query}

【检索上下文】
{context_text}

【待评估答案】
{answer}

评估步骤：
1. 将答案拆分为若干独立的事实陈述。
2. 逐条判断该陈述是否能由【检索上下文】直接推导（可推导=0幻觉，不可推导=1幻觉）。
3. 计算 hallucination_rate = 不可推导陈述数 / 总陈述数。

请严格按以下格式输出（不要输出任何其他内容）：
SCORE: <0到1之间的浮点数>
REASON: <一句话说明，包含 幻觉陈述数/总陈述数>
"""

    def _parse_response(self, text: str) -> Tuple[float, str]:
        score_match = re.search(r"SCORE:\s*([0-9]*\.?[0-9]+)", text, re.IGNORECASE)
        reason_match = re.search(r"REASON:\s*(.+)", text, re.IGNORECASE | re.DOTALL)

        if not score_match:
            raise ValueError(f"无法从 LLM 输出中解析 SCORE: {text[:200]}")

        score = max(0.0, min(1.0, float(score_match.group(1))))
        comment = reason_match.group(1).strip() if reason_match else "未提供评估理由"
        return score, comment


# =============================================================================
# 评估器分组（便于 run_eval 按模式选择）
# =============================================================================

def _make_retrieval_evaluators() -> List[RunEvaluator]:
    return [
        HitRateEvaluator(k=DEFAULT_RETRIEVAL_K),
        MRREvaluator(k=DEFAULT_RETRIEVAL_K),
        ContextRecallEvaluator(),
    ]


def _make_generation_evaluators() -> List[RunEvaluator]:
    return [
        FaithfulnessEvaluator(),
        AnswerCorrectnessEvaluator(),
        AnswerRelevancyEvaluator(),
        HallucinationRateEvaluator(),
    ]


RETRIEVAL_EVALUATORS: List[RunEvaluator] = _make_retrieval_evaluators()
"""检索阶段评估器集合（3 个）。"""

GENERATION_EVALUATORS: List[RunEvaluator] = _make_generation_evaluators()
"""生成阶段评估器集合（4 个）。"""

ALL_EVALUATORS: List[RunEvaluator] = RETRIEVAL_EVALUATORS + GENERATION_EVALUATORS
"""全链路评估器集合（7 个）。"""


# =============================================================================
# 便捷函数：批量异步执行评估器
# =============================================================================

async def run_evaluators_async(
    evaluators: List[RunEvaluator],
    inputs: dict,
) -> List[EvaluationResult]:
    """
    并发执行多个评估器。

    Args:
        evaluators: 评估器列表。
        inputs:     评估输入（传给每个评估器的 ajudge / aevaluate_run）。

    Returns:
        EvaluationResult 列表，顺序与 evaluators 一致。
    """
    # 优先使用 ajudge（直接接受 dict 输入，避免构造 Run/Example 的开销）
    tasks = []
    for ev in evaluators:
        if isinstance(ev, LLMJudge):
            tasks.append(ev.ajudge(**inputs))
        else:
            # 兜底：非 LLMJudge 子类，构造空 Run/Example 调用协议方法
            tasks.append(ev.aevaluate_run(run=Run(id=uuid.uuid4(), name="eval", inputs=inputs, outputs=inputs), example=None))
    results = await asyncio.gather(*tasks, return_exceptions=True)

    final: List[EvaluationResult] = []
    for ev, res in zip(evaluators, results):
        if isinstance(res, Exception):
            logger.error(f"[{getattr(ev, 'evaluation_name', ev)}] 评估异常: {res}")
            final.append(EvaluationResult(
                key=getattr(ev, "evaluation_name", "unknown"),
                score=0.0, value=0.0,
                comment=f"评估异常: {res}",
            ))
        else:
            final.append(res)
    return final


def docs_to_texts(docs: List[Any]) -> List[str]:
    """
    将 Document 列表（或字符串列表）统一转为字符串列表。

    兼容 langchain_core.documents.Document 与裸字符串。
    """
    result: List[str] = []
    for doc in docs:
        if isinstance(doc, Document):
            result.append(doc.page_content)
        elif isinstance(doc, str):
            result.append(doc)
        else:
            result.append(str(doc))
    return result
