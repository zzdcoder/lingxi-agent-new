"""
评估数据集管理模块

职责：
    1. 定义评估样例数据结构 EvalExample（query + ground_truth_answer + ground_truth_contexts）
    2. 提供从 JSON 文件加载评估数据集的能力
    3. 提供将数据集上传到 LangSmith 平台的能力（使用 langsmith.Client）
    4. 内置示例数据，便于开箱即用验证

设计原则：
    - 类型安全：使用 pydantic BaseModel 定义数据结构，自动校验字段
    - 健壮性：JSON 加载失败时抛出明确异常，避免后续评估流程静默失败
    - 复用性：示例数据与加载逻辑分离，便于扩展为从数据库 / 远程接口加载
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# 数据结构定义
# =============================================================================

class EvalExample(BaseModel):
    """
    单条评估样例。

    Attributes:
        query:                  用户原始问题
        ground_truth_answer:    标准答案（人工标注或权威来源）
        ground_truth_contexts:  标准上下文文档列表（用于检索阶段评估）
        query_id:               可选的样例唯一标识，便于追溯
    """

    query: str = Field(..., description="用户原始问题")
    ground_truth_answer: str = Field(..., description="标准答案")
    ground_truth_contexts: List[str] = Field(
        default_factory=list,
        description="标准上下文文档列表（用于检索阶段评估）",
    )
    query_id: Optional[str] = Field(
        default=None, description="样例唯一标识，便于追溯"
    )

    # 兼容多种字段命名（gt_answer / answer / expected）
    model_config = {"extra": "ignore"}

    @field_validator("query", "ground_truth_answer")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("query 与 ground_truth_answer 不能为空")
        return v.strip()

    @field_validator("ground_truth_contexts")
    @classmethod
    def _dedup_contexts(cls, v: List[str]) -> List[str]:
        # 去重并去除空串，保证 ground truth contexts 干净
        seen, result = set(), []
        for ctx in v:
            ctx = (ctx or "").strip()
            if ctx and ctx not in seen:
                seen.add(ctx)
                result.append(ctx)
        return result


class EvalDataset(BaseModel):
    """评估数据集容器，便于整体校验与序列化。"""

    name: str = Field(default="rag-eval-dataset", description="数据集名称")
    description: str = Field(default="", description="数据集描述")
    examples: List[EvalExample] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self):
        return iter(self.examples)


# =============================================================================
# JSON 加载
# =============================================================================

def load_dataset(json_path: str | Path) -> List[EvalExample]:
    """
    从 JSON 文件加载评估数据集。

    支持两种 JSON 结构：
        1. 列表结构：[{"query": "...", "ground_truth_answer": "...", ...}, ...]
        2. 对象结构：{"name": "...", "examples": [...]}

    Args:
        json_path: JSON 文件路径。

    Returns:
        EvalExample 列表。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError:        JSON 格式错误或样例校验失败。
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"评估数据集文件不存在: {path}")

    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"评估数据集 JSON 解析失败: {path}, 错误: {e}") from e

    # 统一为列表结构
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict) and "examples" in raw:
        items = raw["examples"]
    else:
        raise ValueError(
            f"评估数据集 JSON 结构不合法，应为列表或含 'examples' 字段的对象: {path}"
        )

    if not items:
        raise ValueError(f"评估数据集为空: {path}")

    examples: List[EvalExample] = []
    for idx, item in enumerate(items, 1):
        try:
            examples.append(EvalExample(**item))
        except Exception as e:
            raise ValueError(
                f"评估数据集第 {idx} 条样例校验失败: {e}"
            ) from e

    logger.info(f"评估数据集加载完成: {path}, 样例数: {len(examples)}")
    return examples


# =============================================================================
# LangSmith Dataset 创建
# =============================================================================

def create_langsmith_dataset(
    examples: List[EvalExample],
    dataset_name: str = "rag-eval-dataset",
    description: str = "RAG 评估数据集（lingxi-agent）",
    client: Optional[Any] = None,
) -> Any:
    """
    在 LangSmith 平台创建（或复用）数据集，并写入样例。

    Args:
        examples:     EvalExample 列表。
        dataset_name: LangSmith 数据集名称。同名数据集已存在时复用。
        description:  数据集描述。
        client:       可选的 langsmith.Client 实例。为 None 时自动创建。

    Returns:
        LangSmith Dataset 对象。

    Raises:
        ImportError: 未安装 langsmith 或未配置 LANGSMITH_API_KEY。
        Exception:   LangSmith API 调用失败。
    """
    from langsmith import Client

    ls_client = client or Client()

    # 复用同名数据集，避免重复创建导致脏数据
    try:
        dataset = ls_client.read_dataset(dataset_name=dataset_name)
        logger.info(f"复用已有 LangSmith 数据集: {dataset_name} (id={dataset.id})")
    except Exception:
        dataset = ls_client.create_dataset(
            dataset_name=dataset_name,
            description=description,
        )
        logger.info(f"创建 LangSmith 数据集: {dataset_name} (id={dataset.id})")

    # 写入样例。input 为 query，output 为标准答案 + 标准上下文
    for idx, ex in enumerate(examples, 1):
        ls_client.create_example(
            request_id=ex.query_id or f"eval-{idx}",
            inputs={"query": ex.query},
            outputs={
                "ground_truth_answer": ex.ground_truth_answer,
                "ground_truth_contexts": ex.ground_truth_contexts,
            },
            dataset_id=dataset.id,
        )

    logger.info(
        f"LangSmith 数据集样例写入完成: {dataset_name}, 样例数: {len(examples)}"
    )
    return dataset


# =============================================================================
# 从 LangSmith 拉取数据集
# =============================================================================

def fetch_dataset_from_langsmith(
    dataset_name: str,
    client: Optional[Any] = None,
) -> List[EvalExample]:
    """
    从 LangSmith 平台拉取已有数据集，转换为 EvalExample 列表。

    约定 LangSmith Example 的结构（与 create_langsmith_dataset 写入格式一致）：
        inputs:  {"query": str}
        outputs: {"ground_truth_answer": str, "ground_truth_contexts": List[str]}

    Args:
        dataset_name: LangSmith 数据集名称。
        client:       可选的 langsmith.Client 实例。为 None 时自动创建。

    Returns:
        EvalExample 列表。

    Raises:
        ImportError:  未安装 langsmith。
        LookupError:  数据集不存在。
        ValueError:   样例字段缺失或校验失败。
    """
    from langsmith import Client

    ls_client = client or Client()

    # 读取数据集（不存在时抛出明确异常）
    try:
        dataset = ls_client.read_dataset(dataset_name=dataset_name)
    except Exception as e:
        raise LookupError(
            f"LangSmith 数据集不存在或读取失败: {dataset_name}, 错误: {e}"
        ) from e

    logger.info(f"开始拉取 LangSmith 数据集: {dataset_name} (id={dataset.id})")

    examples: List[EvalExample] = []
    failed_count = 0

    # list_examples 返回迭代器，逐条转换为 EvalExample
    for raw_example in ls_client.list_examples(dataset_id=dataset.id):
        try:
            inputs = dict(raw_example.inputs or {})
            outputs = dict(raw_example.outputs or {})

            query = inputs.get("query", "")
            gt_answer = outputs.get("ground_truth_answer", "")
            gt_contexts = outputs.get("ground_truth_contexts", [])
            query_id = str(raw_example.id) if hasattr(raw_example, "id") else None

            example = EvalExample(
                query=query,
                ground_truth_answer=gt_answer,
                ground_truth_contexts=gt_contexts if isinstance(gt_contexts, list) else [],
                query_id=query_id,
            )
            examples.append(example)
        except Exception as e:
            failed_count += 1
            logger.warning(
                f"LangSmith 样例转换失败 (id={getattr(raw_example, 'id', '?')}): {e}"
            )

    if not examples:
        raise ValueError(
            f"LangSmith 数据集 '{dataset_name}' 拉取后无可用样例，"
            f"失败 {failed_count} 条。请检查数据集格式是否符合约定。"
        )

    logger.info(
        f"LangSmith 数据集拉取完成: {dataset_name}, "
        f"成功 {len(examples)} 条, 失败 {failed_count} 条"
    )
    return examples


def fetch_dataset_from_langsmith_by_id(
    dataset_id: str,
    client: Optional[Any] = None,
) -> List[EvalExample]:
    """
    通过数据集 ID 从 LangSmith 拉取样例（备用入口，用于重名场景）。

    Args:
        dataset_id: LangSmith 数据集 UUID 字符串。
        client:     可选的 langsmith.Client 实例。

    Returns:
        EvalExample 列表。
    """
    from langsmith import Client

    ls_client = client or Client()

    logger.info(f"开始通过 ID 拉取 LangSmith 数据集: {dataset_id}")

    examples: List[EvalExample] = []
    failed_count = 0

    for raw_example in ls_client.list_examples(dataset_id=dataset_id):
        try:
            inputs = dict(raw_example.inputs or {})
            outputs = dict(raw_example.outputs or {})

            example = EvalExample(
                query=inputs.get("query", ""),
                ground_truth_answer=outputs.get("ground_truth_answer", ""),
                ground_truth_contexts=outputs.get("ground_truth_contexts", []) or [],
                query_id=str(raw_example.id) if hasattr(raw_example, "id") else None,
            )
            examples.append(example)
        except Exception as e:
            failed_count += 1
            logger.warning(f"样例转换失败: {e}")

    if not examples:
        raise ValueError(
            f"LangSmith 数据集 (id={dataset_id}) 拉取后无可用样例，"
            f"失败 {failed_count} 条"
        )

    logger.info(
        f"LangSmith 数据集拉取完成 (id={dataset_id}), "
        f"成功 {len(examples)} 条, 失败 {failed_count} 条"
    )
    return examples


def list_langsmith_datasets(client: Optional[Any] = None) -> List[Dict[str, Any]]:
    """
    列出 LangSmith 账号下所有数据集，便于查找可用数据集名称。

    Args:
        client: 可选的 langsmith.Client 实例。

    Returns:
        数据集摘要列表，每项包含 id、name、description、created_at。
    """
    from langsmith import Client

    ls_client = client or Client()

    datasets = []
    for ds in ls_client.list_datasets():
        datasets.append({
            "id": str(ds.id),
            "name": ds.name,
            "description": getattr(ds, "description", "") or "",
            "created_at": str(getattr(ds, "created_at", "")),
        })

    logger.info(f"LangSmith 账号下共 {len(datasets)} 个数据集")
    return datasets


# =============================================================================
# 内置示例数据
# =============================================================================

SAMPLE_EXAMPLES: List[EvalExample] = [
    EvalExample(
        query_id="sample-1",
        query="TouchVue 的核心功能有哪些？",
        ground_truth_answer=(
            "TouchVue 是一个面向移动端的低代码可视化搭建平台，核心功能包括："
            "1) 拖拽式页面搭建，支持组件自由组合与布局；"
            "2) 丰富的组件库，涵盖表单、图表、列表等业务组件；"
            "3) 数据源绑定，支持 REST API 与静态数据；"
            "4) 实时预览与发布，所见即所得；"
            "5) 多端适配，一套配置同时生成 H5/小程序页面。"
        ),
        ground_truth_contexts=[
            "TouchVue 是面向移动端的低代码可视化搭建平台，提供拖拽式页面搭建、"
            "丰富组件库、数据源绑定、实时预览发布、多端适配等核心能力。",
        ],
    ),
    EvalExample(
        query_id="sample-2",
        query="BM25 与向量检索混合时，为什么要使用 RRF 融合？",
        ground_truth_answer=(
            "BM25 与向量检索的分数尺度完全不同（BM25 通常为 10~30，向量相似度为 0~1），"
            "直接加权求和会让分数尺度大的一路主导结果。"
            "RRF 只关心文档的排名位置而非绝对分数，天然消除尺度差异，"
            "无需训练、零超参（k=60 为通用最优值），实现简单且效果稳定。"
        ),
        ground_truth_contexts=[
            "RRF（Reciprocal Rank Fusion）只利用排名的相对位置，不依赖分数绝对值，"
            "天然适配不同检索方式的分数尺度差异，k=60 是 Cormack 等人在 TREC 实验中验证的通用最优值。",
            "BM25 擅长精确匹配关键词，向量检索擅长语义理解，混合检索通过 RRF 融合两者优势。",
        ],
    ),
    EvalExample(
        query_id="sample-3",
        query="语义缓存的相似度阈值如何设置？",
        ground_truth_answer=(
            "语义缓存基于 Qdrant 余弦相似度实现，阈值越高越严格（默认 0.92）。"
            "阈值过高会导致缓存命中率低，失去缓存价值；"
            "阈值过低会产生语义近但答案不一致的误命中。"
            "建议业务知识库初始设置为 0.90~0.95，并根据实际命中率与误命中率动态调整。"
        ),
        ground_truth_contexts=[
            "语义缓存基于 Qdrant 余弦相似度匹配历史查询，cache_similarity_threshold 默认 0.92，"
            "阈值越高越严格，过低会导致语义近但答案不一致的误命中。",
        ],
    ),
]
"""内置示例数据，3 条覆盖产品功能、检索原理、缓存配置的样例。

用于开箱即用验证评估流程，无需准备 JSON 文件即可运行 run_eval。
"""
