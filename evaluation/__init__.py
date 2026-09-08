"""
LangSmith RAG 评估模块

提供 RAG 全链路评估能力，覆盖检索阶段与生成阶段共 7 个核心指标：

检索阶段（3 个）：
    - HitRateEvaluator:        Hit Rate@K，Top-K 是否命中 ground truth 文档
    - MRREvaluator:            MRR，第一个命中 ground truth 文档的排名倒数
    - ContextRecallEvaluator:  Context Recall，检索上下文覆盖 ground truth 要点的比例

生成阶段（4 个）：
    - FaithfulnessEvaluator:        答案是否完全基于 context，不包含幻觉
    - AnswerCorrectnessEvaluator:   答案与 ground truth 答案的一致性
    - AnswerRelevancyEvaluator:     答案与用户问题的相关程度
    - HallucinationRateEvaluator:   幻觉内容占比（与 Faithfulness 互补）

数据集管理：
    - EvalExample:           评估样例数据结构
    - load_dataset:          从 JSON 加载评估数据集
    - create_langsmith_dataset: 创建 LangSmith Dataset

评估入口：
    - run_eval.main:         命令行评估入口，支持 retrieval/generation/full 三种模式
"""

from evaluation.dataset import (
    EvalExample,
    load_dataset,
    create_langsmith_dataset,
    fetch_dataset_from_langsmith,
    fetch_dataset_from_langsmith_by_id,
    list_langsmith_datasets,
)
from evaluation.evaluators import (
    LLMJudge,
    HitRateEvaluator,
    MRREvaluator,
    ContextRecallEvaluator,
    FaithfulnessEvaluator,
    AnswerCorrectnessEvaluator,
    AnswerRelevancyEvaluator,
    HallucinationRateEvaluator,
    RETRIEVAL_EVALUATORS,
    GENERATION_EVALUATORS,
    ALL_EVALUATORS,
)

__all__ = [
    # 数据集
    "EvalExample",
    "load_dataset",
    "create_langsmith_dataset",
    "fetch_dataset_from_langsmith",
    "fetch_dataset_from_langsmith_by_id",
    "list_langsmith_datasets",
    # 基类
    "LLMJudge",
    # 检索阶段评估器
    "HitRateEvaluator",
    "MRREvaluator",
    "ContextRecallEvaluator",
    # 生成阶段评估器
    "FaithfulnessEvaluator",
    "AnswerCorrectnessEvaluator",
    "AnswerRelevancyEvaluator",
    "HallucinationRateEvaluator",
    # 评估器分组
    "RETRIEVAL_EVALUATORS",
    "GENERATION_EVALUATORS",
    "ALL_EVALUATORS",
]
