"""
Agent 图状态定义

使用 LangGraph 的 StateGraph 管理主图状态。状态字段按职责分块：
- 会话上下文：会话/用户/模型/消息
- 意图识别：多标签意图 + 理由 + 置信度
- 知识库问答：检索文档与 RAG 分支产出
- 任务执行：任务记录 / 工具调用 / 审批相关
- 输出：汇总节点合并后的最终回答
"""
from typing import TypedDict, Annotated, Optional, Any
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    # ---- 会话上下文 ----
    conversation_id: str                 # 会话 ID（= LangGraph thread_id）
    user_id: Optional[str]               # 用户 ID
    username: Optional[str]              # 用户名（用于知识库权限过滤）
    model: str                           # 模型名
    messages: Annotated[list, add_messages]   # 消息列表（含历史 + 当前输入）

    # ---- 意图识别（多标签） ----
    intents: list                        # 多意图列表，如 ["task","knowledge_base"] / ["chat"]
    intent_reason: Optional[str]         # LLM 分类理由（可观测）
    intent_confidence: Optional[float]   # 置信度
    last_intents: Optional[list]         # 上一轮意图（会话内），供短追问继承（§16）
    slash_command: Optional[str]         # 命中的斜杠命令名（§16 规则层短路）
    slash_args: Optional[str]            # 斜杠命令参数
    slash_handoff: Optional[str]         # 斜杠命令本地回执文案（免 LLM 直接输出）
    query_embedding: Optional[list]      # query 稠密向量（知识库分支预计算，供意图层复用，§16）

    # ---- 知识库问答 ----
    context_docs: list                   # 检索到的文档
    context_text: Optional[str]          # 格式化后的上下文
    cached_answer: Optional[str]         # 语义缓存命中时直接使用
    rag_answer: Optional[str]            # 知识库分支产出（并行场景下与任务分支隔离）

    # ---- 任务执行 ----
    task_execution_id: Optional[str]     # 任务执行记录 ID
    tool_calls: list                     # 本次任务的工具调用序列（审计）
    tool_results: list                   # 工具执行结果
    pending_write: Optional[dict]        # 待审批的写操作（工具名/表/参数）
    approval_required: bool              # 是否需要审批
    approval_status: Optional[str]       # pending / approved / rejected
    approval_id: Optional[str]           # 审批单 ID

    # ---- 输出 ----
    task_answer: Optional[str]           # 任务分支产出（含审批后结果，供汇总节点合并）
    final_response: Optional[str]        # 汇总节点合并后的最终回答（供落库与回溯）
    error: Optional[str]                 # 错误信息
