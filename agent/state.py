"""
Agent 图状态定义

使用 LangGraph 的 StateGraph 管理主图状态。状态字段按职责分块：
- 会话上下文：会话/用户/模型/消息
- 意图识别：多标签意图 + 理由 + 置信度
- 知识库问答：检索文档与 RAG 分支产出
- 任务执行：任务记录 / 工具调用 / 审批相关
- 输出：汇总节点合并后的最终回答
"""
from typing import TypedDict, Annotated, Optional
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
    # 已弃用（§18.5）：向量改由 `config["configurable"]` 在同一次 run 内传递，
    # 不再写入 state —— 1024 维向量进检查点会被逐轮序列化落库，且上一轮残留
    # 会污染下一轮的意图判定。保留声明仅为兼容老检查点中的同名字段。
    query_embedding: Optional[list]

    # ---- 答案缓存准入（§18.2，由入口节点每轮覆盖写入） ----
    # 语义「答案缓存」命中会跳过整条执行链路（不检索、不生成、不执行工具、
    # 不留审计），因此必须先由意图判定它的准入结果，再决定是否去查缓存。
    # 这两项与 executed_branches 同理，由 intent_router_node 每轮无条件覆盖，
    # 避免检查点里上一轮的策略残留影响本轮。
    cache_policy: Optional[str]          # allow / short / deny
    cache_ttl_seconds: Optional[int]     # 本轮答案缓存可用的 TTL（deny 时为 0）

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

    # ---- 本轮分支标记（§17.2 跨轮状态隔离） ----
    # 由入口节点 intent_router_node 每轮**无条件覆盖**写入，值为本轮**计划执行**
    # 的分支名列表（"knowledge" / "task_agent"）。
    #
    # 为什么需要它：LangGraph 检查点（挂 MySQLAsyncSaver，thread_id=conversation_id）
    # 是**按 thread 累积的 channel 存储**——节点只写自己产出的字段，未执行的节点
    # 对应字段会**保留上一轮的旧值**。例如本轮只路由 task 单分支时，上一轮
    # knowledge/chat 分支写入的 rag_answer 仍在 channel 中，merge_node 若按
    # 「字段非空」判断，会把**与本轮无关的历史内容**当成知识库结论去做 LLM 合并，
    # 既白付一次 LLM（实测 17.4s），又产生串味幻觉。
    #
    # 因此 merge 改用本字段判断「本轮到底跑过哪些分支」，彻底与历史残留解耦。
    # 注意：**不要**改成清空 rag_answer/task_answer 等业务字段——并行审批中断
    # 场景下 rag_answer 必须保留给 resume 后合并（§17.2.4 场景论证）。
    executed_branches: list
