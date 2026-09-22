"""
任务执行节点

基于 langchain.agents.create_agent 构建任务 Agent，绑定数据库工具（阶段 2：
只读工具 + insert；update/delete 待阶段 3 挂载 HITL 审批后接入）。

流程：
1. 创建任务执行记录（status=running，记录路由意图）；
2. 构建 DbToolExecutor 并绑定工具；
3. Agent 执行任务，产出人类可读的执行报告（task_answer）；
4. 完成/失败时更新审计记录（tool_calls 脱敏摘要 + status + result/error）。

安全要求（设计文档 §5.5、§11）：
- 工具参数化执行，白名单由 DbToolExecutor 强制校验；
- 审计只记录参数摘要与结果摘要，不落完整参数值。
"""
import asyncio
import logging
import time
import uuid
from typing import Any, List, Optional

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphInterrupt, GraphRecursionError
from langgraph.types import Command

from core.config import settings
from core.llm import get_chat_model
from prompt.prompt_storage import TASK_AGENT_SYSTEM_PROMPT
from models.task_model import TaskExecution
from rag.memory_mysql import MySQLChatMessageHistory
from agent.state import AgentState
from agent.stream_emitter import StreamEmitter, extract_reasoning, extract_text
from agent.streaming import get_sse_queue, put_content, put_status, put_tool
from agent.tools.db_tools import DbToolExecutor
from agent.tools.knowledge_tool import KnowledgeToolExecutor
from agent.tools.manager import bind_active_tools
from tools.clarify_tool import ASK_USER_TOOL_NAME
from agent.observability import (
    K_TASK_CLARIFY_ANSWER_NO_WRITE,
    K_TASK_HISTORY_INJECTED,
    K_TASK_HISTORY_LOAD_FAILED,
    K_TASK_RESUME_PAYLOAD_MISMATCH,
    K_TASK_RESUME_SECONDARY_INTERRUPT,
    K_TASK_STALE_RUNNING_CLOSED,
    K_TASK_STALE_RUNNING_REJECTED,
    get_trace_id,
    metrics,
    node_trace,
)

logger = logging.getLogger(__name__)

# 模型-工具循环的递归上限（langgraph 每执行一个节点计 1 步，模型/工具各占步数）。
# create_agent 无 max_iterations 参数，超限保护统一走 langgraph 的 recursion_limit
#（默认 9999，此处收敛为小值），达到上限抛 GraphRecursionError 由节点层转友好错误。
# 按每轮 1 次模型 + N 个工具估算，100 步约等于 25~50 轮往返，足够正常任务且防死循环。
# 阶段 4：改为配置驱动（agent_graph_recursion_limit），保留该常量作为取不到配置时的兜底。
AGENT_MAX_RECURSION_LIMIT: int = 100


def _recursion_limit() -> int:
    """读取递归上限（配置优先，非法值回落常量）。"""
    value = getattr(settings, "agent_graph_recursion_limit", None)
    return value if isinstance(value, int) and value > 0 else AGENT_MAX_RECURSION_LIMIT


# =============================================================================
# §21 任务分支流式输出（ainvoke → astream）
# =============================================================================
#
# 改造前：任务分支是全链路**唯一非流式**的分支。`chat_node` 与 `knowledge_node`
# 都走 `chain.astream` 逐 token 推送，只有最耗时的任务分支「跑完才说话」——
# 多轮工具调用动辄几十秒，用户界面全程空白，只能靠心跳维持连接。
#
# 改造后：`agent.astream(..., stream_mode=["messages", "updates"])` 逐帧产出。
#
# **两个必须处理的坑**（都踩过，见 §21.4）：
#
# 1. `stream_mode="messages"` 会把**所有**节点产出的消息都抖出来，包括
#    `tools` 节点的 `ToolMessage`。ToolMessage 的 content 是**数据库原始结果**
#    （整行数据、SQL 回显），直接推给用户既是体验灾难也是数据泄漏。
#    故必须按 `meta["langgraph_node"] == NODE_MODEL` 过滤，只取模型节点的输出。
# 2. `updates` 模式用于捕获工具执行**结束**事件（`messages` 模式下的 ToolMessage
#    恰好就是它），两者配合才能给出「工具开始 / 工具结束」的完整进度。
#
# 失败降级：`astream` 的任何非 GraphInterrupt 异常（老版本 langgraph 不支持该
# stream_mode、消息结构变化等）都回落到 `ainvoke` 一次性执行，保证「流式是增强、
# 不是必需」——最坏情况退回改造前行为，而不是任务跑不起来。

# create_agent 内部节点名（factory.py 中 NODE_MODEL / NODE_TOOLS 的取值）。
# 硬编码而非 import：这两个名字来自 langchain 库内部常量，跨版本可能改名，
# 用字面量 + 过滤兜底（取不到 meta 时不推）比 import 更稳。
NODE_MODEL = "model"
NODE_TOOLS = "tools"


# =============================================================================
# §22 恢复载荷与中断类型对齐（修复「KeyError: 'decisions'」）
# =============================================================================
#
# 生产 trace=07dbbe8d604a4106：用户「删除用户dada」→ ask_user 追问 3 问 →
# 用户答完恢复 → 模型重新规划后**发出 delete_data**（走到审批门口）→
# 却抛出 `KeyError: 'decisions'`，整轮作废，用户看到「请提供筛选条件」。
#
# 根因（已最小复现，见 scripts/smoke_clarify_hitl_collision.py）：
# 内层任务 Agent 上**同时存在两类 interrupt()**：
#
#   ① `ask_user` 工具内部 `interrupt({"type":"clarification",...})`
#      —— 期望恢复载荷 `{"answers": [...]}` / `{"canceled": True}`
#   ② `HumanInTheLoopMiddleware.after_model` 的 `interrupt(hitl_request)`
#      —— 期望恢复载荷 `{"decisions": [{"type": "approve"|"reject"}]}`
#
# 而 langchain 的 HITL 中间件是**下标取值、无兜底**
#（`human_in_the_loop.py:435`：`decisions = interrupt(hitl_request)["decisions"]`），
# 把 ① 的载荷投给 ② 必然 `KeyError`。
#
# 关键陷阱：**悬挂中断的类型会随轮次变化** —— clarify 中断被消费后，
# 模型若发出写操作，新的悬挂中断就变成 HITL 形态；而本轮载荷常量仍是
# `{"answers": ...}`。所以载荷**不能**按「本轮开头是什么中断」假定，
# 必须在**投递前一刻**重新读悬挂中断并与载荷配对。
#
# 本节提供两个设施：
#   - `ResumeMismatchError`：载荷/中断类型不匹配的**确定性**错误
#     （修复 2：不得回落重跑 —— 重投同一 Command 必然再炸）；
#   - `_classify_resume_payload` / `_classify_interrupt` / `_match_resume_target`：
#     载荷与中断的配对判据（修复 1/4 共用）。


class ResumeMismatchError(RuntimeError):
    """
    §22 恢复载荷与悬挂中断类型不匹配（确定性错误，**不得回落重跑**）。

    与「环境类故障」（langgraph 版本不支持 stream_mode、消息结构变化）不同，
    载荷格式不匹配**重投同一 Command 必然得到同样的异常**，
    回落 ainvoke 只会把同一个错误再抛一次（实测见
    `scripts/diag_keyerror_timing.py` Q2.1）。故必须显式排除在回落白名单之外，
    并在上层转成用户可读提示。
    """


# 载荷 / 中断的两种形态标识
PAYLOAD_CLARIFY = "clarify"   # ask_user 内生中断（期望 answers / canceled）
PAYLOAD_HITL = "hitl"         # HITL 中间件中断（期望 decisions）


def _classify_resume_payload(resume_payload: Any) -> str:
    """
    判定恢复载荷属于哪一类中断（§22 修复 1/4 的配对判据）。

    判据（按优先级）：
      - `decisions` 键存在 -> HITL（写操作审批）
      - `answers` / `canceled` 键存在 -> clarify（追问）
      - 其余 -> 未知（返回空串，调用方按「不做配对校验」处理，
        保持对旧载荷形态的兼容，不引入新的拒绝路径）

    :param resume_payload: approval_service / clarify_service 下发的恢复载荷
    :return: PAYLOAD_HITL / PAYLOAD_CLARIFY / ""（未知）
    """
    if not isinstance(resume_payload, dict):
        return ""
    if "decisions" in resume_payload:
        return PAYLOAD_HITL
    if "answers" in resume_payload or "canceled" in resume_payload:
        return PAYLOAD_CLARIFY
    return ""


def _classify_interrupt(interrupt_obj: Any) -> str:
    """
    判定一个悬挂中断属于哪一类（§22）。

    判据基于中断的 `value` 形态：
      - `{"type": "clarification", ...}` -> clarify
      - 含 `action_requests` -> HITL
      - 其余 -> 未知（空串）

    `value` 取不到（None / 非 dict）时返回空串 —— 宁可「无法判定」也不要
    误判成某一类，否则会把本来正确的投递拦下。
    """
    value = getattr(interrupt_obj, "value", None)
    if not isinstance(value, dict):
        return ""
    if value.get("type") == "clarification":
        return PAYLOAD_CLARIFY
    if "action_requests" in value:
        return PAYLOAD_HITL
    return ""


# §23 修复 1：`HumanInTheLoopMiddleware` 对 decisions 的下标取值不设兜底
# （`langchain/agents/middleware/human_in_the_loop.py:435`：
#   `decisions = interrupt(hitl_request)["decisions"]`），
# 因此「clarify 载荷被投给 HITL 中断」的必然后果就是这个 KeyError。
_HITL_DECISIONS_KEY = "decisions"


def _translate_secondary_keyerror(exc: BaseException) -> Optional["ResumeMismatchError"]:
    """
    §23 修复 1：把「图内部二次中断」拖出的 `KeyError('decisions')` 转成
    `ResumeMismatchError`（确定性错误），使其走「不回落 + 可读提示」路径。

    背景（生产 trace=9b8c4b2d4e714637，2026-09-22 09:47:33）：

    `_build_resume_command` 的配对校验只在**节点进入时**执行一次，针对当时悬挂的
    中断（clarify，与载荷 `{answers}` 配对 → 通过。这是对的）。但模型消费该中断后
    **自己发出写操作** → `after_model` 在 `astream` **内部**抛出**新的 HITL 中断**
    → 新中断期望的是上一级的 `interrupt()` 返回值，即 `{"decisions": [...]}`，
    而本级投进去的仍是常量 `{"answers": [...]}` → 下标取值 → `KeyError('decisions')`。

    第二次投递**完全在图内部产生**，不经过 `_build_resume_command`，配对校验在此
    彻底失位。故必须在异常出口补一道转译。

    **必须只认 `'decisions'` 这一个键名**：泛化捕获 `KeyError` 会把检查点结构变化、
    字段重命名这类**真实缺陷**也静默转成「载荷不匹配」的可读提示，
    让确定性的 bug 变成用户看不懂的降级 —— 那是更坏的结果。

    :param exc: 待判定的异常
    :return: 转译后的 ResumeMismatchError；不属于目标场景时返回 None（原样上抛）
    """
    if not isinstance(exc, KeyError):
        return None
    # KeyError 的 str() 带引号（"'decisions'"），args 里才是裸键名
    key = exc.args[0] if exc.args else None
    if key != _HITL_DECISIONS_KEY:
        return None
    return ResumeMismatchError(
        "本轮恢复后模型重新发起了写操作审批，而当前投递的是追问答案。"
        "请刷新会话，在界面上的**审批卡片**中处理该写操作（批准或拒绝）后重试。"
    )


# =============================================================================
# §23 修复 2：任务 Agent 的跨轮上下文注入
# =============================================================================
#
# 问题（生产 trace=9b8c4b2d4e714637，2026-09-22）：
# 用户第一轮已说「删除用户 dada 的信息」，任务 Agent 仍追问「请指明要处理的用户」；
# 用户在答案里被迫又填了一遍 "dada"。
#
# 根因是**跨分支功能不对称**：
#   - `chat_node` 有 `MySQLChatMessageHistory.aget_messages()` +
#     `history_messages[-HISTORY_RECENT_NUM:]` 拼进 prompt；
#   - `task_agent_node` 构造 `invoke_input` **只放当前轮一条 HumanMessage**，
#     从不加载历史。
#
# 而 §19 的内层 thread 隔离（thread_id = `{conv}::task_agent::{tid}`）又把
# 「靠检查点兜底拿到历史」这条后路也断了 —— 内层图从零开始，上下文**物理上不存在**。
# 于是模型看到的是「给我把这个用户的会话信息也失效了」，「这个用户」指代谁无从得知。
#
# 消息其实**已经落库**（`_save_conversation_messages` → `conversation_message` 表），
# 只是没有任何调用方读回来。本函数补上这一环。
#
# 注意：日志里的 `user=dada` 是**会话所属用户名**（HTTP 认证身份），不是用户提到的
# 目标用户 —— 模型无法从系统提示或元数据推断，只能靠历史。

# 注入给任务 Agent 的历史条数上限（配置驱动，默认见 `agent_task_history_recent_num`）。
# 复用 chat 分支的经验值：任务场景的实体（用户名、表名）通常出现在**紧邻上一轮**，
# 10 条足够；再多会被任务 Agent 的长工具结果一起挤爆上下文预算
#（内层每条 AI 消息都可能带大段工具输出）。
TASK_HISTORY_RECENT_NUM = 10


def _history_recent_num() -> int:
    """取本轮的注入条数上限（配置优先，异常回落常量）。"""
    try:
        n = int(getattr(settings, "agent_task_history_recent_num", TASK_HISTORY_RECENT_NUM))
        return n if n > 0 else 0
    except Exception:
        return TASK_HISTORY_RECENT_NUM


async def _load_task_history(
    db, conversation_id: str
) -> List[Any]:
    """
    §23 修复 2：加载本会话最近的历史对话，供任务 Agent 消解指代。

    返回**不含本轮输入**的历史消息（`HumanMessage` / `AIMessage`），
    按时间正序。任何失败都返回空列表 —— 上下文是**增强**，
    拿不到就退回改动前行为（只是模型可能重复追问），
    绝不能因为历史加载失败让任务本身跑不起来。

    三条关键边界：

    1. **裁掉末尾那条 user 消息**。对话消息落库与图执行是两条独立路径：
       - 首轮进入时历史里不含本轮，末条是**上一轮**的 AI 回复；
       - 但审批/追问恢复轮重入时，中断前若已落库，末条**就是本轮输入**。
       不裁会让同一问题出现两次 —— 更严重的是，本轮会用稳定的
       `id=task-input-{tid}` 构造消息，而历史里那条**没有 id**，
       `add_messages` 的按 id 去重**完全不生效**，两条并存。
       故统一按「末条若是 human 则丢弃」处理。
    2. **不给历史消息补 id**。`conversation_message` 表没有 message_id 列
       （见 `MySQLChatMessageHistory._deserialize_message`），天然无 id，满足要求。
       补 id 反而会与检查点里的记录冲突、破坏 `add_messages` 的 upsert 语义。
    3. **只取最近 `TASK_HISTORY_RECENT_NUM` 条**。

    :param db: 数据库会话
    :param conversation_id: 会话 ID
    :return: 历史消息列表（可能为空）
    """
    if db is None or not conversation_id:
        return []
    try:
        history = MySQLChatMessageHistory(session=db, conversation_id=conversation_id)
        messages = list(await history.aget_messages() or [])
    except Exception as e:
        metrics.incr(K_TASK_HISTORY_LOAD_FAILED)
        logger.warning(
            f"[Agent-任务] 加载历史对话失败（按无上下文继续，非致命）: "
            f"{type(e).__name__}: {e}, trace_id={get_trace_id()}"
        )
        return []

    # 边界 1：末条若是 human，说明它就是本轮输入（恢复轮重入场景），必须裁掉
    if messages:
        last = messages[-1]
        if getattr(last, "type", None) == "human":
            messages = messages[:-1]

    # 只保留 human / ai（system 不进历史；工具消息不落库，无需过滤）
    messages = [
        m for m in messages
        if getattr(m, "type", None) in ("human", "ai")
    ]

    # 边界 3：条数上限
    limit = _history_recent_num()
    recent = messages[-limit:] if limit > 0 else []

    # 边界 2：剥离 id（防止与检查点记录冲突；表里本就没有，防御性处理）
    for m in recent:
        if getattr(m, "id", None):
            try:
                m.id = None
            except Exception:
                pass

    if recent:
        metrics.incr(K_TASK_HISTORY_INJECTED)
    return recent


def _match_resume_target(
    resume_payload: Any, interrupts: List[Any]
) -> tuple:
    """
    §22 修复 1/4：把恢复载荷与悬挂中断配对，返回 (目标中断, 错误信息)。

    返回三元组语义：
      - `(None, "")`         : 无悬挂中断（调用方按裸 resume 继续，保持旧行为）
      - `(interrupt, "")`    : 找到配对目标
      - `(None, "原因")`      : **明确不匹配**，调用方必须拒绝投递
                                 （投递必炸 KeyError，且重投无效）

    配对规则：
      1. 载荷类型未知（空串）-> 不做校验，返回第一个中断（兼容旧载荷）；
      2. 载荷类型已知 -> 在悬挂中断里**筛选同类型**候选：
         - 恰好命中 -> 用该中断（即便它不是列表第一个）；
         - 命中多个 -> 用第一个命中项（调用方会告警）；
         - 一个都不命中 -> 再看**是否存在类型无法判定的中断**：
           存在则说明是判据能力不足（非真实冲突），按兼容路径放行首项；
           全部可判定且都不符 -> `(None, 原因)`，原因里带上「实际悬挂的是什么
           中断」，便于定位（例如「本轮实际在执行写操作审批，请先在审批卡片上处理」）。

    :param resume_payload: 恢复载荷
    :param interrupts: `_inner_pending_interrupts` 读到的悬挂中断列表
    :return: (目标中断或 None, 不匹配原因或 "")
    """
    if not interrupts:
        return None, ""

    payload_kind = _classify_resume_payload(resume_payload)
    if not payload_kind:
        # 载荷类型未知：不做配对校验（保持兼容，不新增拒绝面）
        return interrupts[0], ""

    matched = [i for i in interrupts if _classify_interrupt(i) == payload_kind]
    if matched:
        return matched[0], ""

    # 一个都不匹配。但「中断类型未知」（value 取不到 / 结构不认识）**不足以**
    # 判定不匹配 —— 那是判据能力不足，不是真实冲突。此时宁可放行（取首项，
    # 保持旧行为），也不要拦下一个本来正确的投递。
    unknown = [i for i in interrupts if not _classify_interrupt(i)]
    if unknown:
        logger.warning(
            f"[Agent-任务] 悬挂中断类型无法判定（value 形态不识别），"
            f"按兼容路径放行首项: 中断数={len(interrupts)}, trace_id={get_trace_id()}"
        )
        return interrupts[0], ""

    # 真实冲突：所有悬挂中断都能判定类型，且与载荷类型不符 -> 拒绝投递
    actual = [_classify_interrupt(i) for i in interrupts]
    if payload_kind == PAYLOAD_CLARIFY:
        reason = (
            "本轮待处理的是**写操作审批**，而收到的是追问答案；"
            "请先在界面上的审批卡片中处理该写操作（批准或拒绝）"
        )
    else:
        reason = (
            "本轮待处理的是**信息追问**，而收到的是写操作审批决策；"
            "请先在界面上的追问卡片中提交答案"
        )
    return None, f"{reason}（当前悬挂中断类型: {', '.join(actual)}）"



def _is_llm_message(chunk: Any) -> bool:
    """判断一个流式 chunk 是否是**模型**产出的消息（排除 ToolMessage 等）。"""
    # ToolMessage 是数据库原始结果，绝不能推给前端
    if type(chunk).__name__ in ("ToolMessage", "FunctionMessage"):
        return False
    return True


def _tool_calls_of(chunk: Any) -> List[str]:
    """取出该 chunk 携带的**工具调用**名（用于「工具开始」进度事件）。"""
    try:
        calls = getattr(chunk, "tool_call_chunks", None) or getattr(
            chunk, "tool_calls", None
        )
        names: List[str] = []
        for call in calls or []:
            if isinstance(call, dict):
                name = call.get("name")
                if name:
                    names.append(str(name))
        return names
    except Exception:
        return []


async def _run_task_agent_streaming(
    agent,
    invoke_input: Any,
    invoke_config: dict,
    emitter: StreamEmitter,
    token: int,
    queue,
) -> dict:
    """
    §21：以流式方式执行内层任务 Agent，边跑边推送。

    返回结构与 `ainvoke` 一致（含 `messages` 与可能的 `__interrupt__`），
    使调用方（含 `_absorb_reissued_interrupt` / `_extract_answer`）**无需改动**。

    :param agent: 内层任务 Agent（CompiledStateGraph）
    :param invoke_input: 首轮为 {"messages": [...]}，恢复轮为 Command(resume=...)
    :param invoke_config: 内层运行配置（独立 thread_id）
    :param emitter: 流式派发器（负责思考块过滤 / 单写者栅栏 / 去重）
    :param token: 本轮流写者 token
    :param queue: SSE 队列
    :return: 与 ainvoke 等价的合并结果
    """
    merged: dict = {}
    messages: List[Any] = []
    interrupts: List[Any] = []
    reasoning_chars = 0
    max_reasoning = max(0, int(getattr(settings, "agent_reasoning_max_chars", 0) or 0))
    show_tools = bool(getattr(settings, "agent_task_tool_events_enabled", True))
    started_tools: set = set()

    # §23 修复 1：图内部二次中断拖出的 KeyError('decisions') 在此转译。
    #
    # 位置说明：转译必须包在 `astream` 之外 —— 异常是在**迭代过程中**从图内部
    # 抛出的（`human_in_the_loop.after_model` 的下标取值），只能在 `async for`
    # 的外层接住。转译成 ResumeMismatchError 后由 `_run_task_agent` 的
    # 异常白名单原样上抛（不回落），最终由节点层转成用户可读提示。
    try:
        async for mode, payload in agent.astream(
            invoke_input, config=invoke_config, stream_mode=["messages", "updates"]
        ):
            if mode == "messages":
                # `messages` 模式产出 (message_chunk, metadata) 二元组
                chunk, meta = payload if isinstance(payload, tuple) and len(payload) == 2 \
                    else (payload, {})
                node_name = (meta or {}).get("langgraph_node")

                # 关键过滤：只认模型节点的产出。不滤会把 tools 节点的
                # ToolMessage（数据库原始结果）当作正文推给用户。
                if node_name != NODE_MODEL:
                    if node_name == NODE_TOOLS and show_tools:
                        # 兜底：部分版本 messages 模式不吐 tool 节点消息，
                        # 此处不动（结束事件由 updates 分支负责）
                        pass
                    continue
                if not _is_llm_message(chunk):
                    continue

                # 思考过程（Qwen/DeepSeek 等思考模型）
                reasoning = extract_reasoning(chunk)
                if reasoning:
                    # 超长思考的保护：达到上限后停止推送（任务仍在跑，只是不再刷思考区）
                    if max_reasoning == 0 or reasoning_chars < max_reasoning:
                        if await emitter.emit_thinking(reasoning, token):
                            metrics.incr("agent.reasoning.deltas")
                            reasoning_chars += len(reasoning)

                # 正文
                text = extract_text(chunk)
                if text:
                    if await emitter.emit_content(text, token):
                        metrics.incr("agent.task.stream_chunks")

                # 工具调用发起 → 进度事件（同一工具名只报一次 start）
                if show_tools:
                    for name in _tool_calls_of(chunk):
                        if name in started_tools:
                            continue
                        started_tools.add(name)
                        await put_tool(queue, name, "start")
                        metrics.incr("agent.task.tool_events")

            elif mode == "updates":
                # updates 模式：节点执行完成后产出 {节点名: 状态增量}
                if not isinstance(payload, dict):
                    continue
                for node_name, delta in payload.items():
                    if node_name == "__interrupt__":
                        # 中断以 ("__interrupt__", (Interrupt(...),)) 形式出现在 updates 里
                        values = delta if isinstance(delta, (list, tuple)) else (delta,)
                        interrupts.extend(values)
                        continue
                    if not isinstance(delta, dict):
                        continue
                    if node_name == NODE_TOOLS and show_tools:
                        for msg in (delta.get("messages") or []):
                            name = getattr(msg, "name", None) or "tool"
                            await put_tool(queue, str(name), "end")
                            metrics.incr("agent.task.tool_events")
                    for msg in (delta.get("messages") or []):
                        messages.append(msg)
    except BaseException as e:
        # §23 修复 1：只转译 `KeyError('decisions')`，其余异常原样上抛。
        # 注意用 BaseException 接：`GraphInterrupt` 继承自 `Exception`，
        # 但保留 BaseException 可避免未来继承链变化时漏接中断。
        translated = _translate_secondary_keyerror(e)
        if translated is None:
            raise
        metrics.incr(K_TASK_RESUME_SECONDARY_INTERRUPT)
        logger.warning(
            f"[Agent-任务] 恢复后模型重新发起写操作，产生新的审批中断，"
            f"载荷与中断类型不匹配（已转译，不回落）: {e!r}, "
            f"trace_id={get_trace_id()}"
        )
        raise translated from e

    merged["messages"] = messages
    if interrupts:
        merged["__interrupt__"] = tuple(interrupts)
    return merged


async def _run_task_agent(
    agent,
    invoke_input: Any,
    invoke_config: dict,
    emitter: Optional[StreamEmitter],
    token: int,
    queue,
    *,
    use_timeout: bool,
) -> dict:
    """
    §21：执行内层任务 Agent（优先流式，失败回落非流式）。

    流式是**增强**：**环境类**异常回落到 `ainvoke`，
    保证任务本身照常跑完（最坏退回改造前的「跑完一次性推送」）。

    三类异常**必须原样上抛、不得回落**（§22 修复 2）：

    - `GraphInterrupt`：审批/追问中断依赖它向上冒泡，在此吞掉会导致
      审批链路彻底失效（用户看不到卡片，写操作悬空）；更严重的是回落会
      用同一份 `invoke_input` 重跑，等于让已批准的写操作**执行两遍**。
    - `asyncio.TimeoutError`：整体超时，交由上层统一转友好错误。
    - `ResumeMismatchError`：**确定性**错误（恢复载荷与悬挂中断类型不匹配）。
      回落重投**同一个** `Command(resume=...)` 必然得到同样的异常
      （实测见 `scripts/diag_keyerror_timing.py` Q2.1），回落只是把同一个
      错误再抛一次、白白多跑一轮模型。必须在投递前拦下并转可读提示。
      （§23 修复 1 起，图内部二次中断拖出的 `KeyError('decisions')` 也在
      `_run_task_agent_streaming` 里转译成本异常，走同一条路径。）

    除此之外的异常（langgraph 版本不支持 stream_mode、消息结构变化等
    **环境类**故障）才回落 —— 那时非流式路径确实能跑通。

    :param use_timeout: 是否施加整体墙钟超时（非审批模式才加，审批等待不计时）
    """
    streaming_on = (
        emitter is not None
        and bool(getattr(settings, "agent_task_streaming_enabled", True))
    )

    if streaming_on:
        try:
            coro = _run_task_agent_streaming(
                agent, invoke_input, invoke_config, emitter, token, queue
            )
            if use_timeout:
                return await asyncio.wait_for(
                    coro, timeout=settings.agent_task_timeout_seconds
                )
            return await coro
        except (GraphInterrupt, asyncio.TimeoutError, ResumeMismatchError):
            # GraphInterrupt：审批/追问中断必须冒泡（不得降级重跑，否则重复执行写操作）
            # TimeoutError：整体超时，交由上层统一转友好错误
            # ResumeMismatchError：确定性错误，重投无效（§22 修复 2）
            raise
        except Exception as e:
            # 流式路径自身故障（版本不支持 stream_mode / 结构变化等**环境类**问题）：
            # 计数告警后回落非流式，任务照常执行。
            metrics.incr("agent.task.stream_fallback")
            logger.warning(
                f"[Agent-任务] 流式执行失败，回落非流式（任务不受影响）: "
                f"{type(e).__name__}: {e} trace_id={get_trace_id()}"
            )

    if use_timeout:
        return await asyncio.wait_for(
            agent.ainvoke(invoke_input, config=invoke_config),
            timeout=settings.agent_task_timeout_seconds,
        )
    return await agent.ainvoke(invoke_input, config=invoke_config)


# =============================================================================
# 内层任务 Agent 的检查点隔离（修复跨轮串味）
# =============================================================================
#
# 背景：`create_agent(...)` 产出的任务 Agent 是**一张独立的 LangGraph 图**，
# 审批模式下会挂上 `get_checkpointer()`（与主图同一个 MySQL saver）。
#
# 修复前调用它时直接透传了主图的 config：
#
#     invoke_config = {**(config or {}), "recursion_limit": ...}
#     agent.ainvoke({"messages": [...]}, config=invoke_config)
#
# `config["configurable"]["thread_id"]` 就是 conversation_id，于是**内层图与主图
# 共用同一个 thread**。两张图的 state 都有 `messages` channel（同名同 reducer），
# 后果是互相读写对方的检查点：
#
#   1. 内层图加载 thread=conversation_id 时，读到的是**主图的检查点**——
#      里面存着本会话**历轮累积的 messages**（第一轮的知识库问题也在里面），
#      再 append 本轮问题 → 喂给 LLM 的第一条 user message 是**上一轮的问题**。
#      这正是「LangSmith 上任务这轮显示的用户输入是第一轮的」的直接原因。
#   2. 内层图每执行一个 superstep 就把自己写出的 channel 回写该 thread，
#      其中与主图**同名**的 `messages` 会被内层图版本整条覆盖：
#      [上一轮历史…, 本轮问题, 任务 Agent 的 AI/工具消息]。于是串味被**固化**
#      进检查点——主图下一轮、或审批恢复后重入时，读到的 `messages` 已是
#      内层图塞进去的那一份，越跑越脏。审批中断场景尤其致命。
#
# 修法：给内层图一个**独立命名空间**的 thread_id（带上 task_execution_id，
# 保证同一任务的「中断 → 恢复」仍能命中同一条 thread）。
# 主图与内层图的检查点从此互不干扰，同时 LangSmith 上的 run 也各自独立，
# 不再出现「任务轮的输入/输出是第一轮内容」的错乱。

INNER_THREAD_PREFIX = "task_agent"


def _inner_thread_id(conversation_id: str, task_execution_id: str) -> str:
    """
    构造内层任务 Agent 的独立 thread_id。

    带上 `task_execution_id` 而非只用 conversation_id：审批/追问中断恢复时
    节点会被重入，`_find_running_task` 复用同一条 running 记录 → 同一个
    task_execution_id → 同一个内层 thread → 恢复能命中中断前的状态。
    """
    return f"{conversation_id}::{INNER_THREAD_PREFIX}::{task_execution_id}"


def _build_subgraph_config(
    config: Any, thread_id: str, recursion_limit: int
) -> dict:
    """
    构造内层 Agent 的运行配置：复用 db / sse_queue / trace_id / metadata / tags，
    但把 `thread_id` 换成独立命名空间。

    保留 metadata 与 tags 是有意的：LangSmith 上内层 run 仍挂在本轮 trace 树下，
    只是不再与主图共享检查点。
    """
    cfg = dict(config or {})
    configurable = dict(cfg.get("configurable") or {})
    configurable["thread_id"] = thread_id
    cfg["configurable"] = configurable
    cfg["recursion_limit"] = recursion_limit
    return cfg


@node_trace("task_agent")
async def task_agent_node(
    state: AgentState, config: RunnableConfig | None = None
) -> dict:
    """
    任务执行节点（阶段 3：接入 HITL 人工审批）。

    审批模式（checkpointer 可用）：绑定全部 5 个工具，写操作经
    HumanInTheLoopMiddleware 中断，审批回调后以同一 thread 恢复执行；
    节点在恢复时会被重入，因此审计逻辑需**幂等**（复用 running 任务记录）。

    :param state: 图状态
    :param config: 运行配置（configurable 携带 db 会话与 sse_queue）
    :return: 状态增量（task_answer / task_execution_id / tool_calls）
    """
    queue = get_sse_queue(config)
    db = (config or {}).get("configurable", {}).get("db")
    if db is None:
        logger.error("[Agent-任务] 图运行配置缺少 db 会话")
        return {"task_answer": "任务执行失败：缺少数据库会话", "error": "missing db session"}

    conversation_id = state.get("conversation_id", "")
    user_id = state.get("user_id")
    model = state.get("model", "qwen-turbo")
    user_input = state["messages"][-1].content
    intents = state.get("intents") or []

    # 审批模式：仅检查点可用时启用（写工具 + HITL 中间件）
    from agent.graph_builder import get_checkpointer
    approval_mode = get_checkpointer() is not None

    executor = DbToolExecutor(db)
    # 知识库检索执行器：username 服务端注入（权限过滤），不暴露给模型
    kb_executor = KnowledgeToolExecutor(db, username=state.get("username"))
    start = time.perf_counter()

    # 1. 任务记录（幂等）：**仅恢复轮**复用中断前创建的 running 记录。
    #
    # §24 修复：`_find_running_task` 现在会拒收「陈旧记录」（内层已无悬挂中断）。
    # 拒收后必须**顺手把它终结**（标 completed），否则它会一直留在 running 状态
    # 被后续每一轮反复查询、反复拒收 —— 既浪费一次 `aget_state`，也让审计表
    # 里堆积僵尸记录，运维侧无法分辨「真的在等审批」与「已终结」。
    #
    # 注意区分两种「本轮」：
    #   - 恢复轮：`resume_payload` 非空 → 一定要复用（决策要投给中断的那层）；
    #   - 新轮  ：`resume_payload` 为空 → 只有「真在等审批」的记录才复用。
    resume_payload_hint = (config or {}).get("configurable", {}).get("approval_resume")
    if resume_payload_hint is None:
        resume_payload_hint = (config or {}).get("configurable", {}).get("clarify_resume")
    is_resume_round = resume_payload_hint is not None

    record = await _find_running_task(
        db, conversation_id, reject_stale=not is_resume_round
    )
    if record is None:
        # 拒收过陈旧记录时，把僵尸记录收尾（best-effort，非致命）
        if not is_resume_round:
            await _close_stale_running_tasks(db, conversation_id)
        task_execution_id = str(uuid.uuid4())
        await _create_task_record(
            db, task_execution_id, conversation_id, user_id,
            intent=",".join(intents) if intents else "task",
        )
        await put_status(queue, "task_started", task_execution_id=task_execution_id)
    else:
        task_execution_id = record.id
        # 恢复中断前的工具调用审计（中断时已持久化）
        executor.tool_calls = list(record.tool_calls or [])

    # §21：流式派发器。**必须在 try 之前创建并 claim**——
    # 审批中断后节点会被重入，claim 递增写者 token 使上一轮残留的流全部失效
    #（否则旧流的尾帧会与新一轮内容交错，得到语义错乱的文本）。
    # `_delivered` 去重集合跨 claim 保留：重入后模型会把审批前那段说明再生成一遍，
    # 已推过的内容靠指纹拦下，用户不会看到同样的话出现两次。
    emitter = StreamEmitter(queue, task_execution_id=task_execution_id)
    stream_token = emitter.claim()

    try:
        # 2. 构建任务 Agent（审批模式绑定全部生效工具；工具从 tool_registry 查询 status=1 生效项）
        agent = await _build_task_agent(model, db, executor, kb_executor, approval_mode)

        # 3. 执行任务（恢复执行时从检查点继续，审批决策自动路由）
        # 注入 recursion_limit 覆盖 create_agent 默认值，防模型-工具死循环
        # 注：config 可能为 None（离线/单测场景），必须兜底，否则 {**None} 抛 TypeError。
        logger.info(
            f"[Agent-任务] 开始执行: task_execution_id={task_execution_id}, "
            f"approval_mode={approval_mode}, trace_id={get_trace_id()}"
        )
        # 内层 Agent 的 thread 与主图隔离（见 `_inner_thread_id` 的说明）。
        # 关闭开关即逐字节回到改动前行为（共用 conversation_id 作为 thread_id）。
        if getattr(settings, "agent_task_thread_isolate", True):
            invoke_config = _build_subgraph_config(
                config,
                _inner_thread_id(conversation_id, task_execution_id),
                _recursion_limit(),
            )
        else:
            invoke_config = {
                **(config or {}),
                "recursion_limit": _recursion_limit(),
            }

        # 输入消息带**稳定 id**：审批/追问中断恢复时节点会被重入，若每次都新建
        # HumanMessage（随机 id），`add_messages` 会再 append 一条重复的用户消息，
        # 模型于是看到同一个问题出现两次。固定 id 后按 LangGraph 语义为 upsert。
        #
        # §23 修复 2：在**当前轮之前**注入最近的历史对话，让模型能消解
        # 「这个用户 / 刚才那个 / 上一个」这类指代（见 `_load_task_history`）。
        # 内层 thread 隔离后内层图物理上拿不到任何历史，这是唯一的上下文来源。
        #
        # 两条约束互不冲突：历史消息**不带 id**（表里无 id 列，也刻意不补），
        # 本轮消息 id 稳定 —— `add_messages` 只对本轮这条做 upsert，历史原样保留。
        history_messages: List[Any] = []
        if getattr(settings, "agent_task_history_inject", True):
            history_messages = await _load_task_history(db, conversation_id)

        invoke_input: Any = {
            "messages": [
                *history_messages,
                HumanMessage(
                    content=user_input, id=f"task-input-{task_execution_id}"
                ),
            ]
        }
        if history_messages:
            logger.info(
                f"[Agent-任务] 注入历史上下文 {len(history_messages)} 条"
                f"（供指代消解）: task_execution_id={task_execution_id}, "
                f"trace_id={get_trace_id()}"
            )

        # ---- §20 F3.0：恢复轮必须把审批决策**透传给内层 Agent** ----
        #
        # 原实现无论首轮还是恢复轮，都传全新 dict（上面那条 HumanMessage）。
        # 但内层 Agent 挂了自己的 checkpointer（thread 见 `_inner_thread_id`），
        # 传「新输入」等于让 LangGraph 把它当成**一次全新的 run**：
        # 检查点里那条「待审批中断」被**静默丢弃**（既没 approve 也没 reject）→
        # 内层从头重跑 → 模型看到用户问题**又发起一次同一个写操作** →
        # HITL 中间件**再次中断** → 外层只准备了一次 resume，无法处理 →
        # **写操作从未执行**，而外层以为恢复成功了。
        #
        # 最小复现（scripts/smoke_approval_resume.py）：
        #   GOOD（Command(resume)）: 恢复后中断=False, delete_data 执行=['user:id=3']
        #   BAD （全新 input）      : 恢复后中断=True,  delete_data 执行=[]
        #
        # 修法：恢复轮改传 `Command(resume=...)`。审批决策由
        # approval_service.resume_graph 写入 `configurable.approval_resume`
        # （clarify_service 同理写 `clarify_resume`），这里取出后转交内层。
        #
        # 为什么不采用「内层不挂 checkpointer、让中断冒泡到主图」的简化方案：
        # 内层 thread 隔离是 §17.x 刻意的设计（见上方长注释），共用检查点会让
        # 主图 messages channel 被内层整条覆盖、串味固化。故保留隔离，
        # 改为显式透传决策。
        resume_payload = (config or {}).get("configurable", {}).get("approval_resume")
        if resume_payload is None:
            resume_payload = (config or {}).get("configurable", {}).get("clarify_resume")
        if resume_payload is not None:
            # ---- §20 F3.5：恢复轮先对齐「到底有几个悬挂中断」----
            #
            # 稳定 id 的副作用（生产实测 trace=00b0b22f2b0446b7）：
            # `_find_running_task` 复用同一条 running 记录 → 同一个内层 thread，
            # 后续任何一次「new input 进入」都会 upsert 覆盖那条 HumanMessage，
            # 而内层 `last_ai_msg.tool_calls` 仍带着上一轮未消费的写操作 →
            # `after_model` **再次 interrupt**。于是同一 thread 上会累积
            # **多个 id 不同**的悬挂中断。
            #
            # 此时若只传一个裸 `Command(resume=<决策>)`，LangGraph 会把它当作
            # 「下一个中断的答案」，消费掉**别的那一个**，目标中断仍在 →
            # 恢复后再次中断 → 外层误判失败。
            #
            # 解法：按中断 id **逐个**投递（langgraph 1.2.10 支持
            # `Command(resume={interrupt_id: value})`，见 `pregel/_loop.py`
            # 的 CONFIG_KEY_RESUME_MAP 与 `pregel/_algo.py` 的 resume_map 分支）。
            invoke_input = await _build_resume_command(
                agent, invoke_config, resume_payload
            )
            logger.info(
                f"[Agent-任务] 恢复轮透传人工介入决策: "
                f"type={type(resume_payload).__name__} "
                f"keys={list(resume_payload)[:4] if isinstance(resume_payload, dict) else '-'} "
                f"trace_id={get_trace_id()}"
            )

        # §21：流式执行（边跑边推送正文 / 思考过程 / 工具进度）。
        # 审批模式不设整体墙钟超时（等待用户审批可能很久，超时语义由审批侧
        # APPROVAL_WAIT_TIMEOUT 负责），仍受工具级/LLM 超时与递归上限保护；
        # 非审批模式施加整体超时兜底。流式失败自动回落 ainvoke（见 `_run_task_agent`）。
        result = await _run_task_agent(
            agent,
            invoke_input,
            invoke_config,
            emitter,
            stream_token,
            queue,
            use_timeout=not approval_mode,
        )

        # 流结束：吐出被思考块过滤器扣留的正常尾巴（未闭合思考块内的内容丢弃）
        await emitter.flush(stream_token)

        # ---- §20 F3.5b：恢复轮被模型"原地重发"拖出的新中断，在此就地消化 ----
        #
        # 生产实测：`Command(resume=approve)` 消费旧中断 → 写工具**真的执行了**
        # → ToolMessage 回灌 → 模型（qwen3.8-max-0902）**又发一次同一个写操作**
        # → `after_model` 抛出**第三个**中断，出现在 result["__interrupt__"] 里。
        # 原实现把这个中断一路冒泡给 `approval_service`，后者据此判
        # 「决策未生效」→ 标 failed → 已执行的写操作被当成失败上报。
        #
        # 判据：若 `resume_payload` 是**已批准**的（decisions 里全是 approve），
        # 且本轮确有写操作执行成功，则该新中断是模型的重复请求，**不是**
        # 本次决策失效 → 就地丢弃，按成功路径继续。
        # 反例（不得丢弃）：决策是 reject/canceled 时出现的写中断是真信号，
        # 或本轮没有任何写操作成功 —— 说明决策确实没生效，照旧冒泡。
        if resume_payload is not None:
            result = await _absorb_reissued_interrupt(
                agent,
                invoke_config,
                result,
                resume_payload,
                _merge_tool_calls(executor, kb_executor),
            )

        task_answer = _extract_answer(result)

        # ---- §22 修复 3：澄清答案「退化」告警 ----
        #
        # 生产 trace=07dbbe8d604a4106 暴露的体验问题：用户已经回答了 3 个追问，
        # 模型却重新走了一遍「列出表结构 → 请提供筛选条件」，把问题抛回用户。
        # 这与 KeyError 是**独立**的缺陷（提示词/上下文层面），必须让它**可见**，
        # 否则这类退化只会以「用户抱怨没执行」的形式被间接发现。
        #
        # 判据：本轮载荷是 clarify 答案（非取消），但审计里没有任何写操作调用。
        # 非致命 —— 只计数 + 告警日志，不改变任务结果。
        _warn_clarify_answer_without_write(resume_payload, executor, kb_executor)

        # 4. 兜底推送（§21.5）
        #
        # 流式路径下正文已经在执行过程中逐帧推给了前端，这里**不能**再整段推一次
        # （否则用户看到两遍）。判据用「本轮是否真的推过内容」而不是「是否走了流式」
        # —— 流式路径也可能因为思考块过滤/去重而一帧未推（例如全部内容都在
        # <think> 块内被拦下），那时仍然需要兜底，否则界面上什么都没有。
        #
        # 只推「未推过的尾巴」：流式推了部分内容、模型最后又补了一段时，
        # 整段重推会把前面部分重复一遍，故从已推长度处裁切。
        if queue is not None and task_answer:
            if emitter.delivered_count > 0:
                tail = task_answer[emitter.delivered_chars:]
                if tail:
                    await put_content(queue, tail, branch="task")
            else:
                await put_content(queue, task_answer, branch="task")

        # 5. 更新审计（completed）
        await _update_task_record(
            db, task_execution_id, status="completed",
            task_answer=task_answer,
            tool_calls=_merge_tool_calls(executor, kb_executor),
        )
        # 6. 对话消息落库（保证重新进入会话时任务轮次不丢失）
        await _save_conversation_messages(db, conversation_id, user_input, task_answer)
        tool_calls = _merge_tool_calls(executor, kb_executor)
        cost_ms = int((time.perf_counter() - start) * 1000)
        metrics.observe("agent.task.latency_ms", cost_ms)
        metrics.observe("agent.task.tool_calls", len(tool_calls))
        logger.info(
            f"[Agent-任务] 完成: task_execution_id={task_execution_id}, "
            f"耗时={cost_ms}ms, 工具调用={len(tool_calls)} 次, "
            f"trace_id={get_trace_id()}"
        )
        return {
            "task_answer": task_answer,
            "task_execution_id": task_execution_id,
            "tool_calls": tool_calls,
            # §21.5：标记「本轮已推过该内容」，供 finalize_node 补推去重。
            # 不写这个标记，finalize 会以为任务分支没推过 → 把整段答案再推一次，
            # 用户在流式看完整答案后又看到一遍（§20 F2.4 的 dedup 就靠这个字段）。
            "pushed_contents": [task_answer] if task_answer else [],
        }
    except GraphInterrupt:
        # 审批中断：持久化已发生的工具调用审计后向上传播（等待审批恢复）
        # 状态保持 running：审批恢复时任务节点重入并复用该记录
        await _update_task_record(
            db, task_execution_id, status="running",
            tool_calls=_merge_tool_calls(executor, kb_executor),
        )
        # §20 F3.7：本次中断前若已有写操作**执行成功**，立刻独立留痕。
        # 恢复轮模型可能原地重发同一写操作、产生新中断，导致 resume_graph
        # 误判「决策未生效」—— 届时这条审计就是"操作其实已经做过"的唯一凭据。
        await _persist_write_audit(
            db, task_execution_id, _merge_tool_calls(executor, kb_executor)
        )
        raise
    except GraphRecursionError:
        # 模型-工具循环达到 recursion_limit 上限（防死循环硬保护）
        error_msg = "任务执行超过最大尝试次数，请将请求拆分或简化后重试"
        if queue is not None:
            await put_content(queue, error_msg, branch="task")
        await _update_task_record(
            db, task_execution_id, status="failed",
            error=error_msg, tool_calls=_merge_tool_calls(executor, kb_executor),
        )
        await _save_conversation_messages(db, conversation_id, user_input, error_msg)
        return {"task_answer": None, "error": error_msg}
    except asyncio.TimeoutError:
        # Agent 循环整体超时（非审批模式兜底）
        error_msg = f"任务执行超时（>{settings.agent_task_timeout_seconds}s），请稍后重试"
        if queue is not None:
            await put_content(queue, error_msg, branch="task")
        await _update_task_record(
            db, task_execution_id, status="failed",
            error=error_msg, tool_calls=_merge_tool_calls(executor, kb_executor),
        )
        await _save_conversation_messages(db, conversation_id, user_input, error_msg)
        return {"task_answer": None, "error": error_msg}
    except ResumeMismatchError as e:
        # §22 修复 1/2：恢复载荷与悬挂中断类型不匹配（确定性错误）。
        # 不回落、不重投 —— 直接给用户可读的处置指引，任务保持 running
        #（不要标 failed：卡片还在，用户按指引处理后本轮可正常续跑）。
        error_msg = f"恢复执行被拦截：{e}"
        logger.error(
            f"[Agent-任务] 恢复载荷与中断类型不匹配，已拦截: "
            f"task_execution_id={task_execution_id}, {e}, trace_id={get_trace_id()}"
        )
        if queue is not None:
            await put_content(queue, error_msg, branch="task")
        # 状态保持 running：审批/追问卡片仍有效，用户按提示处理后重入即可继续
        await _update_task_record(
            db, task_execution_id, status="running",
            tool_calls=_merge_tool_calls(executor, kb_executor),
        )
        return {"task_answer": error_msg, "error": error_msg}
    except Exception as e:
        logger.exception(f"[Agent-任务] 执行失败: {e}")
        error_msg = f"任务执行失败：{str(e)}"
        # 推送可读错误到 SSE（不中断连接）
        if queue is not None:
            await put_content(queue, error_msg, branch="task")
        await _update_task_record(
            db, task_execution_id, status="failed",
            error=error_msg, tool_calls=_merge_tool_calls(executor, kb_executor),
        )
        await _save_conversation_messages(db, conversation_id, user_input, error_msg)
        return {"task_answer": None, "error": error_msg}


# =============================================================================
# 任务 Agent 构建
# =============================================================================

async def _build_task_agent(
    model: str,
    db,
    executor: DbToolExecutor,
    kb_executor: KnowledgeToolExecutor,
    approval_mode: bool,
):
    """
    构建任务执行 Agent。

    审批模式（checkpointer 可用）：绑定全部**生效**工具（含写工具），写操作经
    HumanInTheLoopMiddleware 审批后执行；无审批模式仅绑定只读 + 不审批工具。
    工具列表来自 tool_registry 表 status=1（生效）的记录（见 agent/tools/manager.py），
    失效（0）/熔断（2）的工具不会被绑定，实现单工具粒度的故障隔离。
    """
    # §21.2：走统一入口而非裸 ChatOpenAI —— 任务 Agent 也需要思考过程
    #（`ThinkingChatOpenAI` 把 `reasoning_content` 从 delta 捞进 additional_kwargs，
    # 裸 ChatOpenAI 会静默丢弃）。`enable_thinking` 按模型白名单自动判定，
    # 非思考模型不传该参数（传了可能直接报错）。
    llm = get_chat_model(
        model,
        temperature=0.4,
        streaming=True,
        # 单次 LLM 推理超时（模型挂死不再无限等待）
        timeout=settings.agent_llm_timeout_seconds,
    )
    tools = await bind_active_tools(db, executor, kb_executor, approval_mode)
    if not approval_mode:
        # 追问工具依赖 langgraph interrupt（需 checkpointer），非审批模式不可用，
        # 剔除避免模型绑定后调用报错（退化回"如实告知用户信息不足"）
        tools = [t for t in tools if t.name != ASK_USER_TOOL_NAME]

    kwargs: dict = {}
    if approval_mode:
        from agent.graph_builder import get_checkpointer

        kwargs["middleware"] = [
            HumanInTheLoopMiddleware(
                interrupt_on=_interrupt_on_config(),
                description_prefix="数据库写操作待审批",
            )
        ]
        kwargs["checkpointer"] = get_checkpointer()
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=TASK_AGENT_SYSTEM_PROMPT,
        **kwargs,
    )


def _interrupt_on_config() -> dict:
    """构造 HITL 中断配置：update/delete 必审批，insert 随配置开启。"""
    interrupt_on = {
        "update_data": {"allowed_decisions": ["approve", "reject"]},
        "delete_data": {"allowed_decisions": ["approve", "reject"]},
    }
    if settings.agent_insert_requires_approval:
        interrupt_on["insert_data"] = {"allowed_decisions": ["approve", "reject"]}
    return interrupt_on


def _extract_answer(result: Any) -> str:
    """从 create_agent 的返回结构中提取最终回答。"""
    messages = result.get("messages") if isinstance(result, dict) else result
    if isinstance(messages, (list, tuple)) and messages:
        last = messages[-1]
        content = getattr(last, "content", None)
        if content:
            return content if isinstance(content, str) else str(content)
    return "任务已完成（无文本结果）"


def _merge_tool_calls(
    executor: DbToolExecutor, kb_executor: Optional[KnowledgeToolExecutor]
) -> list:
    """
    合并数据库与知识库两个执行器的审计记录（方案 A：合并取并集）。

    知识库检索不触发审批，因此在审批恢复重入时其调用记录不会重复写入；
    两个执行器分别记录，最终一次性合并落库。
    """
    calls = list(executor.tool_calls or [])
    if kb_executor is not None:
        calls.extend(kb_executor.tool_calls or [])
    return calls


# =============================================================================
# §20 F3.7：写操作执行与「结果上报」解耦
# =============================================================================

# 触发人工审批的写工具名（与 `_interrupt_on_config` 保持一致）
WRITE_TOOL_NAMES: tuple = ("insert_data", "update_data", "delete_data")
# 判定「写操作确实执行」的关键字段（工具结果体里回传）
_WRITE_OK_KEYS: tuple = ("affected_rows", "inserted_id", "rowcount")


def _successful_writes(tool_calls: Optional[List[dict]]) -> List[dict]:
    """
    从审计记录中筛出**确实执行成功**的写操作（§20 F3.7）。

    判据：工具名属于写工具 + 结果体 `ok=True` + 命中 `_WRITE_OK_KEYS`
    之一（说明真的落到了库，而不是被拒绝/报错）。

    背景（生产 trace=00b0b22f2b0446b7）：审批通过后写操作**确实执行了**，
    但模型随即又发一次同一写操作、产生第三个中断，`resume_graph` 判定
    「决策未生效」→ 标 `failed`，**已执行的事实与 affected_rows 全部丢失**，
    用户看到"什么都没发生"。故此处提供独立、不依赖后续模型行为的判据。
    """
    out: List[dict] = []
    for call in tool_calls or []:
        if not isinstance(call, dict):
            continue
        if str(call.get("tool", "")) not in WRITE_TOOL_NAMES:
            continue
        summary = call.get("result_summary") or {}
        if not isinstance(summary, dict) or not summary.get("ok"):
            continue
        if not any(k in summary for k in _WRITE_OK_KEYS):
            continue
        out.append(call)
    return out


def _describe_writes(writes: List[dict]) -> str:
    """把已执行的写操作渲染成一句人话（用于失败路径也如实告知用户）。"""
    parts: List[str] = []
    for w in writes:
        params = w.get("params_summary") or {}
        summary = w.get("result_summary") or {}
        table = params.get("table", "?")
        affected = summary.get("affected_rows", summary.get("rowcount"))
        mode = summary.get("mode")
        seg = f"{w.get('tool')}（表 {table}"
        if affected is not None:
            seg += f"，影响 {affected} 行"
        if mode:
            seg += f"，模式 {mode}"
        seg += "）"
        parts.append(seg)
    return "；".join(parts)


async def _persist_write_audit(
    db, task_execution_id: str, tool_calls: List[dict]
) -> None:
    """
    §20 F3.7：写操作执行成功后的**独立**审计落库（best-effort，不阻塞主流程）。

    与 `_update_task_record` 的区别：本函数**不改状态、不改 task_answer**，
    只把「已发生的写操作」写进 `tool_calls`。这样无论后续模型行为如何
    （原地重发、被误判失败、超时），已执行的事实都不会被覆盖掉。
    """
    writes = _successful_writes(tool_calls)
    if not writes:
        return
    try:
        await _update_task_record(
            db, task_execution_id, status=None, tool_calls=tool_calls
        )
        logger.info(
            f"[Agent-任务] 写操作审计已独立落库: task_execution_id={task_execution_id}, "
            f"已执行写操作={len(writes)} 次 [{_describe_writes(writes)}], "
            f"trace_id={get_trace_id()}"
        )
        metrics.incr("agent.task.write_audit.persisted")
    except Exception as e:
        logger.warning(f"[Agent-任务] 写操作审计落库失败（非致命）: {e}")


# =============================================================================
# §20 F3.5：恢复轮的中断对齐
# =============================================================================

async def _inner_pending_interrupts(
    agent, invoke_config: dict
) -> List[Any]:
    """
    读取内层 Agent 当前**悬挂未恢复**的中断列表（best-effort，失败返回空）。

    走 `aget_state` 而不是猜——只有拿到真实 id 才能定向投递 resume 值。
    """
    try:
        state = await agent.aget_state(invoke_config)
    except Exception as e:
        logger.warning(f"[Agent-任务] 读取内层状态失败（按无悬挂中断处理）: {e}")
        return []
    try:
        interrupts: List[Any] = []
        for task in (state.tasks or []):
            for itr in (getattr(task, "interrupts", None) or ()):
                if getattr(itr, "id", None):
                    interrupts.append(itr)
        return interrupts
    except Exception as e:
        logger.warning(f"[Agent-任务] 解析内层中断失败（按无悬挂中断处理）: {e}")
        return []


async def _build_resume_command(agent, invoke_config: dict, resume_payload: Any) -> Any:
    """
    §20 F3.5：为恢复轮构造精确的 `Command(resume=...)`。

    分三种情况：

    1. **无悬挂中断**：说明中断已被消费或从未产生。原样传裸 resume 值，
       让 LangGraph 按老路径处理（不改变既有行为）。
    2. **恰好一个悬挂中断**：传裸 resume 值。这是最理想的情形，语义清晰，
       且不与 `Command(resume={id: value})` 的映射语义混淆。
    3. **多个悬挂中断**：必须按中断 id 定向投递，且**只投给第一个**
       （即真正等待本次决策的那一个）。其余保留悬挂 —— 它们来自更早的
       残留轮次，不该被本次审批决策消费掉。
       langgraph 1.2.10 支持 `{interrupt_id: value}` 映射（见
       `pregel/_loop.py` 的 `CONFIG_KEY_RESUME_MAP` 与 `_algo.py` 的
       `resume_map` 分支），映射命中时会追加到该任务的 resume 列表。

    :param agent: 内层任务 Agent
    :param invoke_config: 内层运行配置（含独立 thread_id）
    :param resume_payload: 审批/追问决策载荷
    :return: 传给 `agent.ainvoke` 的输入
    """
async def _build_resume_command(agent, invoke_config: dict, resume_payload: Any) -> Any:
    """
    §20 F3.5 + §22 修复 1/4：为恢复轮构造精确的 `Command(resume=...)`。

    分四种情况：

    1. **无悬挂中断**：说明中断已被消费或从未产生。原样传裸 resume 值，
       让 LangGraph 按老路径处理（不改变既有行为）。
    2. **单悬挂中断且与载荷配对**：传裸 resume 值。这是最理想的情形，语义清晰。
    3. **单悬挂中断但**与载荷**类型不匹配**：§22 新增 —— **拒绝投递**，
       抛 `ResumeMismatchError`。投递必然 `KeyError: 'decisions'`
       （见 `langchain/agents/middleware/human_in_the_loop.py:435` 的下标取值），
       且回落重投无效，必须在投递前拦下。
    4. **多个悬挂中断**：§22 修复 4 —— **按载荷类型筛选候选**后再定向投递，
       而不是盲投 `interrupt_list[0]`（列表首项未必是本次决策对应的那个）。
       候选恰好一个时按 `{interrupt_id: value}` 定向投递；若一个都不匹配，
       与情况 3 同样拒绝。

    定向投递的语义依据：`Interrupt.id = xxh3_128_hexdigest(ns)`，
    而 `_scratchpad` 的 `namespace_hash = xxh3_128_hexdigest(task_checkpoint_ns)`
    与之同源（`langgraph/pregel/_algo.py:1039`、`langgraph/types.py` 的
    `Interrupt.from_ns`），故 `{id: value}` 映射能精确命中该任务。

    :param agent: 内层任务 Agent
    :param invoke_config: 内层运行配置（含独立 thread_id）
    :param resume_payload: 审批/追问决策载荷
    :return: 传给 `agent.astream` / `agent.ainvoke` 的输入
    :raises ResumeMismatchError: 载荷与悬挂中断类型不匹配（确定性错误）
    """
    interrupt_list = await _inner_pending_interrupts(agent, invoke_config)

    if not interrupt_list:
        logger.warning(
            "[Agent-任务] 恢复轮未在内层发现悬挂中断（可能已被消费），"
            "按裸 resume 值继续"
        )
        metrics.incr("agent.task.resume.no_hanging_interrupt")
        return Command(resume=resume_payload)

    # §22 修复 1/4：按载荷类型筛选候选（不再盲取 interrupt_list[0]）
    target, mismatch = _match_resume_target(resume_payload, interrupt_list)

    if target is None:
        # 载荷与中断类型不匹配 -> 确定性错误，拒绝投递
        payload_kind = _classify_resume_payload(resume_payload) or "未知"
        logger.error(
            f"[Agent-任务] 恢复载荷与悬挂中断类型不匹配，拒绝投递: "
            f"payload={payload_kind}, 悬挂中断数={len(interrupt_list)}, "
            f"详情={mismatch}, trace_id={get_trace_id()}"
        )
        metrics.incr(K_TASK_RESUME_PAYLOAD_MISMATCH)
        raise ResumeMismatchError(mismatch)

    if len(interrupt_list) > 1:
        # 多中断：按类型筛选后定向投递（只投目标那一个，其余保留悬挂）
        payload_kind = _classify_resume_payload(resume_payload) or "未知"
        logger.warning(
            f"[Agent-任务] 内层存在 {len(interrupt_list)} 个悬挂中断，"
            f"按载荷类型（{payload_kind}）筛选后定向投递: target={target.id}"
        )
        metrics.incr("agent.task.resume.multi_interrupt")
        return Command(resume={target.id: resume_payload})

    # 单中断且配对成功 -> 裸 resume（语义最清晰）
    logger.info(
        f"[Agent-任务] 内层单个悬挂中断且与载荷配对，使用裸 resume 值: "
        f"id={target.id} kind={_classify_interrupt(target) or '未知'}"
    )
    return Command(resume=resume_payload)


def _is_approval_granted(resume_payload: Any) -> bool:
    """判断恢复载荷是否为「已批准」（decisions 非空且全是 approve）。"""
    if not isinstance(resume_payload, dict):
        return False
    decisions = resume_payload.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        return False
    return all(
        isinstance(d, dict) and d.get("type") == "approve" for d in decisions
    )


# =============================================================================
# §22 修复 3：澄清答案「退化」告警
# =============================================================================

def _has_write_intent_in_clarify_answers(resume_payload: Any) -> bool:
    """
    判断本轮恢复载荷是否为**追问答案**（且未被取消）。

    取消（`{"canceled": True}`）不算 —— 用户主动放弃补充信息时，
    模型如实告知并给建议是**预期行为**，不该计为退化。
    """
    if not isinstance(resume_payload, dict):
        return False
    if resume_payload.get("canceled"):
        return False
    answers = resume_payload.get("answers")
    return isinstance(answers, list) and bool(answers)


def _warn_clarify_answer_without_write(
    resume_payload: Any,
    executor: DbToolExecutor,
    kb_executor: Optional[KnowledgeToolExecutor],
) -> None:
    """
    §22 修复 3：检测「用户已给出追问答案，但本轮最终没有任何写操作调用」的退化。

    生产 trace=07dbbe8d604a4106：用户答完 3 个追问，模型却重新列了一遍表结构、
    把问题抛回用户（「请提供筛选条件」），写操作从未发起。用户视角是
    「我问也回答了，它还是没做」。

    与 KeyError 的关系：**独立缺陷**。KeyError 会让这一轮整轮作废；
    但即使 KeyError 修好，模型仍可能「答完又问」。故单列指标让它可见。

    非致命：只计数 + 告警日志，不改任务结果、不改状态机。
    注意判据是「有写工具**调用**」，不是「写操作成功」——
    调用后进入审批等待同样算「已推进」，不算退化。

    :param resume_payload: 本轮恢复载荷
    :param executor: 数据库工具执行器（审计来源）
    :param kb_executor: 知识库工具执行器（可为 None）
    """
    if not _has_write_intent_in_clarify_answers(resume_payload):
        return
    try:
        calls = _merge_tool_calls(executor, kb_executor)
        has_write_call = any(
            isinstance(c, dict) and str(c.get("tool", "")) in WRITE_TOOL_NAMES
            for c in calls
        )
        if has_write_call:
            return
        metrics.incr(K_TASK_CLARIFY_ANSWER_NO_WRITE)
        logger.warning(
            f"[Agent-任务] 已获得追问答案但本轮未发起任何写操作调用"
            f"（疑似把问题抛回用户）：本轮工具调用={len(calls)} 次, "
            f"trace_id={get_trace_id()}"
        )
    except Exception as e:  # 观测不得影响主流程
        logger.warning(f"[Agent-任务] 追问退化检测失败（非致命）: {e}")


async def _absorb_reissued_interrupt(
    agent,
    invoke_config: dict,
    result: Any,
    resume_payload: Any,
    tool_calls: List[dict],
) -> Any:
    """
    §20 F3.5b：就地消化「因模型原地重发写操作而产生的新中断」。

    场景（生产 trace=00b0b22f2b0446b7）：
      1. `Command(resume=approve)` 消费旧中断，写工具**执行成功**；
      2. ToolMessage 回灌后，模型**又发一次**同一个写操作；
      3. `after_model` 生成**新中断**，落在 `result["__interrupt__"]`；
      4. 外层 `resume_graph` 据此判「决策未生效」→ 标 failed。

    这是**误判**：工具A已经执行了，用户的目标已达成。此处把它纠正为成功。

    只有同时满足以下条件才「消化」：
      - 恢复载荷是**已批准**的（reject/canceled 时的新中断是真信号，不可吞）；
      - 本轮**确有写操作执行成功**（`_successful_writes` 非空）。

    否则原样返回，让中断照常冒泡给 `approval_service`（保持 F3.1 护栏效力）。

    :return: 修正后的 result（消化时是去掉 `__interrupt__` 的副本）
    """
    if not isinstance(result, dict):
        return result
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return result

    writes = _successful_writes(tool_calls)
    if not _is_approval_granted(resume_payload):
        logger.warning(
            f"[Agent-任务] 恢复轮再次中断，但决策非「批准」，不消化"
            f"（中断数={len(interrupts)}, trace_id={get_trace_id()}）"
        )
        return result
    if not writes:
        logger.warning(
            f"[Agent-任务] 恢复轮再次中断且本轮无写操作成功，不消化"
            f"（中断数={len(interrupts)}, trace_id={get_trace_id()}）"
        )
        return result

    logger.warning(
        f"[Agent-任务] 检测到「批准后模型原地重发写操作」产生的冗余中断，已就地消化: "
        f"已执行=[{_describe_writes(writes)}], "
        f"中断数={len(interrupts)}, trace_id={get_trace_id()}"
    )
    metrics.incr("agent.task.resume.reissue_absorbed")

    # 注意：**不要**在这里调 `agent.aupdate_state(...)` 去"清"内层中断。
    # 实测 `aupdate_state(cfg, None, as_node="HumanInTheLoopMiddleware.after_model")`
    # 虽然能清掉 interrupts，但会把 `next` 置为 `('tools',)` —— 相当于让
    # 已批准的写操作**再执行一遍**（重复删除/重复扣款级事故）。
    #
    # ⚠️ §24 修正：冗余中断留在内层**并非「无害」**。原注释断言
    #   「下次该会话的新一轮任务会带上新的 task_execution_id（新内层 thread），
    #    不会误命中」—— **该假设是错的**：
    #   `_find_running_task` 是**按会话**查最近的 running 记录，本轮若因任何原因
    #   残留 running（状态未同步 / 异常退出），下一轮就会复用同一 task_execution_id
    #   → 同一内层 thread → 旧中断被带出 → `after_model` 再 interrupt
    #   → 用户看到**上一轮已处理过的审批卡片**又冒出来（生产事故）。
    #
    #   因此正确做法是**双层防护**：
    #   ① 本函数在返回值层面消化（让本轮按成功上报，不再重复弹卡片）；
    #   ② `_find_running_task(reject_stale=True)` 在**下一轮入口**拒收
    #      「内层已无悬挂中断」的陈旧 running 记录，从根上断开 thread 复用。
    #   两者缺一不可：只有 ① 会让中断留在检查点，只有 ② 会让本轮误报失败。
    cleaned = dict(result)
    cleaned.pop("__interrupt__", None)
    return cleaned


# =============================================================================
# 审计落库
# =============================================================================

async def _find_running_task(
    db, conversation_id: str, *, reject_stale: bool = True
) -> Optional[TaskExecution]:
    """
    查找会话最近的 running 状态任务记录（审批恢复时复用）。

    §24 修复：`reject_stale` 控制是否**拒收陈旧记录**。

    「陈旧」的判据是「该记录对应的内层图**已无悬挂中断**」—— 那种情况下它
    并不是「正在等待人工介入」，而是「已终结但状态没同步过去」的僵尸记录。
    复用它会带来两个后果：

    1. **跨轮串味**：新轮拿到旧 thread_id → 内层图加载旧检查点 → 旧
       `last_ai_msg.tool_calls` 仍在 → `after_model` **再抛一次旧的中断** →
       用户看到**上一轮已处理过的审批卡片**又冒出来（§24 报告的主症状）。
    2. **审计混串**：`executor.tool_calls` 被灌入上一轮的写操作记录 →
       `_successful_writes` 把旧写操作算作本轮成果 → F3.5b 误判「已执行」。

    :param db: 数据库会话
    :param conversation_id: 会话 ID
    :param reject_stale: 拒收内层已无悬挂中断的 running 记录（默认 True）。
        仅用于需要**绝对复用**的边界场景才传 False。
    :return: 可复用的 running 记录；无可用记录时 None
    """
    try:
        from sqlalchemy import select
        result = await db.execute(
            select(TaskExecution)
            .where(
                TaskExecution.conversation_id == conversation_id,
                TaskExecution.status == "running",
            )
            .order_by(TaskExecution.created_at.desc())
            .limit(1)
        )
        record = result.scalars().first()
    except Exception as e:
        await db.rollback()
        logger.warning(f"[Agent-任务] 查询任务记录失败（非致命）: {e}")
        return None

    if record is None:
        return None

    # §24：陈旧记录拒收（内层已无悬挂中断 = 该轮已终结，状态未同步）
    if reject_stale:
        try:
            stale = await _is_stale_running_record(record)
        except Exception as e:
            # 判据失败时**保守放行**（保持旧行为）—— 宁可偶发串味，
            # 也不要因为探测异常把真正的「待审批恢复」拦掉（那会让审批失效）。
            logger.warning(f"[Agent-任务] 陈旧记录判据失败（保守放行）: {e}")
            stale = False
        if stale:
            metrics.incr(K_TASK_STALE_RUNNING_REJECTED)
            logger.warning(
                f"[Agent-任务] 检测到陈旧 running 记录（内层已无悬挂中断），"
                f"拒绝复用并新建本轮任务: task_execution_id={record.id}, "
                f"trace_id={get_trace_id()}"
            )
            return None

    return record


async def _is_stale_running_record(record: TaskExecution) -> bool:
    """
    §24：判断一条 running 任务记录是否为「已终结但状态未同步」的僵尸。

    判据：**该记录对应的内层 thread 上已无悬挂中断**。

    只有「正在等待人工介入」的任务才应该被复用 —— 那种情况下内层图必然
    停在一个未消费的中断上。反之，内层无悬挂中断说明：
      - 该轮已跑完（可能落了 completed，也可能因异常没落）；
      - 或内层图从未真正启动过（run 失败）；
      - 或中断已被消费且没有新中断。

    三种情况都**不该**被新轮复用。

    实现要点：用 `checkpointer.aget_tuple(config)` 读**检查点元组**（轻量、只读），
    而不是 `aget_state`（后者需要 Pregel 实例）。两者的字段位置不同，必须注意：

    - `CheckpointTuple` 没有 `next` / `tasks` 属性；
      `next` 在 `tuple.checkpoint["next"]` 里（`PendingWrite` 在 `pending_writes`）；
    - `StateSnapshot`（`aget_state` 的返回）才有顶层的 `next` / `tasks` / `interrupts`。

    悬挂中断的判据取 `pending_writes` —— HITL/ask_user 中断都会在检查点里留下
    未消费的待写记录；`next` 非空则说明图停在中间（可能在 tools 里，也可能中断）。
    两者都为空即认为该轮已终结。

    :return: True = 陈旧（不应复用）；False = 可能仍有悬挂中断（应复用）
    """
    thread_id = _inner_thread_id(record.conversation_id, record.id)
    config = {"configurable": {"thread_id": thread_id}}
    try:
        from agent.graph_builder import get_checkpointer
        checkpointer = get_checkpointer()
    except Exception as e:
        logger.warning(f"[Agent-任务] 获取 checkpointer 失败（保守放行）: {e}")
        return False

    if checkpointer is None:
        # 无检查点 → 不存在「内层悬挂中断」概念（非审批模式），
        # 此时复用/新建都无串味风险，按「不陈旧」处理保持旧行为。
        return False

    try:
        cpt = await checkpointer.aget_tuple(config)
    except Exception as e:
        logger.warning(f"[Agent-任务] 读取内层检查点失败（保守放行）: {e}")
        return False

    if cpt is None:
        # 内层 thread 根本不存在 → 该轮从未真正进入内层图 → 陈旧
        return True

    try:
        pending_writes = getattr(cpt, "pending_writes", None) or ()
        if pending_writes:
            # 有未消费的待写记录 → 极可能停在中断上 → 不陈旧（应复用）
            return False
        checkpoint = getattr(cpt, "checkpoint", None) or {}
        nxt = checkpoint.get("next") or ()
        if nxt:
            # 图停在中间（next 非空）：可能在中断，也可能在 tools。
            # 该场景判据不足 → 保守放行（宁可复用，也不要拦掉真正的恢复）。
            return False
        # next 为空且无待写 → 该轮已跑完（图已到 END）→ 陈旧
        return True
    except Exception as e:
        logger.warning(f"[Agent-任务] 解析内层检查点失败（保守放行）: {e}")
        return False


async def _close_stale_running_tasks(db, conversation_id: str) -> int:
    """
    §24：把本会话所有「陈旧」的 running 任务记录收尾为 `completed`。

    触发时机：新轮进入时 `_find_running_task` 拒收了陈旧记录 →
    说明该记录对应的内层图已终结，只是状态没同步过来。必须落库收尾，
    否则它会一直以 running 存在，被后续每一轮反复查询、反复拒收。

    判据与 `_is_stale_running_record` 一致（内层无悬挂中断）。
    逐条判定而非批量 UPDATE —— 批量会把「真在等审批」的记录一并误杀
    （同一会话理论上不应有两条 running，但检查点异常时可能出现）。

    全部 best-effort：任何失败只告警，不影响本轮任务。

    :return: 实际收尾的记录数
    """
    try:
        from sqlalchemy import select
        result = await db.execute(
            select(TaskExecution).where(
                TaskExecution.conversation_id == conversation_id,
                TaskExecution.status == "running",
            )
        )
        records = list(result.scalars().all())
    except Exception as e:
        await db.rollback()
        logger.warning(f"[Agent-任务] 查询僵尸 running 记录失败（非致命）: {e}")
        return 0

    closed = 0
    for rec in records:
        try:
            if not await _is_stale_running_record(rec):
                continue
            rec.status = "completed"
            if not rec.result:
                rec.result = "（该轮已结束，状态由后续轮次同步收尾）"
            closed += 1
        except Exception as e:
            logger.warning(
                f"[Agent-任务] 收尾僵尸记录失败（跳过）: id={rec.id}, {e}"
            )

    if closed:
        try:
            await db.commit()
        except Exception as e:
            await db.rollback()
            logger.warning(f"[Agent-任务] 收尾僵尸记录提交失败（非致命）: {e}")
            return 0
        metrics.incr(K_TASK_STALE_RUNNING_CLOSED, closed)
        logger.warning(
            f"[Agent-任务] 已收尾 {closed} 条陈旧 running 记录"
            f"（避免被后续轮次复用）: conversation_id={conversation_id}, "
            f"trace_id={get_trace_id()}"
        )
    return closed


async def _create_task_record(
    db, task_execution_id: str, conversation_id: str,
    user_id: Optional[str], intent: str,
) -> None:
    """创建任务执行记录（status=running）。"""
    try:
        record = TaskExecution(
            id=task_execution_id,
            conversation_id=conversation_id,
            user_id=user_id,
            intent=intent,
            status="running",
        )
        db.add(record)
        await db.commit()
    except Exception as e:
        await db.rollback()
        logger.warning(f"[Agent-任务] 创建审计记录失败（非致命）: {e}")


async def _update_task_record(
    db, task_execution_id: str, status: Optional[str] = None,
    task_answer: Optional[str] = None,
    error: Optional[str] = None,
    tool_calls: Optional[List[dict]] = None,
) -> None:
    """
    更新任务执行记录（completed / failed / rejected / running）。

    §20 F3.7：`status=None` 表示**只更新审计字段、不动状态机** ——
    供「写操作已执行」的独立落库使用（此时任务仍在跑，不该改状态）。
    """
    try:
        from sqlalchemy import select
        result = await db.execute(
            select(TaskExecution).where(TaskExecution.id == task_execution_id)
        )
        record = result.scalars().first()
        if record is None:
            return
        if status is not None:
            record.status = status
        if task_answer is not None:
            record.task_answer = task_answer
        if error is not None:
            record.error = error
        if tool_calls is not None:
            record.tool_calls = tool_calls
        await db.commit()
    except Exception as e:
        await db.rollback()
        logger.warning(f"[Agent-任务] 更新审计记录失败（非致命）: {e}")


async def _save_conversation_messages(
    db, conversation_id: str, user_input: str, answer: str
) -> None:
    """任务轮次的对话消息落库（best-effort，失败不阻塞主流程）。

    与聊天/知识库节点一致写入 conversation_message，保证重新进入会话时
    任务轮次不丢失；审批中断路径不调用（恢复完成后由终结路径统一写入一次，
    避免中断/恢复重入导致重复记录）。

    :param db: 数据库会话
    :param conversation_id: 会话 ID（=thread_id）
    :param user_input: 用户提问
    :param answer: Agent 最终回答（成功为 task_answer，失败为错误提示）
    """
    if not conversation_id:
        return
    try:
        mysql_history = MySQLChatMessageHistory(
            session=db, conversation_id=conversation_id
        )
        await mysql_history.add_message(HumanMessage(content=user_input))
        if answer:
            await mysql_history.add_message(AIMessage(content=answer))
        await db.commit()
    except Exception as e:
        await db.rollback()
        logger.warning(f"[Agent-任务] 对话消息落库失败（非致命）: {e}")
