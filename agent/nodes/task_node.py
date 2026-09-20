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
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphInterrupt, GraphRecursionError

from core.config import settings
from prompt.prompt_storage import TASK_AGENT_SYSTEM_PROMPT
from models.task_model import TaskExecution
from rag.memory_mysql import MySQLChatMessageHistory
from agent.state import AgentState
from agent.streaming import get_sse_queue, put_content, put_status
from agent.tools.db_tools import DbToolExecutor
from agent.tools.knowledge_tool import KnowledgeToolExecutor
from agent.tools.manager import bind_active_tools
from tools.clarify_tool import ASK_USER_TOOL_NAME
from agent.observability import get_trace_id, metrics, node_trace

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

    # 1. 任务记录（幂等）：恢复执行时复用中断前创建的 running 记录
    record = await _find_running_task(db, conversation_id)
    if record is None:
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

    try:
        # 2. 构建任务 Agent（审批模式绑定全部生效工具；工具从 tool_registry 查询 status=1 生效项）
        agent = await _build_task_agent(model, db, executor, kb_executor, approval_mode)

        # 3. 执行任务（恢复执行时从检查点继续，审批决策自动路由）
        # 注入 recursion_limit 覆盖 create_agent 默认值，防模型-工具死循环
        # 注入 recursion_limit 覆盖 create_agent 默认值，防模型-工具死循环。
        # 注：config 可能为 None（离线/单测场景），必须兜底，否则 {**None} 抛 TypeError。
        logger.info(
            f"[Agent-任务] 开始执行: task_execution_id={task_execution_id}, "
            f"approval_mode={approval_mode}, trace_id={get_trace_id()}"
        )
        invoke_config = {
            **(config or {}),
            "recursion_limit": _recursion_limit(),
        }
        if approval_mode:
            # 审批模式：不设整体墙钟超时（用户审批可能挂起较久，审批等待超时由
            # 审批侧 APPROVAL_WAIT_TIMEOUT 语义负责），仍受工具级/LLM 超时与递归上限保护
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content=user_input)]}, config=invoke_config
            )
        else:
            # 非审批模式：Agent 循环整体墙钟超时兜底
            result = await asyncio.wait_for(
                agent.ainvoke(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=invoke_config,
                ),
                timeout=settings.agent_task_timeout_seconds,
            )
        task_answer = _extract_answer(result)

        # 4. 推送最终回答到 SSE（任务结果非流式，一次性推送）
        if queue is not None and task_answer:
            await put_content(queue, task_answer)

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
        }
    except GraphInterrupt:
        # 审批中断：持久化已发生的工具调用审计后向上传播（等待审批恢复）
        # 状态保持 running：审批恢复时任务节点重入并复用该记录
        await _update_task_record(
            db, task_execution_id, status="running",
            tool_calls=_merge_tool_calls(executor, kb_executor),
        )
        raise
    except GraphRecursionError:
        # 模型-工具循环达到 recursion_limit 上限（防死循环硬保护）
        error_msg = "任务执行超过最大尝试次数，请将请求拆分或简化后重试"
        if queue is not None:
            await put_content(queue, error_msg)
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
            await put_content(queue, error_msg)
        await _update_task_record(
            db, task_execution_id, status="failed",
            error=error_msg, tool_calls=_merge_tool_calls(executor, kb_executor),
        )
        await _save_conversation_messages(db, conversation_id, user_input, error_msg)
        return {"task_answer": None, "error": error_msg}
    except Exception as e:
        logger.exception(f"[Agent-任务] 执行失败: {e}")
        error_msg = f"任务执行失败：{str(e)}"
        # 推送可读错误到 SSE（不中断连接）
        if queue is not None:
            await put_content(queue, error_msg)
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
    llm = ChatOpenAI(
        model=model,
        openai_api_key=settings.api_key,
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.4,
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
# 审计落库
# =============================================================================

async def _find_running_task(db, conversation_id: str) -> Optional[TaskExecution]:
    """查找会话最近的 running 状态任务记录（审批恢复时复用）。"""
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
        return result.scalars().first()
    except Exception as e:
        await db.rollback()
        logger.warning(f"[Agent-任务] 查询任务记录失败（非致命）: {e}")
        return None


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
    db, task_execution_id: str, status: str,
    task_answer: Optional[str] = None,
    error: Optional[str] = None,
    tool_calls: Optional[List[dict]] = None,
) -> None:
    """更新任务执行记录（completed / failed / rejected）。"""
    try:
        from sqlalchemy import select
        result = await db.execute(
            select(TaskExecution).where(TaskExecution.id == task_execution_id)
        )
        record = result.scalars().first()
        if record is None:
            return
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
