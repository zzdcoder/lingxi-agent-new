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
import logging
import time
import uuid
from typing import Any, List, Optional

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphInterrupt

from core.config import settings
from prompt.prompt_storage import TASK_AGENT_SYSTEM_PROMPT
from models.task_model import TaskExecution
from agent.state import AgentState
from agent.streaming import get_sse_queue, put_content, put_status
from agent.tools.db_tools import DbToolExecutor, DbToolError
from agent.tools.registry import ToolSpec, get_tool

logger = logging.getLogger(__name__)


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
        # 2. 构建任务 Agent（审批模式绑定全部工具）
        agent = _build_task_agent(model, executor, approval_mode)

        # 3. 执行任务（恢复执行时从检查点继续，审批决策自动路由）
        logger.info(f"[Agent-任务] 开始执行: task_execution_id={task_execution_id}")
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=user_input)]}, config=config
        )
        task_answer = _extract_answer(result)

        # 4. 推送最终回答到 SSE（任务结果非流式，一次性推送）
        if queue is not None and task_answer:
            await put_content(queue, task_answer)

        # 5. 更新审计（completed）
        await _update_task_record(
            db, task_execution_id, status="completed",
            task_answer=task_answer,
            tool_calls=executor.tool_calls,
        )
        logger.info(
            f"[Agent-任务] 完成: task_execution_id={task_execution_id}, "
            f"耗时={int((time.perf_counter() - start) * 1000)}ms, "
            f"工具调用={len(executor.tool_calls)} 次"
        )
        return {
            "task_answer": task_answer,
            "task_execution_id": task_execution_id,
            "tool_calls": executor.tool_calls,
        }
    except GraphInterrupt:
        # 审批中断：持久化已发生的工具调用审计后向上传播（等待审批恢复）
        # 状态保持 running：审批恢复时任务节点重入并复用该记录
        await _update_task_record(
            db, task_execution_id, status="running", tool_calls=executor.tool_calls
        )
        raise
    except Exception as e:
        logger.exception(f"[Agent-任务] 执行失败: {e}")
        error_msg = f"任务执行失败：{str(e)}"
        # 推送可读错误到 SSE（不中断连接）
        if queue is not None:
            await put_content(queue, error_msg)
        await _update_task_record(
            db, task_execution_id, status="failed",
            error=error_msg, tool_calls=executor.tool_calls,
        )
        return {"task_answer": None, "error": error_msg}


# =============================================================================
# 任务 Agent 构建
# =============================================================================

def _build_task_agent(model: str, executor: DbToolExecutor, approval_mode: bool):
    """
    构建任务执行 Agent。

    审批模式（checkpointer 可用）：绑定全部工具，写操作经
    HumanInTheLoopMiddleware 审批后执行；无审批模式仅绑定只读 + insert。
    """
    llm = ChatOpenAI(
        model=model,
        openai_api_key=settings.api_key,
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.4
    )
    tools = _bind_tools(executor, approval_mode)

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


def _bind_tools(executor: DbToolExecutor, approval_mode: bool) -> List[BaseTool]:
    """将注册表中的工具绑定到 executor 实例方法，生成 LangChain Tool 列表。"""
    tools: List[BaseTool] = []
    # 审批模式绑定全部工具；无审批模式仅只读 + insert（insert 默认不审批）
    spec_names = ["list_tables", "query_data"]
    if approval_mode or not settings.agent_insert_requires_approval:
        spec_names.append("insert_data")
    if approval_mode:
        spec_names.extend(["update_data", "delete_data"])

    for name in spec_names:
        spec: Optional[ToolSpec] = get_tool(name)
        if spec is None:
            continue
        handler = getattr(executor, spec.handler, None)
        if handler is None:
            logger.warning(f"[Agent-任务] 工具 {name} 无对应执行方法，跳过")
            continue
        tools.append(
            StructuredTool.from_function(
                coroutine=handler,
                name=spec.name,
                description=spec.description,
            )
        )
    return tools


def _extract_answer(result: Any) -> str:
    """从 create_agent 的返回结构中提取最终回答。"""
    messages = result.get("messages") if isinstance(result, dict) else result
    if isinstance(messages, (list, tuple)) and messages:
        last = messages[-1]
        content = getattr(last, "content", None)
        if content:
            return content if isinstance(content, str) else str(content)
    return "任务已完成（无文本结果）"


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
