"""
运行时工具管理器（设计文档 §7.8 / §7.9）

在 task 处理节点需要调用工具时：

1. **查询生效工具**：从 tool_registry 表查询 `status=1`（生效）的工具，
   结合审批模式过滤（非审批模式剔除 requires_approval 的写工具）；
2. **bind_tools**：为每个生效工具解析运行时执行函数（executor 方法 / @tool 对象），
   统一注入：
   - 超时保护（agent_tool_timeout_seconds）；
   - 熔断器（ToolCircuitBreaker：调用失败计数、自动熔断 status=2、快速失败）；
3. **Schema 惰性回填**：注册时未能提取参数 Schema 的工具，首次绑定时补写。

受益：熔断/下线一个工具，仅需把表状态置 2/0，下次任务即不再绑定该工具，
实现"单工具粒度的故障隔离"，减小影响面。
"""
import asyncio
import functools
import logging
from typing import Any, Callable, List, Optional

from langchain_core.tools import BaseTool, StructuredTool

from core.config import settings
from models.tool_model import ToolRegistry, ToolStatus
from agent.tools.circuit_breaker import breaker
from agent.tools.db_tools import DbToolError
from agent.tools.registrar import get_decorated_tools

logger = logging.getLogger(__name__)


def tool_error_content(e: Exception) -> str:
    """
    工具执行失败的模型可见消息（作为 ToolMessage 送回 Agent 循环）。

    基于已脱敏的异常消息（不暴露 SQL/参数值/堆栈）附加修复引导，
    供模型决策"修正参数重试 / 换工具 / 如实告知用户无法完成"。
    仅对 ToolException 子类（如 DbToolError、CircuitOpenError）生效。
    """
    return f"工具执行失败：{e}。请修正参数后重试；若仍失败，请如实告知用户暂时无法完成该操作。"


def with_tool_timeout(handler: Callable, timeout: float):
    """
    为工具调用包裹统一超时（单次调用总时长上限）。

    - functools.wraps 保留原 handler 签名，供 StructuredTool.from_function
      推断参数 schema（否则 wrapper(**kwargs) 会退化 schema，模型无法生成参数）；
    - 超时抛 DbToolError（ToolException 子类），经 handle_tool_error 转为
      ToolMessage 送回 Agent 循环（模型可感知超时并应对），而非中断循环。
    """
    @functools.wraps(handler)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        try:
            return await asyncio.wait_for(handler(*args, **kwargs), timeout=timeout)
        except asyncio.TimeoutError:
            raise DbToolError(f"工具调用超时（>{timeout}s）") from None
    return wrapped


# =============================================================================
# 查询生效工具
# =============================================================================

async def load_active_tools(db, approval_mode: bool) -> List[ToolRegistry]:
    """
    查询工具表中状态=1（生效）的工具，并按审批模式过滤。

    :param db: 数据库会话
    :param approval_mode: 是否审批模式（checkpointer 可用）；False 时剔除需审批的写工具
    :return: 生效工具记录列表（空列表表示查询失败，降级为无工具可绑）
    """
    try:
        from sqlalchemy import select

        result = await db.execute(
            select(ToolRegistry)
            .where(ToolRegistry.status == ToolStatus.ACTIVE)
            .order_by(ToolRegistry.category, ToolRegistry.name)
        )
        rows = list(result.scalars().all())
    except Exception as e:
        await db.rollback()
        logger.warning(f"[工具管理] 查询生效工具失败（返回空）: {e}")
        return []
    if approval_mode:
        return rows
    return [r for r in rows if not r.requires_approval]


# =============================================================================
# 绑定工具
# =============================================================================

async def bind_active_tools(
    db,
    executor,
    kb_executor,
    approval_mode: bool,
) -> List[BaseTool]:
    """
    查询生效工具并生成 LangChain Tool 列表（bind_tools 前身）。

    每个工具统一包裹：超时（with_tool_timeout）+ 熔断器（breaker.call）。

    :param db: 数据库会话（熔断器持久化与生效工具查询共用）
    :param executor: DbToolExecutor 实例（数据库工具执行器）
    :param kb_executor: KnowledgeToolExecutor 实例（知识库检索执行器）
    :param approval_mode: 是否审批模式
    :return: 可绑定给任务 Agent 的 Tool 列表
    """
    rows = await load_active_tools(db, approval_mode)
    tools: List[BaseTool] = []

    for row in rows:
        handler = _resolve_handler(row, executor, kb_executor)
        if handler is None:
            continue
        try:
            if isinstance(handler, BaseTool):
                tool = _wrap_decorated_tool(db, row, handler)
            else:
                tool = StructuredTool.from_function(
                    coroutine=_breaker_wrap(
                        db, row.name,
                        with_tool_timeout(handler, settings.agent_tool_timeout_seconds),
                    ),
                    name=row.name,
                    description=row.description or "",
                    handle_tool_error=tool_error_content,
                )
            await _lazy_refresh_schema(db, row, tool)
            tools.append(tool)
        except Exception as e:
            logger.warning(f"[工具管理] 绑定工具 {row.name} 失败，跳过: {e}")

    logger.info(f"[工具管理] 绑定生效工具 {len(tools)} 个: {[t.name for t in tools]}")
    return tools


def _resolve_handler(
    row: ToolRegistry, executor, kb_executor
) -> Optional[Callable]:
    """
    根据工具来源（source 列）解析运行时执行函数。

    - executor:<方法名>：从 DbToolExecutor / KnowledgeToolExecutor 实例取方法；
    - module:...:<属性名>：取扫描到的 @tool 装饰器工具对象（BaseTool 本身）；
    - 兜底：按工具名在注册表中查找 handler 名并解析。
    """
    source = row.source or ""
    if source.startswith("executor:"):
        method = source.split(":", 1)[1]
        for h in (executor, kb_executor):
            fn = getattr(h, method, None) if h is not None else None
            if fn is not None:
                return fn
        logger.warning(f"[工具管理] 工具 {row.name} 无对应执行方法 {method!r}，跳过")
        return None

    if source.startswith("module:"):
        tool = get_decorated_tools().get(row.name)
        if tool is None:
            logger.warning(f"[工具管理] 工具 {row.name} 的 @tool 对象未找到，跳过")
        return tool

    # 兼容无 source 的存量数据：按注册表 handler 名解析
    try:
        from agent.tools.registry import get_tool
        spec = get_tool(row.name)
        if spec is not None:
            for h in (executor, kb_executor):
                fn = getattr(h, spec.handler, None) if h is not None else None
                if fn is not None:
                    return fn
    except Exception:
        pass
    logger.warning(f"[工具管理] 工具 {row.name} 无法解析执行函数，跳过")
    return None


def _breaker_wrap(db, tool_name: str, handler: Callable):
    """熔断器包裹：每次调用经 ToolCircuitBreaker 判定/上报。"""
    @functools.wraps(handler)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        return await breaker.call(db, tool_name, handler, *args, **kwargs)
    return wrapped


def _wrap_decorated_tool(db, row: ToolRegistry, tool: BaseTool) -> BaseTool:
    """
    包裹 @tool 装饰器工具：保留原参数 Schema，注入超时 + 熔断。

    StructuredTool.from_function 支持显式传入 args_schema，
    因此执行体（**kwargs → tool.ainvoke）不会退化模型可见的参数定义。
    """
    async def _run(**kwargs: Any) -> Any:
        return await tool.ainvoke(kwargs)

    return StructuredTool.from_function(
        coroutine=_breaker_wrap(
            db, row.name,
            with_tool_timeout(_run, settings.agent_tool_timeout_seconds),
        ),
        name=tool.name,
        description=tool.description or "",
        args_schema=tool.args_schema,          # 保留原始参数 Schema
        handle_tool_error=tool_error_content,
    )


async def _lazy_refresh_schema(db, row: ToolRegistry, tool: BaseTool) -> None:
    """注册时未提取到参数 Schema 的工具，首次绑定时惰性回填。"""
    if row.parameters:
        return
    try:
        schema = getattr(tool, "args_schema", None)
        if schema is not None:
            row.parameters = schema.model_json_schema()
            await db.commit()
    except Exception as e:
        await db.rollback()
        logger.debug(f"[工具管理] 回填工具 {row.name} 参数 Schema 失败（非致命）: {e}")
