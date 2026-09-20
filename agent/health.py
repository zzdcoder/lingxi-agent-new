"""
健康巡检（阶段 4 观测与加固，设计文档 §15.4）

Agent 链路依赖多（MySQL 业务库 + LangGraph 检查点 + Qdrant 检索/缓存 + 工具表），
任一环节降级都应**可被发现**，而不是等到用户报障。本模块提供统一的依赖体检，
输出机器可读的健康快照：

- `database`：业务库连通性（SELECT 1）；
- `checkpointer`：LangGraph MySQL 检查点是否可用（决定**审批模式**是否生效，
  不可用时写工具会被整体摘除，是最容易被忽略的静默降级）；
- `graph`：主图是否已编译；
- `tool_registry`：工具注册总量与失效/熔断分布（熔断数 >0 需告警）；
- `retrieval` / `cache`：混合检索器与语义缓存初始化状态。

设计要点：
1. **只读**：体检不修改任何数据；
2. **不抛异常**：任一组件探测失败记为 down 而非拖垮整个接口；
3. **轻量**：避免体检本身成为故障源（无重排模型推理、无大结果集查询）。
"""
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

STATUS_OK = "ok"
STATUS_DEGRADED = "degraded"
STATUS_DOWN = "down"

# 体检 SQL 超时（秒）：避免健康检查被慢查询拖住
_PROBE_TIMEOUT = 3.0


async def collect_health(session_factory) -> Dict[str, Any]:
    """
    收集服务健康快照。

    :param session_factory: 异步会话工厂（core.database.AsyncSessionLocal）
    :return: 健康快照字典（含 status / components / agent 摘要）
    """
    components: Dict[str, Any] = {}

    # 1. 业务数据库连通性
    components["database"] = await _probe_database(session_factory)

    # 2. LangGraph 检查点（审批模式开关）
    components["checkpointer"] = _probe_checkpointer()

    # 3. 主图编译状态
    components["graph"] = _probe_graph()

    # 4. 工具注册/熔断分布（依赖数据库，失败降级为 unknown）
    components["tool_registry"] = (
        await _probe_tool_registry(session_factory)
        if components["database"] == STATUS_OK
        else {"status": "unknown", "detail": "数据库不可用，跳过工具表检查"}
    )

    # 5. 检索与缓存（进程内单例状态）
    components["retrieval"] = _probe_component("rag.rag_conversation_service", "get_hybrid_retriever")
    components["cache"] = _probe_component("rag.semantic_cache", "get_semantic_cache")

    overall = STATUS_DOWN if components["database"] == STATUS_DOWN else STATUS_OK
    if any(
        c.get("status") != STATUS_OK
        for c in components.values()
        if isinstance(c, dict)
    ):
        overall = STATUS_DEGRADED

    approval_mode = components["checkpointer"]["status"] == STATUS_OK
    return {
        "status": overall,
        "approval_mode": approval_mode,
        "components": components,
    }


async def _probe_database(session_factory) -> Dict[str, Any]:
    """探测业务库连通性（SELECT 1，带超时）。"""
    import asyncio

    async def _check() -> Dict[str, Any]:
        try:
            async with session_factory() as session:
                from sqlalchemy import text

                await session.execute(text("SELECT 1"))
            return {"status": STATUS_OK}
        except Exception as e:
            logger.warning(f"[健康巡检] 数据库连通性检查失败: {e}")
            return {"status": STATUS_DOWN, "detail": str(e)[:200]}

    try:
        return await asyncio.wait_for(_check(), timeout=_PROBE_TIMEOUT)
    except asyncio.TimeoutError:
        return {"status": STATUS_DOWN, "detail": f"连通性检查超时(>{_PROBE_TIMEOUT}s)"}


def _probe_checkpointer() -> Dict[str, Any]:
    """探测 LangGraph MySQL 检查点（不可用 = 无审批模式，写工具被摘除）。"""
    try:
        from agent.graph_builder import get_checkpointer

        ok = get_checkpointer() is not None
        return {
            "status": STATUS_OK if ok else STATUS_DEGRADED,
            "detail": None if ok else "检查点不可用，当前为无审批模式（写工具不绑定）",
        }
    except Exception as e:
        return {"status": STATUS_DOWN, "detail": str(e)[:200]}


def _probe_graph() -> Dict[str, Any]:
    """探测主图编译状态（惰性构建，此处只读缓存，不触发重建）。"""
    try:
        from agent.graph_builder import _graph

        compiled = _graph is not None
        return {
            "status": STATUS_OK if compiled else STATUS_DEGRADED,
            "detail": None if compiled else "主图尚未编译（首次调用时构建）",
        }
    except Exception as e:
        return {"status": STATUS_DOWN, "detail": str(e)[:200]}


async def _probe_tool_registry(session_factory) -> Dict[str, Any]:
    """统计工具注册表状态分布（熔断工具数量是关键告警信号）。"""
    import asyncio

    from sqlalchemy import func, select

    from models.tool_model import ToolRegistry, ToolStatus

    async def _check() -> Dict[str, Any]:
        async with session_factory() as session:
            rows = await session.execute(
                select(ToolRegistry.status, func.count())
                .group_by(ToolRegistry.status)
            )
            dist = {int(status): int(count) for status, count in rows.all()}
            broken = dist.get(ToolStatus.CIRCUIT_BREAK, 0)
            return {
                "status": STATUS_OK if broken == 0 else STATUS_DEGRADED,
                "total": sum(dist.values()),
                "active": dist.get(ToolStatus.ACTIVE, 0),
                "disabled": dist.get(ToolStatus.DISABLED, 0),
                "circuit_broken": broken,
                "detail": None if broken == 0 else f"{broken} 个工具处于熔断状态",
            }

    try:
        return await asyncio.wait_for(_check(), timeout=_PROBE_TIMEOUT)
    except asyncio.TimeoutError:
        return {"status": STATUS_DOWN, "detail": "工具表检查超时"}
    except Exception as e:
        logger.warning(f"[健康巡检] 工具注册表检查失败: {e}")
        return {"status": STATUS_DOWN, "detail": str(e)[:200]}


def _probe_component(module_path: str, getter: str) -> Dict[str, Any]:
    """探测进程内单例组件是否已初始化（如混合检索器 / 语义缓存）。"""
    try:
        import importlib

        module = importlib.import_module(module_path)
        instance = getattr(module, getter)()
        initialized = instance is not None
        return {
            "status": STATUS_OK if initialized else STATUS_DEGRADED,
            "detail": None if initialized else "组件未初始化（降级模式运行）",
        }
    except Exception as e:
        return {"status": STATUS_DOWN, "detail": str(e)[:200]}
