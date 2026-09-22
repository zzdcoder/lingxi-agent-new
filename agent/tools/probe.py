"""
工具熔断恢复定时探测（设计文档 §7.7，APScheduler 驱动）

职责：每 agent_probe_interval_seconds 秒扫描 tool_registry 中 status=2（熔断）
的工具，用熔断时沉淀的失败入参（last_error_args）重放调用，验证服务是否恢复：

- 只读工具（risk_level='read'）且沉淀了可重放入参 → 自动探测；
- 探测成功 → breaker.auto_recover 恢复 status=1（生效）；
- 探测失败 / 写工具 / 无沉淀入参 / 入参被截断 → 保持熔断，等待下一轮或人工恢复。

设计要点：
1. 探测**只读工具**：写工具重放入参可能产生重复副作用（重复插入/删除），留人工恢复；
2. 探测不经过熔断器（它本身就是恢复手段），带独立超时保护；
3. 串行执行（熔断工具量小，避免探测请求叠加压垮服务）；
4. 探测失败更新 last_error（最近探测失败原因），便于运维观测。

用法（app lifespan 注册）：
    scheduler.add_job(scan_and_probe, trigger=IntervalTrigger(seconds=...))
"""
import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, Tuple

from langchain_core.tools import BaseTool
from sqlalchemy import select

from core.config import settings
from core.database import AsyncSessionLocal
from models.tool_model import ToolRegistry, ToolStatus
from agent.tools.circuit_breaker import breaker
from agent.tools.db_tools import DbToolExecutor
from agent.tools.knowledge_tool import KnowledgeToolExecutor
from agent.tools.manager import _resolve_handler
from agent.observability import (
    K_PROBE_PREFIX,
    bind_trace_id,
    get_trace_id,
    metrics,
)

logger = logging.getLogger(__name__)


def _excluded_tools() -> set:
    """
    禁止自动重放的工具集合。

    交互式工具（如 ask_user）体质特殊：其内部调用 langgraph `interrupt()`，
    在定时探测这种**无 interrupt 上下文**的场景下重放会直接抛错并污染
    last_error，导致工具**永远无法被自动恢复**。因此由配置显式排除，
    交由人工恢复（§7.7.3）。
    """
    return {
        t.strip()
        for t in (settings.agent_probe_exclude_tools or "").split(",")
        if t.strip()
    }


async def scan_and_probe() -> Dict[str, Any]:
    """扫描并探测熔断工具（APScheduler 定时入口）。

    :return: 本轮统计 {"scanned": N, "probed": N, "recovered": N, "failed": N, "skipped": [...]}
    """
    stats: Dict[str, Any] = {"scanned": 0, "probed": 0, "recovered": 0, "failed": 0, "skipped": []}
    if not settings.agent_probe_enabled:
        return stats

    trace_id = get_trace_id()
    if trace_id == "-":
        # 定时任务不在请求上下文中：自建 trace_id，保证每轮扫描可追踪
        trace_id = bind_trace_id()

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(ToolRegistry).where(
                ToolRegistry.status == ToolStatus.CIRCUIT_BREAK,
                ToolRegistry.risk_level == "read",
            )
        )
        rows = list(result.scalars().all())
        if not rows:
            return stats

        # 探测执行器：与任务节点同源（manager._resolve_handler 复用其方法解析）
        executor = DbToolExecutor(session)
        kb_executor = KnowledgeToolExecutor(session, username=None)
        excluded = _excluded_tools()

        for row in rows:
            stats["scanned"] += 1
            if row.name in excluded:
                stats["skipped"].append(row.name)
                continue
            probed, recovered = await _probe_one(session, executor, kb_executor, row)
            if not probed:
                stats["skipped"].append(row.name)
                continue
            stats["probed"] += 1
            metrics.incr(K_PROBE_PREFIX.format("calls"))
            if recovered:
                stats["recovered"] += 1
                metrics.incr(K_PROBE_PREFIX.format("recovered"))
            else:
                stats["failed"] += 1
                metrics.incr(K_PROBE_PREFIX.format("failed"))

    if stats["probed"]:
        logger.info(
            f"[熔断探测] 扫描={stats['scanned']} 探测={stats['probed']} "
            f"恢复={stats['recovered']} 仍熔断={stats['failed']} 跳过={stats['skipped']} "
            f"trace_id={trace_id}"
        )
    return stats


async def _probe_one(session, executor, kb_executor, row: ToolRegistry) -> Tuple[bool, bool]:
    """探测单个熔断工具：重放失败入参，成功则自动恢复。

    :return: (probed, recovered) —— probed 表示本轮是否发起了探测，
             recovered 表示探测成功并已恢复生效。
    """
    args = row.last_error_args
    if not args or not isinstance(args, dict) or args.get("_truncated"):
        # 无沉淀入参 / 入参超长被截断：无法安全重放，保持熔断留人工恢复
        return False, False

    handler = _resolve_handler(row, executor, kb_executor)
    if handler is None:
        logger.warning(f"[熔断探测] 工具 {row.name} 无法解析执行函数，跳过探测")
        return False, False

    try:
        await asyncio.wait_for(
            _invoke(handler, args), timeout=settings.agent_probe_timeout_seconds
        )
    except Exception as e:
        # 探测失败：保持熔断，记录最近探测失败原因（脱敏）供运维观测
        row.last_error = f"定时探测失败：{str(e).splitlines()[0][:160]}"
        row.last_call_at = datetime.now()
        await session.commit()
        logger.warning(f"[熔断探测] 工具 {row.name} 探测失败，保持熔断: {e}")
        return True, False

    # 探测成功 → 自动恢复生效
    recovered = await breaker.auto_recover(session, row.name)
    return True, recovered


async def _invoke(handler: Any, args: Dict[str, Any]) -> Any:
    """按 handler 类型执行探测调用（executor 方法直接 await；BaseTool 用 ainvoke）。"""
    if isinstance(handler, BaseTool):
        return await handler.ainvoke(args)
    return await handler(**args)
