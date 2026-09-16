"""
工具熔断器（企业级熔断治理，设计文档 §7.7）

目标：当某个工具持续异常（连续失败 / 窗口失败率超阈值）时，**只熔断该工具本身**，
将其状态置为 2（熔断），使任务 Agent 不再绑定/快速失败该工具，从而：
- 减少影响面：不熔断整个 Agent 或整条链路，仅让"坏工具"退出服务；
- 快速失败（fail-fast）：OPEN 期间直接拒绝，不再等待超时拖垮请求；
- 自动恢复：冷却期后进入半开探测，验证工具恢复后自动回到生效状态。

熔断器状态机（circuit_state）：
    CLOSED（关闭/正常）
      │ 连续失败 ≥ failure_threshold，或窗口调用量 ≥ min_calls 且失败率 ≥ failure_ratio
      ▼
    OPEN（打开/熔断）→ 同步 tool_registry.status = 2
      │ 冷却期（cooldown_seconds）到期
      ▼
    HALF_OPEN（半开/探测）
      │ 放行 ≤ half_open_max_trials 个探测请求
      ├─ 任一探测成功 → CLOSED（status=1，计数器清零）
      └─ 任一探测失败 → OPEN（status=2，重新计时冷却期）

说明：
- 判定依据与计数持久化在 tool_registry 表（跨请求、跨重启可恢复）；
- 进程内维护短 TTL 决策缓存，避免每次调用都查库；
- 熔断只统计业务异常（ToolException 子类等），配置可开关（agent_circuit_enabled）。
"""
import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, Optional

from langchain_core.tools import ToolException

from core.config import settings
from models.tool_model import ToolCircuitState, ToolRegistry, ToolStatus

logger = logging.getLogger(__name__)

# 熔断决策缓存 TTL（秒）：OPEN 快速失败判定可容忍的短暂延迟
_CACHE_TTL_SECONDS = 1.0


class CircuitOpenError(ToolException):
    """
    熔断拒绝异常（ToolException 子类）。

    抛给工具框架后经 handle_tool_error 转为 ToolMessage 送回 Agent 循环，
    模型收到"工具暂时不可用"后会尝试其他工具或如实告知用户，而非中断整个任务。
    """


def _now() -> datetime:
    return datetime.now()


class _BreakerDecision:
    """进程内熔断决策缓存（避免每次调用查库）"""

    __slots__ = ("circuit_state", "circuit_open_until", "half_open_trials", "fetched_at")

    def __init__(self, circuit_state: str, circuit_open_until: Optional[datetime],
                 half_open_trials: int):
        self.circuit_state = circuit_state
        self.circuit_open_until = circuit_open_until
        self.half_open_trials = half_open_trials
        self.fetched_at = time.monotonic()

    def expired(self) -> bool:
        return time.monotonic() - self.fetched_at > _CACHE_TTL_SECONDS


class ToolCircuitBreaker:
    """
    工具熔断器（进程级单例，跨请求共享状态决策缓存）。

    用法（在工具绑定/调用处包裹 handler）：
        result = await breaker.call(db, tool_name, handler, *args, **kwargs)
    """

    def __init__(self):
        self._cache: dict[str, _BreakerDecision] = {}
        self._lock = asyncio.Lock()  # 状态迁移与计数变更的进程内互斥

    # =========================================================================
    # 对外主入口
    # =========================================================================

    async def call(
        self,
        db,
        tool_name: str,
        handler: Callable[..., Awaitable[Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        熔断保护的工具调用。

        流程：决策判定（缓存/查库）→ 允许/快速失败 → 执行 → 记录成败 → 必要时触发熔断。

        :param db: 数据库会话（用于读取/持久化熔断状态）
        :param tool_name: 工具名（tool_registry.name）
        :param handler: 实际执行函数（已含超时包装）
        :return: handler 的返回结果
        :raises CircuitOpenError: 熔断 OPEN 期间快速失败
        :raises Exception: 原始业务异常（记录熔断指标后原样上抛）
        """
        if not settings.agent_circuit_enabled:
            return await handler(*args, **kwargs)

        async with self._lock:
            decision = await self._get_decision(tool_name, db)

            # OPEN 且未到期 → 快速失败
            if (
                decision.circuit_state == ToolCircuitState.OPEN
                and decision.circuit_open_until is not None
                and _now() < decision.circuit_open_until
            ):
                raise CircuitOpenError(
                    f"工具 {tool_name} 当前处于熔断状态（熔断至 {decision.circuit_open_until:%Y-%m-%d %H:%M:%S}），"
                    "暂时不可用。请尝试其他可用方式，或如实告知用户稍后重试。"
                )

            # 冷却期到期 → 从 OPEN 进入 HALF_OPEN 探测
            if decision.circuit_state == ToolCircuitState.OPEN:
                await self._enter_half_open(db, tool_name)
                decision = _BreakerDecision(
                    ToolCircuitState.HALF_OPEN, None, 0
                )

            # HALF_OPEN：限制并发探测数量
            if decision.circuit_state == ToolCircuitState.HALF_OPEN:
                if decision.half_open_trials >= settings.agent_circuit_half_open_max_trials:
                    raise CircuitOpenError(
                        f"工具 {tool_name} 正处于熔断恢复探测期，暂不放行新请求，请稍后重试。"
                    )
                self._cache[tool_name] = _BreakerDecision(
                    ToolCircuitState.HALF_OPEN, None, decision.half_open_trials + 1
                )

        # ---- 执行阶段（锁外执行，避免长任务阻塞其他工具的判定） ----
        start = time.perf_counter()
        try:
            result = await handler(*args, **kwargs)
        except Exception as e:
            await self._record_failure(db, tool_name, e, int((time.perf_counter() - start) * 1000))
            raise
        else:
            await self._record_success(db, tool_name, int((time.perf_counter() - start) * 1000))
            return result

    # =========================================================================
    # 决策缓存
    # =========================================================================

    async def _get_decision(self, tool_name: str, db) -> _BreakerDecision:
        """获取决策缓存；未命中或过期时回源查库。"""
        decision = self._cache.get(tool_name)
        if decision is None or decision.expired():
            decision = await self._load_from_db(db, tool_name)
            self._cache[tool_name] = decision
        return decision

    @staticmethod
    async def _load_from_db(db, tool_name: str) -> _BreakerDecision:
        """从 tool_registry 加载熔断状态（表缺失/工具未注册时按 CLOSED 处理）。"""
        try:
            from sqlalchemy import select

            result = await db.execute(
                select(ToolRegistry).where(ToolRegistry.name == tool_name)
            )
            row = result.scalars().first()
            if row is not None:
                return _BreakerDecision(
                    circuit_state=row.circuit_state,
                    circuit_open_until=row.circuit_open_until,
                    half_open_trials=row.half_open_trials,
                )
        except Exception as e:
            logger.warning(f"[熔断器] 读取工具 {tool_name} 状态失败（按关闭处理）: {e}")
        return _BreakerDecision(ToolCircuitState.CLOSED, None, 0)

    # =========================================================================
    # 状态迁移
    # =========================================================================

    async def _enter_half_open(self, db, tool_name: str) -> None:
        """OPEN → HALF_OPEN：冷却期到期，复位探测计数，开始验证恢复。"""
        logger.warning(f"[熔断器] 工具 {tool_name} 冷却期结束，进入半开探测")
        await self._update_row(
            db, tool_name,
            circuit_state=ToolCircuitState.HALF_OPEN,
            half_open_trials=0,
            circuit_open_until=None,
        )
        self._cache[tool_name] = _BreakerDecision(ToolCircuitState.HALF_OPEN, None, 0)

    async def _record_success(self, db, tool_name: str, latency_ms: int) -> None:
        """记录一次成功调用；HALF_OPEN 下成功即关闭熔断，CLOSED 下复位连续失败。"""
        async with self._lock:
            try:
                from sqlalchemy import select

                result = await db.execute(select(ToolRegistry).where(ToolRegistry.name == tool_name))
                row = result.scalars().first()
                if row is None:
                    return
                row.total_calls += 1
                row.success_calls += 1
                row.last_call_at = _now()
                row.last_success_at = _now()
                row.avg_latency_ms = _smoothed_avg(row.avg_latency_ms, latency_ms)

                if row.circuit_state == ToolCircuitState.HALF_OPEN:
                    # 半开探测成功 → 关闭熔断，恢复生效
                    logger.info(f"[熔断器] 工具 {tool_name} 半开探测成功，恢复生效")
                    self._close_circuit(row)
                elif row.circuit_state == ToolCircuitState.CLOSED:
                    row.consecutive_failures = 0

                self._touch_window(row, success=True)
                await db.commit()
                self._cache[tool_name] = _BreakerDecision(
                    row.circuit_state, row.circuit_open_until, row.half_open_trials
                )
            except Exception as e:
                await db.rollback()
                logger.debug(f"[熔断器] 记录成功指标失败（非致命）: {e}")

    async def _record_failure(self, db, tool_name: str, exc: Exception, latency_ms: int) -> None:
        """记录一次失败调用；达到阈值触发熔断（status=2）。"""
        async with self._lock:
            try:
                from sqlalchemy import select

                result = await db.execute(select(ToolRegistry).where(ToolRegistry.name == tool_name))
                row = result.scalars().first()
                if row is None:
                    return
                row.total_calls += 1
                row.fail_calls += 1
                row.last_call_at = _now()
                row.avg_latency_ms = _smoothed_avg(row.avg_latency_ms, latency_ms)
                row.last_error = _safe_error(exc)

                if row.circuit_state == ToolCircuitState.HALF_OPEN:
                    # 半开探测失败 → 重新熔断
                    logger.warning(f"[熔断器] 工具 {tool_name} 半开探测失败，重新熔断")
                    self._open_circuit(db, row, exc)
                elif row.circuit_state == ToolCircuitState.CLOSED:
                    row.consecutive_failures += 1
                    self._touch_window(row, success=False)
                    if self._should_trip(row):
                        logger.warning(
                            f"[熔断器] 工具 {tool_name} 触发熔断："
                            f"连续失败={row.consecutive_failures}/{row.failure_threshold}, "
                            f"窗口失败率={_window_ratio(row):.0%}"
                        )
                        self._open_circuit(db, row, exc)
                await db.commit()
                self._cache[tool_name] = _BreakerDecision(
                    row.circuit_state, row.circuit_open_until, row.half_open_trials
                )
            except Exception as e:
                await db.rollback()
                logger.debug(f"[熔断器] 记录失败指标异常（非致命）: {e}")

    # =========================================================================
    # 熔断判定与迁移辅助
    # =========================================================================

    def _should_trip(self, row: ToolRegistry) -> bool:
        """CLOSED 状态下判定是否触发熔断（连续失败阈值 或 窗口失败率阈值）。"""
        if row.consecutive_failures >= row.failure_threshold:
            return True
        if row.window_calls >= settings.agent_circuit_min_calls:
            return _window_ratio(row) >= (row.failure_ratio or settings.agent_circuit_failure_ratio)
        return False

    @staticmethod
    def _touch_window(row: ToolRegistry, success: bool) -> None:
        """维护时间窗口计数（窗口过期则重置）。"""
        now = _now()
        if (
            row.window_start_at is None
            or (now - row.window_start_at).total_seconds() >= (row.window_seconds or 60)
        ):
            row.window_start_at = now
            row.window_calls = 0
            row.window_failures = 0
        row.window_calls += 1
        if not success:
            row.window_failures += 1

    @staticmethod
    def _open_circuit(db, row: ToolRegistry, exc: Exception) -> None:
        """打开熔断：circuit_state=OPEN + status=2（熔断）+ 冷却期计时。"""
        now = _now()
        cooldown = row.cooldown_seconds or settings.agent_circuit_cooldown_seconds
        row.circuit_state = ToolCircuitState.OPEN
        row.status = ToolStatus.CIRCUIT_BREAK
        row.circuit_open_at = now
        row.circuit_open_until = now + timedelta(seconds=cooldown)
        row.half_open_trials = 0
        row.remark = f"自动熔断：{_safe_error(exc)}"

    @staticmethod
    def _close_circuit(row: ToolRegistry) -> None:
        """关闭熔断：circuit_state=CLOSED + status=1（生效）+ 计数器清零。"""
        row.circuit_state = ToolCircuitState.CLOSED
        row.status = ToolStatus.ACTIVE
        row.consecutive_failures = 0
        row.half_open_trials = 0
        row.window_calls = 0
        row.window_failures = 0
        row.circuit_open_at = None
        row.circuit_open_until = None
        row.remark = None

    @staticmethod
    async def _update_row(db, tool_name: str, **fields: Any) -> None:
        """按工具名更新一行（状态迁移用）。"""
        try:
            from sqlalchemy import select

            result = await db.execute(select(ToolRegistry).where(ToolRegistry.name == tool_name))
            row = result.scalars().first()
            if row is None:
                return
            for key, value in fields.items():
                setattr(row, key, value)
            await db.commit()
        except Exception as e:
            await db.rollback()
            logger.warning(f"[熔断器] 更新工具 {tool_name} 状态失败: {e}")


    # =========================================================================
    # 人工状态流转（工具管理 API 调用）
    # =========================================================================

    async def manual_transition(
        self,
        db,
        tool_name: str,
        target_status: int,
        remark: Optional[str] = None,
    ) -> Optional[ToolRegistry]:
        """
        人工调整工具状态（运维操作，设计文档 §7.7.3）。

        合法流转（其余组合抛出 ValueError）：
        - 1(生效)→0(失效)：人工下线；
        - 0(失效)→1(生效)：人工恢复；
        - 1(生效)→2(熔断)：人工熔断（立即打开熔断器，不再依赖失败统计）；
        - 2(熔断)→1(生效)：人工恢复（关闭熔断器，计数器清零）；
        - 2(熔断)→0(失效)：熔断后直接下线。

        操作后同步更新进程内决策缓存，保证后续调用立即按新状态判定。

        :return: 更新后的工具记录（工具不存在返回 None）
        :raises ValueError: 非法状态流转
        """
        async with self._lock:
            from sqlalchemy import select

            result = await db.execute(select(ToolRegistry).where(ToolRegistry.name == tool_name))
            row = result.scalars().first()
            if row is None:
                return None

            now = _now()
            cooldown = row.cooldown_seconds or settings.agent_circuit_cooldown_seconds
            pairs = {(row.status, target_status)}
            if pairs not in (
                {(ToolStatus.ACTIVE, ToolStatus.DISABLED)},
                {(ToolStatus.DISABLED, ToolStatus.ACTIVE)},
                {(ToolStatus.ACTIVE, ToolStatus.CIRCUIT_BREAK)},
                {(ToolStatus.CIRCUIT_BREAK, ToolStatus.ACTIVE)},
                {(ToolStatus.CIRCUIT_BREAK, ToolStatus.DISABLED)},
            ):
                raise ValueError(
                    f"非法状态流转: {row.status} → {target_status}"
                    "（合法：1↔0、1→2、2→1、2→0）"
                )

            row.status = target_status
            if remark is not None:
                row.remark = remark
            if target_status == ToolStatus.CIRCUIT_BREAK:
                # 人工熔断：立即打开熔断器并计时冷却期
                row.circuit_state = ToolCircuitState.OPEN
                row.circuit_open_at = now
                row.circuit_open_until = now + timedelta(seconds=cooldown)
                row.half_open_trials = 0
            elif target_status == ToolStatus.ACTIVE:
                # 人工恢复：关闭熔断器，计数器清零
                row.circuit_state = ToolCircuitState.CLOSED
                row.consecutive_failures = 0
                row.half_open_trials = 0
                row.window_calls = 0
                row.window_failures = 0
                row.circuit_open_at = None
                row.circuit_open_until = None
                if remark is None:
                    row.remark = None

            await db.commit()
            self._cache.pop(tool_name, None)  # 失效缓存，下次调用按新状态判定
            logger.info(
                f"[熔断器] 人工调整工具 {tool_name}: {target_status}（备注: {remark or '-'}）"
            )
            return row


def _window_ratio(row: ToolRegistry) -> float:
    """窗口失败率。"""
    if not row.window_calls:
        return 0.0
    return row.window_failures / row.window_calls


def _smoothed_avg(current: int, new_ms: int) -> int:
    """指数平滑平均耗时（EMA，alpha=0.3）。"""
    if current <= 0:
        return new_ms
    return int(current * 0.7 + new_ms * 0.3)


def _safe_error(exc: Exception) -> str:
    """脱敏错误信息（不暴露 SQL/参数/堆栈，取首行前 160 字符）。"""
    detail = str(exc).strip()
    line = detail.splitlines()[0] if detail else type(exc).__name__
    return line[:160]


# 进程级熔断器单例（跨请求共享决策缓存）
breaker = ToolCircuitBreaker()
