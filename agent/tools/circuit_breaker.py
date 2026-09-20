"""
工具熔断器（企业级熔断治理，设计文档 §7.7）

目标：当某个工具持续系统级异常（连续失败 / 窗口失败率超阈值）时，**只熔断该工具本身**，
将其状态置为 2（熔断），使任务 Agent 不再绑定/快速失败该工具，从而：
- 减少影响面：不熔断整个 Agent 或整条链路，仅让"坏工具"退出服务；
- 快速失败（fail-fast）：熔断期间直接拒绝，不再等待超时拖垮请求；
- 自动恢复：由定时探测任务（APScheduler 驱动，agent/tools/probe.py）用熔断时
  沉淀的失败入参（last_error_args）重放验证，成功后自动回到生效状态。

熔断触发口径（设计文档 §7.7.2）：**仅系统级故障**（超时 / 连接异常 / 底层执行失败，
即 DbToolSystemError / asyncio.TimeoutError）计入熔断统计；参数/业务错误属模型可
纠正错误，只记录 last_error，不参与熔断，避免 LLM 参数生成错误误熔断核心工具。

熔断器状态机（circuit_state）：
    CLOSED（关闭/正常）
      │ 连续失败 ≥ failure_threshold，或窗口调用量 ≥ min_calls 且失败率 ≥ failure_ratio
      ▼
    OPEN（打开/熔断）→ 同步 tool_registry.status = 2（快速失败）
      │ 恢复仅由定时探测任务（probe.scan_and_probe）驱动：
      │ 只读工具用沉淀的 last_error_args 重放成功 → CLOSED（status=1）
      │ 失败/写工具/无沉淀入参 → 保持 OPEN，等待下一轮探测或人工恢复

性能模型（内存判定 + 批量落库）：
- 判定依据与计数在**进程内缓存行**上即时完成（熔断触发/探测恢复等状态迁移**实时落库**）；
- 纯计数变更（total/success/fail/window/latency 等）标记 dirty，由后台任务
  每 agent_circuit_flush_interval_seconds 批量 UPDATE 合并写回；
- 每次工具调用的 DB 开销从"1 SELECT + 1 UPDATE + 1 COMMIT"降为"接近 0（纯内存）"，
  仅状态迁移（低频）与后台冲刷（每工具每秒 ≤1 次）产生 DB 写；
- 后台冲刷任务生命周期由 app lifespan 管理（启动创建、退出前冲刷一次防计数丢失）。
"""
import asyncio
import json
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


def _int(value: Optional[int]) -> int:
    """归一化计数列（兼容 NULL/None，与列 default=0 语义一致）。

    真实 DB 行经 INSERT 默认值后计数列恒为非空；此处防御手动构造
    （如 mock/历史脏数据）或行字段未初始化导致的 None 运算异常。
    """
    return value or 0


class _ToolState:
    """进程内工具熔断状态（决策缓存 + 待落库变更的载体）。

    - row：内存态行（_load_from_db 浅拷贝自 DB，判定/计数以它为准）；
    - dirty：存在未落库的计数变更（后台 flush 合并写回后清除）；
    - fetched_at：回源时间戳，用于 TTL 过期判定。
    """

    __slots__ = ("row", "dirty", "fetched_at")

    def __init__(self, row: ToolRegistry):
        self.row = row
        self.dirty = False
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
        self._state: dict[str, _ToolState] = {}
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

        流程：决策判定（内存态/回源）→ 快速失败/放行 → 执行 → 记录成败 → 必要时触发熔断。

        :param db: 数据库会话（读取/写回熔断状态）
        :param tool_name: 工具名（tool_registry.name）
        :param handler: 实际执行函数（已含超时包装）
        :param kwargs: 工具的具名参数（系统级失败时脱敏沉淀，供定时探测重放）
        :return: handler 的返回结果
        :raises CircuitOpenError: 熔断期间快速失败
        :raises Exception: 原始业务异常（记录熔断指标后原样上抛）
        """
        if not settings.agent_circuit_enabled:
            return await handler(*args, **kwargs)

        async with self._lock:
            st = await self._get_state(tool_name, db)
            row = st.row

            # 熔断态（status=2，含 OPEN 与存量 HALF_OPEN 数据）→ 快速失败。
            # 恢复只由定时探测 / 人工触发，不再依赖冷却期被动半开（短任务下无流量可用）。
            if row.status == ToolStatus.CIRCUIT_BREAK:
                since = row.circuit_open_at
                detail = f"（自 {since:%Y-%m-%d %H:%M:%S} 起）" if since is not None else ""
                raise CircuitOpenError(
                    f"工具 {tool_name} 当前处于熔断状态{detail}，"
                    "暂时不可用。请尝试其他可用方式，或如实告知用户稍后重试。"
                )

        # ---- 执行阶段（锁外执行，避免长任务阻塞其他工具的判定） ----
        start = time.perf_counter()
        try:
            result = await handler(*args, **kwargs)
        except Exception as e:
            await self._record_failure(
                db, tool_name, e,
                int((time.perf_counter() - start) * 1000),
                error_args=kwargs,
            )
            raise
        else:
            await self._record_success(db, tool_name, int((time.perf_counter() - start) * 1000))
            return result

    # =========================================================================
    # 内存态管理
    # =========================================================================

    async def _get_state(self, tool_name: str, db) -> _ToolState:
        """获取内存态；未命中或（过期且无待落库变更）时回源查库重建。"""
        st = self._state.get(tool_name)
        if st is None or (st.expired() and not st.dirty):
            row = await self._load_from_db(db, tool_name)
            st = _ToolState(row)
            self._state[tool_name] = st
        return st

    @staticmethod
    async def _load_from_db(db, tool_name: str) -> ToolRegistry:
        """从 tool_registry 加载熔断状态（表缺失/工具未注册时按 CLOSED 处理）。

        返回按列值重建的独立瞬态行作为内存态：不共享 DB 会话的 identity map，
        也不持有会随会话关闭而失效的 InstanceState，可跨请求安全改值。
        """
        try:
            from sqlalchemy import select

            result = await db.execute(
                select(ToolRegistry).where(ToolRegistry.name == tool_name)
            )
            row = result.scalars().first()
            if row is not None:
                # 按列值重建独立瞬态对象：不共享会话原对象的 _sa_instance_state。
                # 浅拷贝（copy.copy）会共享 InstanceState，原对象被 GC 后改值即抛
                # ObjectDereferencedError；瞬态对象无会话绑定，可安全跨请求改值。
                return ToolRegistry(
                    **{c.key: getattr(row, c.key) for c in ToolRegistry.__table__.columns}
                )
        except Exception as e:
            logger.warning(f"[熔断器] 读取工具 {tool_name} 状态失败（按关闭处理）: {e}")
        return ToolRegistry(name=tool_name)

    # =========================================================================
    # 状态迁移
    # =========================================================================

    async def _record_success(self, db, tool_name: str, latency_ms: int) -> None:
        """记录一次成功调用；CLOSED 下复位连续失败（恢复仅由定时探测 / 人工触发）。"""
        async with self._lock:
            st = await self._get_state(tool_name, db)
            row = st.row
            row.total_calls = _int(row.total_calls) + 1
            row.success_calls = _int(row.success_calls) + 1
            row.last_call_at = _now()
            row.last_success_at = _now()
            row.avg_latency_ms = _smoothed_avg(_int(row.avg_latency_ms), latency_ms)
            self._touch_window(row, success=True)

            if row.circuit_state == ToolCircuitState.CLOSED:
                row.consecutive_failures = 0
            st.dirty = True

    async def _record_failure(
        self,
        db,
        tool_name: str,
        exc: Exception,
        latency_ms: int,
        error_args: Optional[dict] = None,
    ) -> None:
        """记录一次失败调用；仅系统级故障参与熔断统计，失败时沉淀脱敏入参。

        熔断触发口径（设计文档 §7.7.2）：参数/业务错误（DbToolError 等）属模型可
        纠正错误，只记录 last_error 不参与计数，避免误熔断；系统级故障
        （DbToolSystemError / asyncio.TimeoutError）正常计数并在达到阈值时触发熔断。
        """
        async with self._lock:
            st = await self._get_state(tool_name, db)
            row = st.row
            row.last_call_at = _now()
            row.last_error = _safe_error(exc)

            if not _is_system_failure(exc):
                st.dirty = True
                return

            row.total_calls = _int(row.total_calls) + 1
            row.fail_calls = _int(row.fail_calls) + 1
            row.avg_latency_ms = _smoothed_avg(_int(row.avg_latency_ms), latency_ms)
            if error_args:
                row.last_error_args = _sanitize_args(error_args)

            if row.circuit_state == ToolCircuitState.CLOSED:
                row.consecutive_failures = _int(row.consecutive_failures) + 1
                self._touch_window(row, success=False)
                if self._should_trip(row):
                    logger.warning(
                        f"[熔断器] 工具 {tool_name} 触发熔断："
                        f"连续失败={row.consecutive_failures}/{row.failure_threshold}, "
                        f"窗口失败率={_window_ratio(row):.0%}"
                    )
                    self._open_circuit(row, exc)
                    await self._write_back(db, tool_name, st)
                else:
                    st.dirty = True
            else:
                st.dirty = True  # 防御：熔断态下正常已被快速失败拦截

    # =========================================================================
    # 熔断判定与迁移辅助
    # =========================================================================

    def _should_trip(self, row: ToolRegistry) -> bool:
        """CLOSED 状态下判定是否触发熔断（连续失败阈值 或 窗口失败率阈值）。"""
        if _int(row.consecutive_failures) >= row.failure_threshold:
            return True
        if _int(row.window_calls) >= settings.agent_circuit_min_calls:
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
        row.window_calls = _int(row.window_calls) + 1
        if not success:
            row.window_failures = _int(row.window_failures) + 1

    @staticmethod
    def _open_circuit(row: ToolRegistry, exc: Exception) -> None:
        """打开熔断：circuit_state=OPEN + status=2（熔断）。

        circuit_open_until 仅作"预计恢复时间"展示（实际恢复由定时探测 / 人工触发，
        不再依赖冷却期到期自动进入半开）。
        """
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

    # =========================================================================
    # 落库：即时写回 / 批量冲刷
    # =========================================================================

    async def _write_back(self, db, tool_name: str, st: _ToolState) -> None:
        """内存态整行写回 DB（状态迁移即时落库 / 批量冲刷共用），成功后清除 dirty。

        失败时保留 dirty，由下一轮批量冲刷重试，避免状态/计数静默丢失。
        """
        row = st.row
        st.dirty = True
        try:
            from sqlalchemy import update

            await db.execute(
                update(ToolRegistry)
                .where(ToolRegistry.name == tool_name)
                .values(
                    status=row.status,
                    circuit_state=row.circuit_state,
                    consecutive_failures=_int(row.consecutive_failures),
                    total_calls=_int(row.total_calls),
                    success_calls=_int(row.success_calls),
                    fail_calls=_int(row.fail_calls),
                    window_calls=_int(row.window_calls),
                    window_failures=_int(row.window_failures),
                    window_start_at=row.window_start_at,
                    half_open_trials=_int(row.half_open_trials),
                    circuit_open_at=row.circuit_open_at,
                    circuit_open_until=row.circuit_open_until,
                    last_error=row.last_error,
                    last_error_args=row.last_error_args,
                    last_call_at=row.last_call_at,
                    last_success_at=row.last_success_at,
                    avg_latency_ms=_int(row.avg_latency_ms),
                    remark=row.remark,
                )
            )
            await db.commit()
            st.dirty = False
        except Exception as e:
            await db.rollback()
            logger.warning(f"[熔断器] 写回工具 {tool_name} 状态失败（保留 dirty 下轮重试）: {e}")

    async def flush(self, db) -> None:
        """批量落库所有 dirty 内存态（后台任务周期调用 / 退出前冲刷）。

        持锁串行执行，避免与人工运维（manual_transition 丢弃脏态）产生写回竞态；
        冲刷频率低（每 agent_circuit_flush_interval_seconds 一次），锁占用可忽略。
        """
        async with self._lock:
            for name in list(self._state):
                st = self._state.get(name)
                if st is not None and st.dirty:
                    await self._write_back(db, name, st)

    # =========================================================================
    # 定时探测恢复（agent/tools/probe.py 调用）
    # =========================================================================

    async def auto_recover(self, db, tool_name: str) -> bool:
        """定时探测成功后自动恢复工具（status=1，circuit_state=CLOSED，计数器清零）。

        仅对熔断态工具生效；探测失败 / 写工具 / 无沉淀入参的恢复决策由调用方
        （probe.scan_and_probe）控制，本方法只执行"验证成功后的复位"。

        :return: 是否执行了恢复（工具非熔断态时返回 False）
        """
        async with self._lock:
            st = await self._get_state(tool_name, db)
            row = st.row
            if row.status != ToolStatus.CIRCUIT_BREAK:
                return False

            self._close_circuit(row)
            row.remark = "定时探测成功，自动恢复"
            await self._write_back(db, tool_name, st)
            logger.info(f"[熔断器] 工具 {tool_name} 定时探测成功，自动恢复生效")
            return True

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

        操作后**丢弃**进程内待落库脏态（下次调用回源按新状态判定），
        避免未落库的旧计数覆盖人工复位的结果。

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
            self._state.pop(tool_name, None)  # 丢弃 pending 脏态，下次调用回源按新状态判定
            logger.info(
                f"[熔断器] 人工调整工具 {tool_name}: {target_status}（备注: {remark or '-'}）"
            )
            return row


async def flush_loop(session_factory) -> None:
    """后台批量落库循环（由 app lifespan 创建 task，退出时取消）。

    :param session_factory: 异步会话工厂（如 AsyncSessionLocal），每次循环新建会话。
    """
    while True:
        await asyncio.sleep(settings.agent_circuit_flush_interval_seconds)
        try:
            async with session_factory() as session:
                await breaker.flush(session)
        except Exception as e:
            logger.warning(f"[熔断器] 后台批量落库异常: {e}")


def _window_ratio(row: ToolRegistry) -> float:
    """窗口失败率。"""
    calls = _int(row.window_calls)
    if not calls:
        return 0.0
    return _int(row.window_failures) / calls


def _smoothed_avg(current: Optional[int], new_ms: int) -> int:
    """指数平滑平均耗时（EMA，alpha=0.3）。"""
    current = _int(current)
    if current <= 0:
        return new_ms
    return int(current * 0.7 + new_ms * 0.3)


def _safe_error(exc: Exception) -> str:
    """脱敏错误信息（不暴露 SQL/参数/堆栈，取首行前 160 字符）。"""
    detail = str(exc).strip()
    line = detail.splitlines()[0] if detail else type(exc).__name__
    return line[:160]


# 沉淀入参中需排除的敏感键（与 db_tools.SENSITIVE_COLUMNS 同口径，防凭据泄露）
_SENSITIVE_ARG_KEYS = {
    "password", "password_hash", "secret", "secret_key", "access_token",
    "refresh_token", "token", "api_key", "apikey", "private_key", "salt",
}
# 单个参数值的最大展示长度（探测重放只需"可调用"，无需完整语义，超长截断防大参数）
_MAX_ARG_VALUE_LENGTH = 256
# 沉淀入参 JSON 整体最大长度（超限则不沉淀，探测时跳过该工具留人工恢复）
_MAX_ARGS_DUMP_LENGTH = 4096


def _is_system_failure(exc: Exception) -> bool:
    """熔断触发口径：仅系统级故障（超时 / 连接异常 / 底层执行失败）计入熔断统计。

    参数/业务错误（DbToolError 等）属模型可纠正错误，不触发熔断（设计文档 §7.7.2），
    避免 LLM 参数生成错误误熔断核心工具。
    """
    from agent.tools.db_tools import DbToolSystemError

    return isinstance(exc, (DbToolSystemError, asyncio.TimeoutError))


def _truncate_value(value: Any, limit: int) -> Any:
    """递归截断参数值：字符串截断为限长；list/dict 逐项截断；标量原样保留。"""
    if isinstance(value, dict):
        return {str(k): _truncate_value(v, limit) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_truncate_value(v, limit) for v in value]
    if isinstance(value, str):
        return value[:limit] if len(value) > limit else value
    return value


def _sanitize_args(args: Optional[dict]) -> Optional[dict]:
    """脱敏沉淀失败时的工具入参（供定时探测重放验证恢复）。

    规则：
    1. 排除敏感键（token/密码等，安全底线）；
    2. 递归截断超长值（防大参数/大文本撑爆 JSON 列）；
    3. 整体超限返回 {"_truncated": True, ...}，探测时跳过该工具（留人工恢复）。

    返回值为可安全 JSON 序列化的 dict；args 为空返回 None（不沉淀）。
    """
    if not args:
        return None
    clean: dict[str, Any] = {}
    for key, value in args.items():
        if str(key).lower() in _SENSITIVE_ARG_KEYS:
            continue
        clean[str(key)] = _truncate_value(value, _MAX_ARG_VALUE_LENGTH)
    dumped = json.dumps(clean, ensure_ascii=False, default=str)
    if len(dumped) > _MAX_ARGS_DUMP_LENGTH:
        return {"_truncated": True, "keys": sorted(clean.keys())}
    return clean


# 进程级熔断器单例（跨请求共享决策缓存）
breaker = ToolCircuitBreaker()
