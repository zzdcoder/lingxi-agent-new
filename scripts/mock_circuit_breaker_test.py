"""
工具熔断器状态机 mock 测试脚本（设计文档 §7.7）

验证目标：模拟工具连续失败触发熔断，走完 CLOSED → OPEN → HALF_OPEN 完整状态机，
并验证 OPEN 快速失败、半开探测成功自动恢复（CLOSED）、半开探测失败重新熔断（OPEN）。

原理（不依赖真实数据库/网络）：
- MockAsyncDB：内存持有 {工具名: ToolRegistry}，仅实现熔断器用到的
  select(ToolRegistry).where(name == ?)（按绑定参数解析工具名）与 commit/rollback；
- MockToolHandler：可切换"失败/成功"的模拟工具执行器；
- 调整 settings 与工具行内的熔断参数，让测试秒级跑完（阈值 3、冷却期 1s）。

运行方式：
    python scripts/mock_circuit_breaker_test.py

退出码：0=全部通过，非 0=存在断言失败。
"""
import asyncio
import sys
from datetime import datetime, timedelta

from core.config import settings
from models.tool_model import ToolCircuitState, ToolRegistry, ToolStatus
from agent.tools.circuit_breaker import CircuitOpenError, breaker

# ---------------------------------------------------------------------------
# 测试配置：缩短阈值 / 冷却期，加速状态机流转
# ---------------------------------------------------------------------------
settings.agent_circuit_enabled = True          # 熔断器总开关
settings.agent_circuit_min_calls = 10          # 窗口失败率判定最小调用量（本测试走连续失败路径）
settings.agent_circuit_failure_ratio = 0.5     # 窗口失败率阈值
settings.agent_circuit_cooldown_seconds = 1    # 冷却期 1s（行级快照会被覆盖）
settings.agent_circuit_half_open_max_trials = 3  # 半开最大放行次数

TOOL_NAME = "mock_tool"

# ---------------------------------------------------------------------------
# 最小化 mock：异步 DB 会话
# ---------------------------------------------------------------------------


class _MockScalars:
    def __init__(self, rows):
        self._rows = rows

    def first(self):
        return self._rows[0] if self._rows else None

    def all(self):
        return list(self._rows)


class _MockResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return _MockScalars(self._rows)


class MockAsyncDB:
    """
    模拟 AsyncSession：
    - 内存持有 {工具名: ToolRegistry}；
    - execute 从 select 语句的绑定参数中解析工具名，返回对应行（与熔断器真实查询语义一致）；
    - commit/rollback 为 no-op（变更直接作用在内存行对象上）。
    """

    def __init__(self, rows_by_name):
        self.rows = dict(rows_by_name)

    async def execute(self, stmt):
        params = stmt.compile().params
        if not params:  # 全量查询（本测试未用到）
            return _MockResult(list(self.rows.values()))
        name = next(iter(params.values()))
        row = self.rows.get(name)
        return _MockResult([row] if row is not None else [])

    async def commit(self):
        pass

    async def rollback(self):
        pass


class MockToolHandler:
    """模拟工具执行器：可切换失败/成功；执行时把当前表的熔断状态记录下来（观测半开）。"""

    def __init__(self, db: MockAsyncDB):
        self.db = db
        self.fail = True          # True=模拟执行失败
        self.calls = 0
        self.observed_states = []  # 每次执行时观测到的 circuit_state（用于证明 HALF_OPEN）

    async def __call__(self, **kwargs):
        self.calls += 1
        # 执行瞬间观测 DB 中的熔断状态（此时 _enter_half_open 已提交）
        self.observed_states.append(self.db.rows[TOOL_NAME].circuit_state)
        if self.fail:
            raise RuntimeError("模拟工具执行失败")
        return {"ok": True, "echo": kwargs}


def seed_tool(failure_threshold: int = 3) -> ToolRegistry:
    """构造一条 CLOSED 状态的工具记录。"""
    return ToolRegistry(
        name=TOOL_NAME,
        description="mock 测试工具",
        category="custom",
        risk_level="read",
        status=ToolStatus.ACTIVE,
        circuit_state=ToolCircuitState.CLOSED,
        failure_threshold=failure_threshold,
        failure_ratio=settings.agent_circuit_failure_ratio,
        window_seconds=60,
        cooldown_seconds=settings.agent_circuit_cooldown_seconds,
        half_open_max_trials=settings.agent_circuit_half_open_max_trials,
    )


def snapshot(row: ToolRegistry) -> dict:
    """打印/断言用的状态快照。"""
    return {
        "status": row.status,
        "circuit_state": row.circuit_state,
        "consecutive_failures": row.consecutive_failures,
        "fail_calls": row.fail_calls,
        "success_calls": row.success_calls,
    }


def invalidate_cache():
    """熔断器决策缓存有 1s TTL；手动改库后需失效缓存强制回源。"""
    breaker._cache.pop(TOOL_NAME, None)


FAILURES: list[str] = []


def check(cond: bool, msg: str):
    tag = "PASS" if cond else "FAIL"
    print(f"  [{tag}] {msg}")
    if not cond:
        FAILURES.append(msg)


async def main():
    print("=" * 60)
    print("工具熔断器状态机 mock 测试（CLOSED → OPEN → HALF_OPEN）")
    print("=" * 60)

    db = MockAsyncDB({TOOL_NAME: seed_tool()})
    row = db.rows[TOOL_NAME]
    handler = MockToolHandler(db)

    # ---------------------------------------------------------------
    # 阶段一：CLOSED → OPEN（连续失败触发熔断，status=1 → 2）
    # ---------------------------------------------------------------
    print("\n[阶段一] CLOSED 下连续失败触发熔断")
    check(row.status == ToolStatus.ACTIVE and row.circuit_state == ToolCircuitState.CLOSED,
          f"初始状态: status=1 生效, CLOSED（实际 {snapshot(row)}）")

    for i in range(1, 4):  # failure_threshold=3，连败 3 次
        try:
            await breaker.call(db, TOOL_NAME, handler, query="fail")
        except RuntimeError:
            pass
        print(f"  第 {i} 次失败 → {snapshot(row)}")

    check(row.circuit_state == ToolCircuitState.OPEN, "连败 3 次后进入 OPEN")
    check(row.status == ToolStatus.CIRCUIT_BREAK, "status 已置为 2（熔断）")
    check(row.circuit_open_until is not None and row.circuit_open_until > datetime.now(),
          "已记录熔断到期时间（冷却期内快速失败）")

    # ---------------------------------------------------------------
    # 阶段二：OPEN 期间快速失败（fail-fast）
    # ---------------------------------------------------------------
    print("\n[阶段二] OPEN 期间快速失败")
    fail_calls_before = row.fail_calls
    try:
        await breaker.call(db, TOOL_NAME, handler, query="fail")
        check(False, "OPEN 期间调用应抛 CircuitOpenError，但未抛出")
    except CircuitOpenError as e:
        check(True, f"快速失败抛 CircuitOpenError（消息节选: {str(e)[:40]}...）")
    except Exception as e:
        check(False, f"期望 CircuitOpenError，实际抛 {type(e).__name__}")
    check(row.fail_calls == fail_calls_before, "快速失败不重复计数（fail_calls 不变）")

    # ---------------------------------------------------------------
    # 阶段三：冷却期到期 → HALF_OPEN（探测成功 → 自动恢复 CLOSED）
    # ---------------------------------------------------------------
    print("\n[阶段三] 冷却期到期进入 HALF_OPEN，探测成功自动恢复")
    # 手动把熔断到期时间拨回过去，模拟冷却期结束
    row.circuit_open_until = datetime.now() - timedelta(seconds=1)
    invalidate_cache()
    handler.fail = False          # 工具恢复：探测请求成功

    result = await breaker.call(db, TOOL_NAME, handler, query="recover")
    print(f"  探测调用执行时观测到的状态: {handler.observed_states[-1]}")
    check(handler.observed_states[-1] == ToolCircuitState.HALF_OPEN,
          "探测请求确实在 HALF_OPEN 状态下执行（状态机真实经过半开）")
    check(result == {"ok": True, "echo": {"query": "recover"}}, "探测请求执行成功并返回结果")
    check(row.circuit_state == ToolCircuitState.CLOSED, "半开探测成功 → 回到 CLOSED")
    check(row.status == ToolStatus.ACTIVE, "status 已恢复为 1（生效）")
    check(row.consecutive_failures == 0 and row.fail_calls == 3,
          "计数器复位（连续失败清零，累计失败保留）")
    print(f"  恢复后状态: {snapshot(row)}")

    # 恢复后应正常放行
    await breaker.call(db, TOOL_NAME, handler, query="ok")
    check(row.success_calls >= 2, "恢复后工具调用正常放行")

    # ---------------------------------------------------------------
    # 阶段四：再次熔断，HALF_OPEN 探测失败 → 重新 OPEN
    # ---------------------------------------------------------------
    print("\n[阶段四] 再次触发熔断，半开探测失败 → 重新 OPEN")
    handler.fail = True
    handler.observed_states.clear()
    for i in range(1, 4):
        try:
            await breaker.call(db, TOOL_NAME, handler, query="fail")
        except RuntimeError:
            pass
    check(row.circuit_state == ToolCircuitState.OPEN and row.status == ToolStatus.CIRCUIT_BREAK,
          "再次连败 3 次进入 OPEN（status=2）")

    row.circuit_open_until = datetime.now() - timedelta(seconds=1)
    invalidate_cache()
    try:
        await breaker.call(db, TOOL_NAME, handler, query="fail")
    except RuntimeError:
        pass
    print(f"  探测调用执行时观测到的状态: {handler.observed_states[-1]}")
    check(handler.observed_states[-1] == ToolCircuitState.HALF_OPEN,
          "探测请求在 HALF_OPEN 状态下执行")
    check(row.circuit_state == ToolCircuitState.OPEN, "半开探测失败 → 重新 OPEN")
    check(row.status == ToolStatus.CIRCUIT_BREAK, "status 保持 2（熔断）")
    check(row.circuit_open_until > datetime.now(), "冷却期重新计时")
    print(f"  最终状态: {snapshot(row)}")

    # ---------------------------------------------------------------
    # 汇总
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    if FAILURES:
        print(f"测试未通过，共 {len(FAILURES)} 处断言失败：")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("全部断言通过：CLOSED → OPEN → HALF_OPEN 完整状态机流转验证成功")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
