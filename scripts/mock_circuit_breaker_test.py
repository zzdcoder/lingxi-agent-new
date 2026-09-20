"""
工具熔断器状态机 mock 测试脚本（设计文档 §7.7）

验证目标（改造后状态机 CLOSED → OPEN →(定时探测)→ CLOSED）：
1. **熔断口径**：参数/业务错误（DbToolError）不参与熔断统计；仅系统级故障
   （DbToolSystemError / asyncio.TimeoutError）连续失败触发熔断；
2. OPEN 快速失败（fail-fast）；
3. 失败入参脱敏沉淀（last_error_args：敏感键排除、值截断、整体限长）；
4. 定时探测恢复：auto_recover 成功恢复（CLOSED/status=1/计数清零）；
5. auto_recover 对非熔断态工具不误操作（返回 False）。

原理（不依赖真实数据库/网络）：
- MockAsyncDB：内存持有 {工具名: ToolRegistry}，仅实现熔断器用到的
  select(ToolRegistry).where(name == ?) 与 UPDATE 写回、commit/rollback；
- MockToolHandler：可切换失败异常类型/成功状态的模拟工具执行器；
- 调整 settings 与工具行内的熔断参数，让测试秒级跑完（阈值 3）。

运行方式：
    python scripts/mock_circuit_breaker_test.py

退出码：0=全部通过，非 0=存在断言失败。
"""
import asyncio
import sys

from core.config import settings
from models.tool_model import ToolCircuitState, ToolRegistry, ToolStatus
from agent.tools.circuit_breaker import CircuitOpenError, _sanitize_args, breaker
from agent.tools.db_tools import DbToolError, DbToolSystemError

# ---------------------------------------------------------------------------
# 测试配置：缩短阈值，加速状态机流转
# ---------------------------------------------------------------------------
settings.agent_circuit_enabled = True          # 熔断器总开关
settings.agent_circuit_min_calls = 10          # 窗口失败率判定最小调用量（本测试走连续失败路径）
settings.agent_circuit_failure_ratio = 0.5     # 窗口失败率阈值

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
    - execute 支持熔断器的 UPDATE 写回（_write_back）：where 参数键为 "name"，values 键为字段名；
    - commit/rollback 为 no-op（变更直接作用在内存行对象上）。
    """

    def __init__(self, rows_by_name):
        self.rows = dict(rows_by_name)

    async def execute(self, stmt):
        from sqlalchemy.sql.dml import Update

        if isinstance(stmt, Update):
            params = stmt.compile().params
            # where 参数键为 name 列（SQLAlchemy 编译时可能加后缀，如 name_1），values 键为字段名
            name_key = next((k for k in params if k == "name" or k.startswith("name_")), None)
            row = self.rows.get(params.get(name_key)) if name_key else None
            if row is not None:
                for key, value in params.items():
                    if key != name_key and hasattr(row, key):
                        setattr(row, key, value)
            return _MockResult([])
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
    """模拟工具执行器：可切换异常类型/成功；把每次执行入参记录下来（观测沉淀）。"""

    def __init__(self, db: MockAsyncDB):
        self.db = db
        self.fail_exc: type[Exception] | None = DbToolSystemError  # None=成功
        self.calls = 0
        self.last_kwargs = None

    async def __call__(self, **kwargs):
        self.calls += 1
        self.last_kwargs = dict(kwargs)
        if self.fail_exc is not None:
            raise self.fail_exc(f"模拟{self.fail_exc.__name__}")
        return {"ok": True, "echo": kwargs}


def seed_tool(failure_threshold: int = 3) -> ToolRegistry:
    """构造一条 CLOSED 状态的工具记录（只读工具，符合定时探测条件）。"""
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
        cooldown_seconds=1,
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
    breaker._state.pop(TOOL_NAME, None)


FAILURES: list[str] = []


def check(cond: bool, msg: str):
    tag = "PASS" if cond else "FAIL"
    print(f"  [{tag}] {msg}")
    if not cond:
        FAILURES.append(msg)


async def main():
    print("=" * 60)
    print("工具熔断器 mock 测试（熔断口径 + 入参沉淀 + 定时探测恢复）")
    print("=" * 60)

    db = MockAsyncDB({TOOL_NAME: seed_tool()})
    row = db.rows[TOOL_NAME]
    handler = MockToolHandler(db)

    # ---------------------------------------------------------------
    # 阶段一：参数/业务错误不熔断（熔断触发口径）
    # ---------------------------------------------------------------
    print("\n[阶段一] 参数/业务错误（DbToolError）不触发熔断")
    check(row.status == ToolStatus.ACTIVE and row.circuit_state == ToolCircuitState.CLOSED,
          f"初始状态: status=1 生效, CLOSED（实际 {snapshot(row)}）")

    handler.fail_exc = DbToolError
    for i in range(1, 5):  # 连续 4 次参数错误（> 阈值 3）
        try:
            await breaker.call(db, TOOL_NAME, handler, query="bad-param")
        except DbToolError:
            pass
        await breaker.flush(db)
    check(row.circuit_state == ToolCircuitState.CLOSED,
          f"连败 4 次参数错误仍未熔断（consecutive_failures={row.consecutive_failures}, 期望保持 0）")
    check(row.consecutive_failures == 0, "参数错误不累计连续失败")
    check(row.fail_calls == 0, "参数错误不累计失败计数")
    check(row.last_error is not None, "参数错误仍记录 last_error（供模型修正参数）")
    check(row.last_error_args is None, "参数错误不沉淀探测入参")

    # ---------------------------------------------------------------
    # 阶段二：系统级故障连续失败触发熔断（CLOSED → OPEN）
    # ---------------------------------------------------------------
    print("\n[阶段二] 系统级故障（DbToolSystemError）连续失败触发熔断")
    handler.fail_exc = DbToolSystemError
    for i in range(1, 4):  # failure_threshold=3，连败 3 次
        try:
            await breaker.call(db, TOOL_NAME, handler, query="fail", table="t1")
        except DbToolSystemError:
            pass
        await breaker.flush(db)
        print(f"  第 {i} 次失败 → {snapshot(row)}")

    check(row.circuit_state == ToolCircuitState.OPEN, "连败 3 次后进入 OPEN")
    check(row.status == ToolStatus.CIRCUIT_BREAK, "status 已置为 2（熔断）")
    check(row.consecutive_failures == 3, "连续失败计数为 3")
    check(row.last_error_args is not None, "系统级失败已沉淀入参")
    check(row.last_error_args.get("table") == "t1", f"沉淀入参含调用参数（实际 {row.last_error_args}）")

    # ---------------------------------------------------------------
    # 阶段三：OPEN 期间快速失败（fail-fast）
    # ---------------------------------------------------------------
    print("\n[阶段三] OPEN 期间快速失败")
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
    # 阶段四：入参脱敏沉淀（敏感键排除 / 值截断 / 整体限长）
    # ---------------------------------------------------------------
    print("\n[阶段四] 入参脱敏沉淀")
    clean = _sanitize_args({"table": "t1", "api_key": "sk-secret", "filters": {"kw": "x" * 300}})
    check("api_key" not in clean, "敏感键（api_key）被排除")
    check(clean["table"] == "t1", "普通标量原样保留")
    check(len(clean["filters"]["kw"]) <= 256, f"超长字符串被截断（len={len(clean['filters']['kw'])})")
    # 单值会被截断到 256 字符，需多个键叠加才可能触发整体 4096 限长
    huge = _sanitize_args({f"k{i}": "y" * 300 for i in range(30)})
    check(huge.get("_truncated") is True, "整体超限标记 _truncated（探测时跳过留人工）")
    check(_sanitize_args(None) is None and _sanitize_args({}) is None, "空入参不沉淀")

    # ---------------------------------------------------------------
    # 阶段五：定时探测成功 → auto_recover 恢复（OPEN → CLOSED）
    # ---------------------------------------------------------------
    print("\n[阶段五] 定时探测成功，auto_recover 自动恢复")
    recovered = await breaker.auto_recover(db, TOOL_NAME)
    check(recovered is True, "auto_recover 返回 True（已恢复）")
    check(row.status == ToolStatus.ACTIVE, "status 已恢复为 1（生效）")
    check(row.circuit_state == ToolCircuitState.CLOSED, "circuit_state 已回到 CLOSED")
    check(row.consecutive_failures == 0, "连续失败计数已清零")
    check(row.remark == "定时探测成功，自动恢复", "已记录恢复原因")
    print(f"  恢复后状态: {snapshot(row)}")

    # 恢复后正常放行
    handler.fail_exc = None
    result = await breaker.call(db, TOOL_NAME, handler, query="ok")
    await breaker.flush(db)
    check(result == {"ok": True, "echo": {"query": "ok"}}, "恢复后工具调用正常放行")
    check(row.success_calls >= 1, "成功计数已累计")

    # ---------------------------------------------------------------
    # 阶段六：非熔断态不误恢复；探测失败保持熔断
    # ---------------------------------------------------------------
    print("\n[阶段六] 边界：非熔断态不误恢复 / 再次熔断后保持 OPEN")
    again = await breaker.auto_recover(db, TOOL_NAME)
    check(again is False, "工具已生效时 auto_recover 返回 False（不误操作）")

    # 再次系统故障熔断，模拟"探测失败 → 不调用 auto_recover"：状态保持 OPEN
    handler.fail_exc = DbToolSystemError
    for i in range(1, 4):
        try:
            await breaker.call(db, TOOL_NAME, handler, query="fail-again")
        except DbToolSystemError:
            pass
        await breaker.flush(db)
    check(row.circuit_state == ToolCircuitState.OPEN and row.status == ToolStatus.CIRCUIT_BREAK,
          f"再次连败 3 次进入 OPEN（实际 {snapshot(row)}）")

    # 失败后不恢复 → 状态保持熔断（等价于探测失败分支：_probe_one 不调 auto_recover）
    try:
        await breaker.call(db, TOOL_NAME, handler, query="x")
        check(False, "熔断态应快速失败")
    except CircuitOpenError:
        check(True, "探测失败（未恢复）时工具保持熔断、继续快速失败")
    check(row.status == ToolStatus.CIRCUIT_BREAK, "状态保持 2（熔断），等待下一轮探测/人工恢复")
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
    print("全部断言通过：熔断口径 / 入参沉淀 / OPEN 快速失败 / 定时探测恢复验证成功")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
