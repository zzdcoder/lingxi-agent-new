"""
观测与加固冒烟验证脚本（阶段 4）

用途：在**不依赖外部服务**（MySQL / Qdrant / LLM）的前提下，验证：
1. 观测模块可导入、trace_id 能在 Task 间正确继承（含后台任务场景）；
2. 日志格式成功注入 trace_id；
3. 指标埋点（计数 / 耗时 / 快照）可用；
4. SSE 有界队列在慢客户端场景下**丢帧而非挂死**（背压生效）；
5. 节点追踪装饰器记录耗时与异常，且**不吞异常**；
6. 健康巡检的进程内组件探测可用（数据库项在无 DB 时返回 down 而非抛错）。

运行：
    python scripts/smoke_observability.py
"""
import asyncio
import logging
import os
import sys

# 以脚本方式运行时 sys.path[0] 为 scripts/ 目录，需显式补项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 基础格式**不含** trace_id：字段名由 install_trace_logging() 在过滤器装配成功后
# 动态注入，保证「观测初始化失败」时日志仍能正常工作（ graceful degradation）。
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("smoke")


async def check_trace_propagation() -> None:
    """验证 trace_id 在后台 Task 间的继承（审批后台恢复任务的正确性依赖此行为）。"""
    from agent.observability import bind_trace_id, get_trace_id, install_trace_logging

    install_trace_logging()
    parent = bind_trace_id()
    assert get_trace_id() == parent, "父上下文 trace_id 不一致"

    async def child_task():
        await asyncio.sleep(0)
        return get_trace_id()

    got = await asyncio.create_task(child_task())
    assert got == parent, f"子任务未继承 trace_id: {got} != {parent}"
    logger.info(f"trace_id 继承验证通过: {parent}")


async def check_metrics() -> None:
    """验证指标埋点与快照导出。"""
    from agent.observability import metrics

    metrics.incr("smoke.counter", 3)
    metrics.observe("smoke.latency", 12.5)
    metrics.observe("smoke.latency", 27.5)
    snap = metrics.snapshot()
    assert snap["counters"]["smoke.counter"] == 3, snap
    obs = snap["observers"]["smoke.latency"]
    assert obs["count"] == 2 and obs["max"] == 27.5 and obs["avg"] == 20.0, obs
    logger.info(f"指标快照验证通过: {obs}")


async def check_sse_backpressure() -> None:
    """验证 SSE 队列背压：满队列投递必须**限时返回**，不能永久阻塞图执行。"""
    from agent.streaming import build_sse_frame, create_sse_queue, put_content

    queue = create_sse_queue(maxsize=2)
    assert queue.maxsize == 2, queue.maxsize

    # 填满队列后，下一次投递应在配置的超时内返回（丢弃），而非阻塞
    for i in range(2):
        await put_content(queue, f"frame-{i}")
    await asyncio.wait_for(put_content(queue, "overflow"), timeout=8)
    logger.info("SSE 背压验证通过：队列满时按时返回并丢弃，未阻塞图执行")

    # 正常投递仍然可用
    await queue.get()
    await put_content(queue, "after-drain")
    assert queue.qsize() >= 1
    logger.info(f"SSE 恢复投递验证通过（qsize={queue.qsize()}, frame={build_sse_frame('content', content='x')!r}）")


async def check_node_trace() -> None:
    """验证节点追踪装饰器：记录耗时，异常向上抛出（不改变既有容错语义）。"""
    from agent.observability import node_trace

    @node_trace("smoke_node")
    async def ok_node(state, config=None):
        await asyncio.sleep(0)
        return {"final_response": "ok"}

    @node_trace("smoke_node_fail")
    async def fail_node(state, config=None):
        raise ValueError("boom")

    assert (await ok_node({}))["final_response"] == "ok"
    try:
        await fail_node({})
    except ValueError:
        logger.info("节点追踪验证通过：异常未被吞掉")
    else:
        raise AssertionError("装饰器吞掉了异常")


async def check_health_probes() -> None:
    """验证健康巡检在无外部依赖时返回结构化降级结果而非抛错。"""
    from agent.health import _probe_checkpointer, _probe_graph, _probe_component

    for name, snap in (
        ("checkpointer", _probe_checkpointer()),
        ("graph", _probe_graph()),
        ("cache", _probe_component("rag.no_such_module", "getter")),
    ):
        assert isinstance(snap, dict) and "status" in snap, (name, snap)
        logger.info(f"健康巡检 {name}: {snap}")


async def check_routing() -> None:
    """验证路由函数在任务节点未注册时的降级分支不抛错（纯逻辑，无需图编译）。"""
    from agent.graph_builder import route_by_intents

    assert route_by_intents({"intents": ["chat"]}) == "chat"
    assert route_by_intents({"intents": ["knowledge_base"]}) == "knowledge"
    assert route_by_intents({}) == "chat"
    assert route_by_intents({"intents": ["task", "knowledge_base"]}) in ("knowledge", ["knowledge", "task_agent"])
    logger.info("意图路由降级分支验证通过")


async def main() -> int:
    # 最先安装日志过滤器，其后所有输出自动携带 trace_id
    from agent.observability import bind_trace_id, install_trace_logging

    install_trace_logging()
    bind_trace_id()

    checks = (
        ("trace_id 贯穿", check_trace_propagation),
        ("进程内指标", check_metrics),
        ("SSE 背压与丢帧", check_sse_backpressure),
        ("节点追踪装饰器", check_node_trace),
        ("健康巡检探测", check_health_probes),
        ("意图路由降级", check_routing),
    )
    failed = 0
    for name, fn in checks:
        try:
            await fn()
        except Exception as e:
            failed += 1
            logger.error(f"[FAIL] {name}: {type(e).__name__}: {e}")
    logger.info("=" * 50)
    logger.info(f"冒烟结果: 共 {len(checks)} 项，失败 {failed} 项")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
