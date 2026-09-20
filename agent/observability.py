"""
统一观测模块（设计文档 §15「观测与加固」，阶段 4）

企业级 Agent 服务要求**全链路可追踪、可度量、可诊断**。本模块提供四件能力：

1. **trace_id 贯穿**：基于 `contextvars` 在一次请求（含后台恢复任务）生命期内
   贯穿 API 层 → 图节点 → 工具 → 审计落库，并自动注入每条日志
   （`%(trace_id)s`），无需各模块手工传参；
2. **LangSmith 追踪**：集中管理 `LANGSMITH_*` 环境变量与开关，统一构造
   run metadata / tags，使链路在 LangSmith 控制台可按会话、用户、意图检索；
3. **进程内指标（Metrics）**：轻量级计数器 / 耗时观测器，覆盖请求量、
   意图分布、节点耗时、工具调用、SSE 丢帧等关键信号，暴露给
   `GET /api/agent/metrics`（Prometheus 接入前的内建观测面）；
4. **节点/阶段追踪装饰器**：`node_trace` 包裹图节点，`stage` 包裹任意阶段，
   自动记录耗时、成功/失败与异常详情，保证「零日志盲区」。

设计约束：
- **零第三方依赖**（不强制 import langsmith），缺失时自动降级；
- **失败静默**：观测自身异常绝不允许影响主业务链路（best-effort）；
- **低成本**：目标是「可观测」，不是「APM 全家桶」——计数在内存聚合，
  由 /api/agent/metrics 按需拉取，不引入常驻推送线程。
"""
import asyncio
import functools
import logging
import os
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Dict, Iterator, Optional

logger = logging.getLogger("agent.observability")

# =============================================================================
# 1. trace_id 贯穿
# =============================================================================

_TRACE_ID_KEY = "trace_id"
_no_trace = "-"

_trace_id_var: ContextVar[str] = ContextVar(_TRACE_ID_KEY, default=_no_trace)


def new_trace_id() -> str:
    """生成短 trace_id（16 位 hex，兼顾可读性与碰撞概率）。"""
    return uuid.uuid4().hex[:16]


def get_trace_id() -> str:
    """获取当前上下文的 trace_id（未绑定返回占位符 `-`）。"""
    return _trace_id_var.get() or _no_trace


def bind_trace_id(trace_id: Optional[str] = None) -> str:
    """
    在当前上下文绑定 trace_id。

    FastAPI 每个请求自成 Task 上下文，绑定后由其派生的协程/后台任务
    （`asyncio.create_task` 会复制当前 contextvars）自动继承。

    :param trace_id: 指定 ID；为空则自动生成
    :return: 实际绑定的 trace_id（便于回写响应头 / 落库）
    """
    tid = trace_id or new_trace_id()
    _trace_id_var.set(tid)
    return tid


class TraceIdFilter(logging.Filter):
    """日志过滤器：把当前上下文 trace_id 注入 LogRecord。

    用法：handler.addFilter(TraceIdFilter())，格式串中使用 %(trace_id)s。
    所有经过 logging 的记录都会自动带上链路标识，无需调用方感知。
    """

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        if not hasattr(record, _TRACE_ID_KEY):
            record.trace_id = get_trace_id()  # type: ignore[attr-defined]
        return True


def install_trace_logging(level: int = logging.INFO) -> bool:
    """
    为根 logger 的所有 handler 安装 trace_id 过滤器，并改造格式串。

    幂等：重复调用不会重复注入，也不会破坏已有自定义格式。

    :return: 是否成功安装
    """
    try:
        root = logging.getLogger()
        installed = getattr(root, "_trace_installed", False)
        if installed:
            return True

        for handler in root.handlers:
            if not any(isinstance(f, TraceIdFilter) for f in handler.filters):
                handler.addFilter(TraceIdFilter())
            fmt = handler.formatter
            if fmt is not None and "%(trace_id)s" not in (fmt._fmt or ""):
                base = fmt._fmt or "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                handler.setFormatter(
                    logging.Formatter(base.replace("- %(name)s", "- [trace=%(trace_id)s] %(name)s"))
                )
        setattr(root, "_trace_installed", True)
        logger.info(f"[观测] trace_id 日志注入已启用 (level={logging.getLevelName(level)})")
        return True
    except Exception as e:  # 观测自身异常不得影响启动
        logging.getLogger(__name__).warning(f"[观测] trace_id 日志注入失败（忽略）: {e}")
        return False


# =============================================================================
# 2. LangSmith 追踪
# =============================================================================

_TRACING_ENV_FLAG = ("true", "1", "yes", "on")


def _is_truthy(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in _TRACING_ENV_FLAG


def init_tracing(
    enabled: bool,
    project: str,
    endpoint: str,
    *,
    api_key_env: str = "LANGSMITH_API_KEY",
) -> bool:
    """
    初始化 LangSmith 链路追踪（幂等，失败降级）。

    最佳实践：由**环境变量**在进程启动时注入（`LANGSMITH_TRACING=true` 等），
    因 langsmith 在 import 时读取配置；本函数在此基础上做统一校验与补齐，
    使 `core/config.py` + `.env.dev` 亦能驱动开关，并输出明确的诊断日志。

    :param enabled: 是否启用追踪（settings.langsmith_tracing）
    :param project: 项目名（LangSmith 控制台可见）
    :param endpoint: LangSmith 服务端点（自建/官方）
    :param api_key_env: 读取 API Key 的环境变量名（不落日志）
    :return: 最终是否生效
    """
    try:
        if enabled:
            os.environ.setdefault("LANGSMITH_TRACING", "true")
            os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
            # trace_id 同步上报，便于 LangSmith 与本地日志双向对照
            os.environ.setdefault("LANGCHAIN_PROJECT", project)
            os.environ.setdefault("LANGSMITH_PROJECT", project)
            if endpoint:
                os.environ.setdefault("LANGSMITH_ENDPOINT", endpoint)
                os.environ.setdefault("LANGCHAIN_ENDPOINT", endpoint)
        else:
            os.environ["LANGSMITH_TRACING"] = "false"
            os.environ["LANGCHAIN_TRACING_V2"] = "false"

        active = _is_truthy(os.environ.get("LANGSMITH_TRACING")) and bool(
            os.environ.get(api_key_env)
        )
        if _is_truthy(os.environ.get("LANGSMITH_TRACING")) and not os.environ.get(api_key_env):
            logger.warning(
                "[观测] LangSmith 追踪已开启但未检测到 %s，链路不会上报（仅本地日志可用）",
                api_key_env,
            )
        else:
            logger.info(
                f"[观测] LangSmith 追踪 {'已启用' if active else '未启用'}"
                f"（project={project or '-'}）"
            )
        return active
    except Exception as e:
        logger.warning(f"[观测] LangSmith 初始化失败（降级为本地日志观测）: {e}")
        return False


def run_metadata(
    *,
    conversation_id: Optional[str] = None,
    user_id: Optional[str] = None,
    username: Optional[str] = None,
    intent: Optional[str] = None,
    task_execution_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    构造 LangSmith run metadata / tags，使链路可按业务维度检索。

    用于 `graph.ainvoke(config={"metadata": ..., "tags": ...})`；
    trace_id 一并写入，实现「LangSmith trace ↔ 本地日志 ↔ 审计记录」三方对照。
    """
    metadata: Dict[str, Any] = {"trace_id": get_trace_id()}
    for key, value in (
        ("conversation_id", conversation_id),
        ("user_id", user_id),
        ("username", username),
        ("intent", intent),
        ("task_execution_id", task_execution_id),
    ):
        if value:
            metadata[key] = value
    if extra:
        metadata.update(extra)
    return metadata


def run_tags(intent: Optional[str] = None, approval_mode: bool = False) -> list:
    """构造 LangSmith tags（粗粒度过滤维度）。"""
    tags = ["lingxi-agent"]
    if intent:
        tags.append(f"intent:{intent}")
    tags.append("approval" if approval_mode else "no-approval")
    return tags


# =============================================================================
# 3. 进程内指标
# =============================================================================

class MetricsRegistry:
    """
    轻量进程内指标注册表（线程安全）。

    - `incr`：单调计数（请求数、错误数、丢帧数…）；
    - `observe`：数值观测（耗时毫秒、返回条数…），输出 count/sum/min/max/avg。

    说明：单进程内存聚合，多 worker 部署时各进程独立统计，需由拉取方合并；
    正式接入 Prometheus 时只需在 `snapshot()` 之上做格式适配，调用点零改动。
    """

    _MAX_KEYS = 512  # 防御：避免高基数标签（如会话ID）撑爆内存

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Dict[str, float] = {}
        self._observers: Dict[str, Dict[str, float]] = {}
        self._started_at = time.time()

    def incr(self, key: str, value: float = 1.0) -> None:
        """计数累加（key 超过上限时记录观测告警并丢弃，防内存膨胀）。"""
        try:
            with self._lock:
                if key not in self._counters and len(self._counters) >= self._MAX_KEYS:
                    logger.warning(f"[观测] 指标 key 数量达上限，丢弃 {key}")
                    return
                self._counters[key] = self._counters.get(key, 0.0) + value
        except Exception:
            pass  # 观测失败静默

    def observe(self, key: str, value: float) -> None:
        """数值观测：累计 count / sum / min / max / avg。"""
        try:
            with self._lock:
                if key not in self._observers and len(self._observers) >= self._MAX_KEYS:
                    logger.warning(f"[观测] 观测 key 数量达上限，丢弃 {key}")
                    return
                stat = self._observers.setdefault(
                    key, {"count": 0.0, "sum": 0.0, "min": value, "max": value}
                )
                stat["count"] += 1
                stat["sum"] += value
                stat["min"] = min(stat["min"], value)
                stat["max"] = max(stat["max"], value)
        except Exception:
            pass

    def timing(self, key: str, value_ms: float) -> None:
        """耗时观测（毫秒），语义等价于 observe。"""
        self.observe(key, value_ms)

    def snapshot(self) -> Dict[str, Any]:
        """导出全量快照（sorted，便于 diff / 展示）。"""
        with self._lock:
            observers = {
                k: {
                    "count": int(v["count"]),
                    "sum": round(v["sum"], 2),
                    "min": round(v["min"], 2),
                    "max": round(v["max"], 2),
                    "avg": round(v["sum"] / v["count"], 2) if v["count"] else 0.0,
                }
                for k, v in sorted(self._observers.items())
            }
            return {
                "uptime_seconds": int(time.time() - self._started_at),
                "counters": {k: v for k, v in sorted(self._counters.items())},
                "observers": observers,
            }

    def reset(self) -> None:
        """清空指标（测试用）。"""
        with self._lock:
            self._counters.clear()
            self._observers.clear()
            self._started_at = time.time()


metrics = MetricsRegistry()

# 指标 key 命名规范：域.对象.语义，避免高基数（禁止把会话ID/用户ID 拼进 key）
K_REQUEST_TOTAL = "agent.request.total"
K_REQUEST_FAILED = "agent.request.failed"
K_REQUEST_TIMEOUT = "agent.request.timeout"
K_REQUEST_LATENCY = "agent.request.latency_ms"
K_INTENT_PREFIX = "agent.intent."
K_NODE_LATENCY = "agent.node.{}.latency_ms"
K_NODE_TOTAL = "agent.node.{}.total"
K_NODE_ERROR = "agent.node.{}.errors"
K_INTERRUPT = "agent.interrupt.{}"
K_APPROVAL_PREFIX = "agent.approval.{}"
K_CLARIFY_PREFIX = "agent.clarify.{}"
K_TOOL_PREFIX = "agent.tool.{}"
K_SSE_DROPPED = "agent.sse.frame_dropped"
K_PROBE_PREFIX = "agent.probe.{}"


# =============================================================================
# 4. 节点 / 阶段追踪
# =============================================================================

def node_trace(node_name: str) -> Callable:
    """
    图节点追踪装饰器：记录调用量、耗时、异常并上报指标。

    用法：
        @node_trace("knowledge")
        async def knowledge_node(state, config=None): ...

    约定：LangGraph 节点签名为 `(state, config)`，装饰器据此透传参数；
    异常向上抛出（不吞异常，避免改变图的既有容错语义），仅做观测与计时。
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            metrics.incr(K_NODE_TOTAL.format(node_name))
            try:
                result = await func(*args, **kwargs)
                cost = (time.perf_counter() - start) * 1000
                metrics.timing(K_NODE_LATENCY.format(node_name), cost)
                logger.debug(
                    f"[观测] 节点 {node_name} 完成 耗时={cost:.0f}ms trace_id={get_trace_id()}"
                )
                return result
            except Exception as e:
                cost = (time.perf_counter() - start) * 1000
                metrics.incr(K_NODE_ERROR.format(node_name))
                metrics.timing(K_NODE_LATENCY.format(node_name), cost)
                logger.warning(
                    f"[观测] 节点 {node_name} 异常: {type(e).__name__}: {e} "
                    f"耗时={cost:.0f}ms trace_id={get_trace_id()}"
                )
                raise

        return wrapper

    return decorator


@contextmanager
def stage(name: str, **fields: Any) -> Iterator[Callable[..., None]]:
    """
    阶段观测上下文：记录任意代码段的耗时与结果标签。

    用法：
        with stage("merge_llm", intent="task") as done:
            ...
            done(ok=True, tokens=120)   # 可选：附加结构化字段
    """

    def _dump(extra: Dict[str, Any]) -> str:
        return " ".join(f"{k}={v}" for k, v in extra.items())

    start = time.perf_counter()
    base = f"stage={name} trace_id={get_trace_id()}"
    logger.debug(f"[观测] {base} 开始 {_dump(fields)}")
    try:
        yield lambda **extra: logger.debug(f"[观测] {base} 标记 {_dump(extra)}")
    except Exception as e:
        cost = (time.perf_counter() - start) * 1000
        metrics.incr(f"agent.stage.{name}.errors")
        metrics.timing(f"agent.stage.{name}.latency_ms", cost)
        logger.warning(f"[观测] {base} 失败: {type(e).__name__}: {e} 耗时={cost:.0f}ms")
        raise
    else:
        cost = (time.perf_counter() - start) * 1000
        metrics.timing(f"agent.stage.{name}.latency_ms", cost)
        logger.debug(f"[观测] {base} 结束 耗时={cost:.0f}ms")


async def call_with_timeout(coro, timeout: Optional[float], label: str):
    """
    统一的 awaitable 超时包装（带观测，超时转译为可读错误由调用方处理）。

    :param coro: 可等待对象
    :param timeout: 超时秒数；None / <=0 表示不设限（如审批等待期）
    :param label: 观测标签（写入metrics的 key 片段）
    :return: 协程结果
    :raises asyncio.TimeoutError: 超时时抛出，交由调用方转友好错误
    """
    if not timeout or timeout <= 0:
        return await coro
    start = time.perf_counter()
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        metrics.incr(f"agent.timeout.{label}")
        logger.warning(
            f"[观测] {label} 超时(>{timeout}s) trace_id={get_trace_id()}"
        )
        raise
    finally:
        metrics.timing(f"agent.stage.{label}.latency_ms", (time.perf_counter() - start) * 1000)
