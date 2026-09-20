"""
Agent SSE 流式辅助

图节点将生成 token / 状态事件推入 asyncio.Queue，由 API 层消费并转发为 SSE。
统一事件帧格式（与设计文档 §7.6 对齐）：
- {"event": "content", "content": "..."}   普通 token
- {"event": "status", "status": "...", ...} 任务/审批状态
- {"event": "ping"}                         心跳
- {"event": "done"}                         完成

**背压与丢帧保护（阶段 4 加固）**：

客户端消费慢 / 连接半开时，无界队列会无限堆积把服务端内存吃满。这里统一：
1. 事件队列**有界**（`agent_sse_queue_maxsize`），超限投递等待最多
   `agent_sse_put_timeout_seconds`（超时判定为慢客户端 → **丢弃并记录**）；
2. 投递失败**永不向上传播**（ historically 队列写失败会炸掉整个图执行），
   仅计数告警（`K_SSE_DROPPED` 指标），保证主链路可用性；
3. 所有投递动作经 `_safe_put`，观测埋点集中，消除此前的零日志盲区。
"""
import asyncio
import json
import logging
from typing import Any, Optional

from core.config import settings

logger = logging.getLogger(__name__)

# sentinel：通知 SSE 生成器结束（类型为 None 的帧）
# 说明：keep sentinel as None because api/routes/agent.py checks `frame is None`


def _escape_json(text: str) -> str:
    """转义 JSON 字符串中的特殊字符（与现有 SSE 编码保持一致）"""
    return (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
    )


def build_sse_frame(event: str, **payload: Any) -> bytes:
    """
    构建 SSE 数据帧。

    :param event: 事件类型
    :param payload: 事件附加字段
    :return: `data: {...}\n\n` 字节帧
    """
    data = {"event": event}
    data.update(payload)
    # json.dumps 已内置 JSON 字符串转义，无需额外 _escape_json（否则双重转义）
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


def create_sse_queue(maxsize: Optional[int] = None) -> asyncio.Queue:
    """
    创建 SSE 事件队列（**有界**，防慢客户端堆积内存）。

    :param maxsize: 队列容量；None 时取配置 `agent_sse_queue_maxsize`
                    （配置为 0 时退化为无界，仅建议在压测/排障时临时使用）
    """
    size = maxsize if maxsize is not None else settings.agent_sse_queue_maxsize
    return asyncio.Queue(maxsize=size if size and size > 0 else 0)


async def _safe_put(queue: asyncio.Queue, frame: bytes, kind: str) -> bool:
    """
    向队列安全投递事件帧（带超时与丢帧保护）。

    :param queue: 目标队列
    :param frame: 已编码的 SSE 帧
    :param kind: 事件类型（用于日志与指标）
    :return: 是否投递成功
    """
    if queue is None:
        return False
    try:
        from agent.observability import K_SSE_DROPPED, metrics

        timeout = settings.agent_sse_put_timeout_seconds
        if timeout and timeout > 0:
            await asyncio.wait_for(queue.put(frame), timeout=timeout)
        else:
            queue.put_nowait(frame)
        return True
    except asyncio.TimeoutError:
        # 慢客户端：队列长时间满 → 丢弃该帧，保护服务端内存（连接由上层超时兜底）
        try:
            metrics.incr(K_SSE_DROPPED)
        except Exception:
            pass
        logger.warning(
            f"[Agent-SSE] 队列持续满 >{settings.agent_sse_put_timeout_seconds}s，"
            f"丢弃 {kind} 帧（慢客户端）"
        )
        return False
    except asyncio.QueueFull:
        logger.warning(f"[Agent-SSE] 队列已满，丢弃 {kind} 帧")
        return False
    except Exception as e:
        # 投递异常绝不影响图执行主链路
        logger.warning(f"[Agent-SSE] 投递 {kind} 帧失败（忽略）: {e}")
        return False


async def put_frame(queue: asyncio.Queue, frame: bytes, kind: str = "frame") -> bool:
    """投递已构建的 SSE 帧（带超时丢帧保护），供 API 层直接下发自定义帧。"""
    return await _safe_put(queue, frame, kind)


async def put_content(queue: asyncio.Queue, content: str) -> None:
    """推送普通 token 事件"""
    if queue is not None and content:
        await _safe_put(queue, build_sse_frame("content", content=content), "content")


async def put_status(queue: asyncio.Queue, status: str, **extra: Any) -> None:
    """推送状态事件（如 task_started / approval_required）"""
    if queue is not None:
        frame = build_sse_frame("status", status=status, **extra)
        ok = await _safe_put(queue, frame, f"status:{status}")
        if ok:
            logger.debug(f"[Agent-SSE] 状态事件已推送: {status}")


async def put_done(queue: asyncio.Queue) -> None:
    """推送完成事件"""
    if queue is not None:
        await _safe_put(queue, build_sse_frame("done"), "done")


async def put_ping(queue: asyncio.Queue) -> None:
    """推送心跳事件（审批等待期防连接超时）"""
    if queue is not None:
        await _safe_put(queue, build_sse_frame("ping"), "ping")


def get_sse_queue(config: dict) -> Optional[asyncio.Queue]:
    """从 LangGraph 运行配置中取出 SSE 队列（缺省返回 None）。"""
    return (config or {}).get("configurable", {}).get("sse_queue")
