"""
Agent SSE 流式辅助

图节点将生成 token / 状态事件推入 asyncio.Queue，由 API 层消费并转发为 SSE。
统一事件帧格式（与设计文档 §7.6 对齐）：
- {"event": "content", "content": "..."}   普通 token
- {"event": "status", "status": "...", ...} 任务/审批状态
- {"event": "ping"}                         心跳
- {"event": "done"}                         完成
"""
import asyncio
import json
from typing import Any, Optional


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
    for key, value in payload.items():
        if isinstance(value, str):
            data[key] = _escape_json(value)
        else:
            data[key] = value
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


async def put_content(queue: asyncio.Queue, content: str) -> None:
    """推送普通 token 事件"""
    if queue is not None:
        await queue.put(build_sse_frame("content", content=content))


async def put_status(queue: asyncio.Queue, status: str, **extra: Any) -> None:
    """推送状态事件（如 task_started / approval_required）"""
    if queue is not None:
        await queue.put(build_sse_frame("status", status=status, **extra))


async def put_done(queue: asyncio.Queue) -> None:
    """推送完成事件"""
    if queue is not None:
        await queue.put(build_sse_frame("done"))


async def put_ping(queue: asyncio.Queue) -> None:
    """推送心跳事件（审批等待期防连接超时）"""
    if queue is not None:
        await queue.put(build_sse_frame("ping"))


def get_sse_queue(config: dict) -> Optional[asyncio.Queue]:
    """从 LangGraph 运行配置中取出 SSE 队列（缺省返回 None）。"""
    return (config or {}).get("configurable", {}).get("sse_queue")
