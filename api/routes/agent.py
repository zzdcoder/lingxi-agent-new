"""
Agent 路由

提供智能对话（意图自动路由 + SSE 流式）与任务/审批状态查询接口。
"""
import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.exceptions import ConversationException
from models.agent_schema import AgentChatRequest
from models.user_model import User
from utils.auth import get_current_user
from agent.graph_builder import get_graph
from agent.streaming import build_sse_frame, put_status
from agent.approval import approval_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["Agent"])


# ==========================================
# 审批中断处理
# ==========================================

async def _handle_interrupt(
    result: dict,
    queue: asyncio.Queue,
    db: AsyncSession,
    current_user: User,
    conversation_id: str,
) -> Optional[str]:
    """
    处理图执行中的 HITL 审批中断。

    1. 解析 HITLRequest（action_requests），落库审批单（pending）；
    2. 创建飞书审批实例（配置缺失降级）；
    3. SSE 推送 approval_required 事件并注册队列（供后台恢复推送结果）。

    :return: 审批单 ID（处理失败返回 None）
    """
    interrupted = result.get("__interrupt__") or []
    if not interrupted:
        return None
    # 兼容 Interrupt 对象（.value）与裸 dict 两种形态
    interrupt_value = getattr(interrupted[0], "value", interrupted[0])
    action_requests = (interrupt_value or {}).get("action_requests") or []
    if not action_requests:
        logger.error("[Agent] 中断载荷缺少 action_requests")
        return None

    approval_id = await approval_service.save_pending_approval(
        db,
        conversation_id=conversation_id,
        requester_id=current_user.username,
        action_requests=action_requests,
    )
    await approval_service.create_feishu_instance(
        db, approval_id, current_user.username
    )
    await put_status(queue, "approval_required", approval_id=approval_id)
    approval_service.register_queue(approval_id, queue)
    logger.info(f"[Agent] 审批待处理: approval_id={approval_id}")
    return approval_id


# ==========================================
# 智能对话接口（意图自动路由 + 流式）
# ==========================================

@router.post("/chat")
async def agent_chat(
    chat_request: AgentChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    智能对话接口（SSE 流式）

    内部执行：意图识别 → 按意图路由（知识库 / 聊天 / 任务执行）→ 汇总 → 收尾。
    写操作（update/delete）命中时进入人工审批流程（审批单 + SSE 状态事件）。

    :param chat_request: 智能对话请求
    :param db: 数据库会话
    :param current_user: 当前用户
    :return: SSE 流式响应
    """
    logger.info(
        f"[Agent] 开始智能对话: conversation_id={chat_request.conversation_id}, "
        f"message_count={len(chat_request.messages)}, model={chat_request.model}"
    )

    graph = get_graph()

    # SSE 事件队列：图节点将 token/状态推入，API 层消费转发
    queue: asyncio.Queue = asyncio.Queue()

    inputs = {
        "messages": chat_request.messages,
        "conversation_id": chat_request.conversation_id,
        "username": current_user.username,
        "user_id": str(current_user.id),
        "model": chat_request.model,
        "tool_calls": [],
        "tool_results": [],
    }

    config = {
        "configurable": {
            "thread_id": chat_request.conversation_id,
            "sse_queue": queue,
            "db": db,
        }
    }

    async def run_graph() -> None:
        """后台执行主图，异常兜底推送错误帧，结束推送 sentinel。

        审批路径：图执行命中 HITL 中断（写操作）时——
        1. 落库审批单 + 创建飞书审批实例；
        2. SSE 推送 approval_required 事件；
        3. 等待审批恢复完成（心跳保活），结果由后台恢复任务经同一队列推送；
        4. 结束推送 sentinel。
        """
        approval_id = None
        try:
            result = await graph.ainvoke(inputs, config)
            if isinstance(result, dict) and result.get("__interrupt__"):
                approval_id = await _handle_interrupt(
                    result, queue, db, current_user, chat_request.conversation_id
                )
                if approval_id:
                    # 等待审批完成（心跳保活）；超时后结束等待，可轮询兜底
                    await approval_service.wait_for_final(approval_id, queue)
        except Exception as e:
            logger.exception(f"[Agent] 图执行异常: {e}")
            await queue.put(
                build_sse_frame("content", content=f"处理失败：{str(e)}")
            )
        finally:
            if approval_id:
                approval_service.unregister_queue(approval_id)
            await queue.put(None)  # sentinel：通知生成器结束

    async def event_generator():
        """SSE 事件生成器：消费队列并转发，图结束后发送完成标记。"""
        task = asyncio.create_task(run_graph())
        try:
            while True:
                frame = await queue.get()
                if frame is None:
                    break
                yield frame
            # 图执行完成
            yield build_sse_frame("done")
        finally:
            if not task.done():
                task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ==========================================
# 任务状态查询接口（断连轮询兜底）
# ==========================================

@router.get("/tasks/{task_execution_id}")
async def get_task_status(
    task_execution_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    查询任务执行状态与结果（SSE 断连时前端轮询兜底）。

    :param task_execution_id: 任务执行记录 ID
    :param db: 数据库会话
    :param current_user: 当前用户
    :return: 任务执行记录
    """
    try:
        from models.task_model import TaskExecution
        from sqlalchemy import select
        result = await db.execute(
            select(TaskExecution).where(TaskExecution.id == task_execution_id)
        )
        task = result.scalars().first()
        if not task:
            raise ConversationException(f"任务不存在: {task_execution_id}")
        return task
    except ConversationException:
        raise
    except Exception as e:
        logger.error(f"查询任务状态失败: {e}")
        raise ConversationException(f"查询任务状态失败: {str(e)}")
