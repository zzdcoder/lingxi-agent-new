"""
Agent 路由

提供智能对话（意图自动路由 + SSE 流式）与任务/审批/追问状态查询接口。
"""
import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.exceptions import ConversationException
from models.agent_schema import AgentChatRequest, ClarifyAnswerIn
from models.approval_model import ApprovalRequest
from models.user_model import User
from utils.auth import get_current_user
from agent.graph_builder import get_checkpointer, get_graph
from agent.streaming import build_sse_frame, create_sse_queue, put_frame, put_status
from agent.approval import approval_service
from agent.approval import clarify_service
from agent.observability import (
    K_REQUEST_FAILED,
    K_REQUEST_LATENCY,
    K_REQUEST_TIMEOUT,
    K_REQUEST_TOTAL,
    bind_trace_id,
    get_trace_id,
    metrics,
    run_metadata,
    run_tags,
)
from core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["Agent"])


# ==========================================
# 人工介入中断处理（写审批 / 任务追问）
# ==========================================

async def _handle_interrupt(
    result: dict,
    queue: asyncio.Queue,
    db: AsyncSession,
    current_user: User,
    conversation_id: str,
) -> Optional[str]:
    """
    处理图执行中的人工介入中断（写操作审批 / 任务追问）。

    按中断载荷类型分流：
    - type=clarification：LLM 信息不足追问，落库追问单 + SSE 推送追问卡片；
    - action_requests：写操作审批，脱敏落库审批单 + SSE 推送审批卡片。

    两类统一注册 SSE 队列（按 ID 索引），等待恢复完成（心跳保活）。

    :return: 介入单 ID（审批单或追问单，处理失败返回 None）
    """
    interrupted = result.get("__interrupt__") or []
    if not interrupted:
        return None
    # 兼容 Interrupt 对象（.value）与裸 dict 两种形态
    interrupt_value = getattr(interrupted[0], "value", interrupted[0])
    if not isinstance(interrupt_value, dict):
        return None

    # 任务追问：LLM 信息不足时向用户追问
    if interrupt_value.get("type") == clarify_service.BIZ_TYPE_CLARIFY:
        return await _handle_clarify_interrupt(
            interrupt_value, queue, db, current_user, conversation_id
        )

    # 写操作审批（既有链路）
    action_requests = interrupt_value.get("action_requests") or []
    if not action_requests:
        logger.error("[Agent] 中断载荷缺少 action_requests")
        return None

    approval = await approval_service.save_pending_approval(
        db,
        conversation_id=conversation_id,
        requester_id=current_user.username,
        action_requests=action_requests,
    )
    await put_status(
        queue,
        "approval_required",
        approval_id=approval.id,
        approval=approval_service.build_card_payload(approval),
    )
    approval_service.register_queue(approval.id, queue)
    logger.info(f"[Agent] 审批待处理: approval_id={approval.id}")
    return approval.id


async def _handle_clarify_interrupt(
    interrupt_value: dict,
    queue: asyncio.Queue,
    db: AsyncSession,
    current_user: User,
    conversation_id: str,
) -> Optional[str]:
    """
    处理 LLM 追问中断：落库追问单 + SSE 推送追问卡片 + 注册队列等待。

    LLM 可能一次追问多个问题（interrupt 载荷携带 questions 列表），
    兼容旧载荷单问题（question 字段）。

    :return: 追问单 ID（处理失败返回 None）
    """
    # 批量问题（新载荷）；兼容旧载荷单问题
    questions = interrupt_value.get("questions")
    if not isinstance(questions, list) or not questions:
        legacy = str(interrupt_value.get("question") or "").strip()
        questions = [legacy] if legacy else []
    questions = [str(q).strip() for q in questions if str(q).strip()]
    if not questions:
        logger.error("[Agent] 追问中断缺少 questions")
        return None
    clarify = await clarify_service.save_pending_clarify(
        db,
        conversation_id=conversation_id,
        requester_id=current_user.username,
        questions=questions,
    )
    await put_status(
        queue,
        "clarification_required",
        clarify_id=clarify.id,
        questions=questions,
        clarification=clarify_service.build_card_payload(clarify),
    )
    approval_service.register_queue(clarify.id, queue)
    logger.info(f"[Agent] 追问待回答: clarify_id={clarify.id}, 问题数={len(questions)}")
    return clarify.id


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
    # 阶段 4 观测：本次链路绑定 trace_id，贯穿日志 / LangSmith / 后台恢复任务
    trace_id = bind_trace_id()
    started = time.perf_counter()
    metrics.incr(K_REQUEST_TOTAL)
    logger.info(
        f"[Agent] 开始智能对话: conversation_id={chat_request.conversation_id}, "
        f"message_count={len(chat_request.messages)}, model={chat_request.model}, "
        f"user={current_user.username}, trace_id={trace_id}"
    )

    graph = get_graph()

    # SSE 事件队列（**有界**：慢客户端不会把服务端内存拖垮，见 agent/streaming.py）
    queue: asyncio.Queue = create_sse_queue()

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
            "trace_id": trace_id,
        },
        # LangSmith 检索维度：trace_id / 会话 / 用户 / 意图齐备，可与本地日志对照
        "recursion_limit": settings.agent_graph_recursion_limit,
        "metadata": run_metadata(
            conversation_id=chat_request.conversation_id,
            user_id=str(current_user.id),
            username=current_user.username,
        ),
        "tags": run_tags(approval_mode=get_checkpointer() is not None),
    }

    async def run_graph() -> None:
        """后台执行主图，异常兜底推送错误帧，结束推送 sentinel。

        人工介入路径：图执行命中 HITL 中断（写操作审批 / 任务追问）时——
        1. 落库介入单（审批单 / 追问单）；
        2. SSE 推送 approval_required / clarification_required 事件（含卡片载荷）；
        3. 等待前台恢复完成（心跳保活），决策回显与结果由后台恢复任务经同一队列推送；
        4. 结束推送 sentinel。
        """
        pending_id = None
        try:
            # 阶段 4 加固：主图执行整体墙钟超时兜底。
            # 命中审批/追问中断时 ainvoke 会立即返回 __interrupt__（不阻塞），
            # 故此处超时仅针对「非人为等待」的计算挂死场景。
            result = await asyncio.wait_for(
                graph.ainvoke(inputs, config),
                timeout=settings.agent_request_timeout_seconds,
            )
            if isinstance(result, dict) and result.get("__interrupt__"):
                pending_id = await _handle_interrupt(
                    result, queue, db, current_user, chat_request.conversation_id
                )
                if pending_id:
                    # 等待人工介入完成（心跳保活）；超时后结束等待，可轮询兜底
                    await approval_service.wait_for_final(pending_id, queue)
        except asyncio.TimeoutError:
            metrics.incr(K_REQUEST_TIMEOUT)
            error_msg = (
                f"处理超时（>{settings.agent_request_timeout_seconds}s），"
                f"请稍后重试或简化请求"
            )
            logger.warning(f"[Agent] {error_msg} trace_id={get_trace_id()}")
            await put_frame(queue, build_sse_frame("content", content=error_msg), "content")
        except Exception as e:
            metrics.incr(K_REQUEST_FAILED)
            logger.exception(f"[Agent] 图执行异常 trace_id={get_trace_id()}: {e}")
            await put_frame(
                queue, build_sse_frame("content", content=f"处理失败：{str(e)}"), "content"
            )
        finally:
            cost_ms = int((time.perf_counter() - started) * 1000)
            metrics.observe(K_REQUEST_LATENCY, cost_ms)
            logger.info(f"[Agent] 链路结束 耗时={cost_ms}ms trace_id={get_trace_id()}")
            if pending_id:
                approval_service.unregister_queue(pending_id)
            # sentinel：通知生成器结束（客户端已断开时队列可能无人消费，需限时丢弃）
            try:
                await asyncio.wait_for(queue.put(None), timeout=1.0)
            except Exception:
                pass

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
            # 阶段 4 观测：链路标识回传，便于前端日志/工单与服务端日志逐条对照
            settings.trace_id_header: trace_id,
        },
    )


# ==========================================
# 观测快照接口（阶段 4）
# ==========================================

@router.get("/metrics")
async def agent_metrics(current_user: User = Depends(get_current_user)):
    """
    观测快照（进程内指标，Prometheus 接入前的临时观测面）。

    返回计数器（请求量 / 失败 / 意图分布 / 工具调用 / 丢帧…）与耗时观测
    （请求与各节点 latency）。注意：多 worker 部署时各进程独立统计，
    需由拉取方按实例聚合。

    :return: 指标快照字典
    """
    if not settings.agent_metrics_enabled:
        return {"code": 0, "enabled": False, "detail": "指标采集已关闭"}
    return {
        "code": 0,
        "enabled": True,
        "trace_id": get_trace_id(),
        **metrics.snapshot(),
    }


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


# ==========================================
# 任务追问接口（HITL 澄清：提交答案 / 列表）
# ==========================================

@router.post("/clarify/{clarify_id}/answer")
async def submit_clarify_answer(
    clarify_id: str,
    body: ClarifyAnswerIn,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    前台提交追问答案接口（支持批量回答多问题，兼容旧单问题）。

    1. 校验追问单存在且 biz_type=clarification（避免误操作审批单）；
    2. 越权校验：仅发起人本人可回答（requester_id 比对，越权 403）；
    3. 受理后由 clarify_service.submit_answer 后台恢复图执行
       （幂等由状态机守卫，结果经原 SSE 连接推送，断连可轮询兜底）。

    :param clarify_id: 追问单 ID
    :param body: 答案请求体（answers 批量 / answer 单问题兼容）
    :param db: 数据库会话
    :param current_user: 当前用户（回答人）
    """
    try:
        trace_id = bind_trace_id()   # 追问答案为独立请求，绑定新 trace_id 便于串联后台恢复
        result = await db.execute(
            select(ApprovalRequest).where(ApprovalRequest.id == clarify_id)
        )
        clarify = result.scalars().first()
        if not clarify or clarify.biz_type != clarify_service.BIZ_TYPE_CLARIFY:
            raise ConversationException(f"追问单不存在: {clarify_id}")
        if clarify.requester_id and clarify.requester_id != current_user.username:
            raise ConversationException("仅发起人本人可回答该追问", code=403)

        # 批量答案优先；兼容旧单问题请求
        answers = body.answers if body.answers else ([body.answer] if body.answer else [])
        if not answers:
            raise ConversationException("追问答案不能为空")

        clarify_service.submit_answer(
            clarify_id,
            answers,
            answered_by=current_user.username,
        )
        logger.info(
            f"[Clarify] 前台提交答案已受理: clarify_id={clarify_id}, "
            f"answers={len(answers)}, answered_by={current_user.username}, "
            f"trace_id={trace_id}"
        )
        return {
            "code": 0,
            "clarify_id": clarify_id,
            "status": "answered",
            "trace_id": trace_id,
            "message": "追问已受理，结果将经会话流式推送",
        }
    except ConversationException:
        raise
    except Exception as e:
        logger.error(f"提交追问答案失败: {e}")
        raise ConversationException(f"提交追问答案失败: {str(e)}")


@router.post("/clarify/{clarify_id}/cancel")
async def cancel_clarify(
    clarify_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    前台取消追问接口（终止任务）。

    1. 校验追问单存在且 biz_type=clarification（避免误操作审批单）；
    2. 越权校验：仅发起人本人可取消（requester_id 比对，越权 403）；
    3. 受理后由 clarify_service.submit_cancel 后台恢复图执行：
       ask_user 工具收到取消信号，LLM 如实告知用户并给出修复建议，
       任务执行记录标记 canceled（终止态）。

    :param clarify_id: 追问单 ID
    :param db: 数据库会话
    :param current_user: 当前用户（取消人）
    """
    try:
        trace_id = bind_trace_id()   # 追问取消为独立请求，绑定新 trace_id 便于串联后台恢复
        result = await db.execute(
            select(ApprovalRequest).where(ApprovalRequest.id == clarify_id)
        )
        clarify = result.scalars().first()
        if not clarify or clarify.biz_type != clarify_service.BIZ_TYPE_CLARIFY:
            raise ConversationException(f"追问单不存在: {clarify_id}")
        if clarify.requester_id and clarify.requester_id != current_user.username:
            raise ConversationException("仅发起人本人可取消该追问", code=403)

        clarify_service.submit_cancel(
            clarify_id,
            canceled_by=current_user.username,
        )
        logger.info(
            f"[Clarify] 前台取消已受理: clarify_id={clarify_id}, "
            f"canceled_by={current_user.username}, trace_id={trace_id}"
        )
        return {
            "code": 0,
            "clarify_id": clarify_id,
            "status": "canceled",
            "trace_id": trace_id,
            "message": "追问已取消，任务将终止，Agent 会给出处理建议",
        }
    except ConversationException:
        raise
    except Exception as e:
        logger.error(f"取消追问失败: {e}")
        raise ConversationException(f"取消追问失败: {str(e)}")


@router.get("/clarify")
async def list_conversation_clarifies(
    conversation_id: str = Query(..., description="会话ID"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    按会话查询追问单列表（仅返回发起人本人的追问单）。

    前端刷新后据此重建追问卡片：pending 渲染可回答卡片，answered 渲染只读回显。

    :param conversation_id: 会话 ID
    :param db: 数据库会话
    :param current_user: 当前用户
    :return: 追问单列表（按创建时间倒序）
    """
    try:
        clarifies = await clarify_service.list_conversation_clarifies(
            db, conversation_id, current_user.username
        )
        return [clarify_service.build_card_payload(c) for c in clarifies]
    except Exception as e:
        logger.error(f"查询会话追问列表失败: {e}")
        raise ConversationException(f"查询会话追问列表失败: {str(e)}")
