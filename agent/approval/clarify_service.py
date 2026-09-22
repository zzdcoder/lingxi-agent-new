"""
追问服务（复用人工介入单表 approval_request，biz_type=clarification）

职责（与写操作审批同构，设计文档 §5.6 / HITL §7 泛化）：
1. 解析图执行中断（__interrupt__ 中 type=clarification，支持一批多问题），
   落库追问单（approval_request, biz_type=clarification, status=pending）；
2. 构造追问卡片载荷（SSE clarification_required 事件携带 questions 列表，
   前台据此渲染多问题向导卡片：进度 / 下一步 / 提交 / 取消）；
3. 前台提交批量答案（POST /api/agent/clarify/{id}/answer）受理后，
   以 Command(resume={"answers": [...]}) 恢复同一 thread 的图执行，
   任务节点重入后继续处理（最终结果由任务节点经 SSE 推送）；
4. 前台取消（POST /api/agent/clarify/{id}/cancel）受理后，以
   Command(resume={"canceled": True}) 恢复，ask_user 工具收到取消信号，
   LLM 如实告知用户并给出修复建议，任务执行记录标记 canceled 终止；
5. 复用审批服务的 SSE 队列注册表 / 心跳等待 / 后台任务基础设施
   （按 ID 索引，追问单与审批单互不冲突）。

状态机：pending → answered / canceled（单向流转，幂等抢占）。
"""
import asyncio
import logging
from typing import Optional

from langgraph.types import Command
from sqlalchemy import select, update

from core.config import settings
from core.database import AsyncSessionLocal
from models.approval_model import ApprovalRequest
from models.task_model import TaskExecution
from agent.streaming import put_content, put_status
from agent.graph_builder import get_graph
from agent.approval import approval_service
from agent.observability import (
    K_CLARIFY_PREFIX,
    get_trace_id,
    metrics,
    run_metadata,
    run_tags,
)

logger = logging.getLogger(__name__)

# 追问单业务类型（与 approval_model.biz_type 列值一致）
BIZ_TYPE_CLARIFY = "clarification"

# 答案最大长度（与 approval_model.answer 列一致）
_ANSWER_MAX_LENGTH = 500


# =============================================================================
# 追问单落库
# =============================================================================

async def save_pending_clarify(
    db,
    *,
    conversation_id: str,
    requester_id: str,
    questions: list[str],
) -> ApprovalRequest:
    """
    落库追问单（pending，支持一批多个问题）。

    追问单关联任务执行记录（中断前任务节点已创建 status=running 的记录）。
    问题存 tool_params={"questions": [...]}（兼容旧数据 tool_params={"question": ...}）。

    :param db: 数据库会话
    :param conversation_id: 会话 ID（=thread_id）
    :param requester_id: 发起人（JWT username）
    :param questions: LLM 批量追问的问题（每个截断防异常）
    :return: 追问单对象（供卡片载荷构造）
    :raises ValueError: 问题列表为空
    """
    questions = [str(q).strip()[:_ANSWER_MAX_LENGTH] for q in (questions or []) if str(q).strip()]
    if not questions:
        raise ValueError("追问问题不能为空")

    # 追问单关联任务记录（中断前任务节点已创建 status=running 的记录）
    task = await _find_running_task(db, conversation_id)

    clarify = ApprovalRequest(
        task_execution_id=task.id if task else None,
        conversation_id=conversation_id,
        requester_id=requester_id,
        biz_type=BIZ_TYPE_CLARIFY,
        approval_type="clarification",
        tool_params={"questions": questions},
        status="pending",
    )
    db.add(clarify)
    await db.commit()
    await db.refresh(clarify)
    metrics.incr(K_CLARIFY_PREFIX.format("created"))
    logger.info(
        f"[Clarify] 追问单创建: id={clarify.id}, 问题数={len(questions)}, "
        f"首问={questions[0][:40]}, trace_id={get_trace_id()}"
    )
    return clarify


# =============================================================================
# 追问卡片载荷（前台可视化）
# =============================================================================

def _get_questions(clarify: ApprovalRequest) -> list[str]:
    """提取追问单的问题列表（兼容新 questions / 旧 question 存储）。"""
    params = clarify.tool_params or {}
    questions = params.get("questions")
    if isinstance(questions, list) and questions:
        return [str(q) for q in questions]
    legacy = params.get("question")
    return [str(legacy)] if legacy else []


def _get_answers(clarify: ApprovalRequest) -> list[str]:
    """提取追问单的回答列表（兼容批量 answers / 旧单 answer 存储）。"""
    payload = clarify.callback_payload or {}
    answers = payload.get("answers")
    if isinstance(answers, list) and answers:
        return [str(a) for a in answers]
    return [clarify.answer] if clarify.answer else []


def build_card_payload(clarify: ApprovalRequest) -> dict:
    """
    构造前台追问卡片载荷（SSE clarification_required 事件携带）。

    datetime 转字符串，避免 SSE JSON 编码失败；回答仅在已提交后非空。
    保留 question / answer 字段（单问题兼容，前端已逐步切到 questions / answers）。
    """
    created_at = clarify.created_at
    updated_at = clarify.updated_at
    questions = _get_questions(clarify)
    answers = _get_answers(clarify)
    return {
        "id": clarify.id,
        "question": questions[0] if questions else "",
        "questions": questions,
        "count": len(questions),
        "status": clarify.status,
        "requester_id": clarify.requester_id,
        "created_at": created_at.strftime("%Y-%m-%d %H:%M:%S") if created_at else None,
        "answer": answers[0] if answers else None,
        "answers": answers,
        "canceled": bool((clarify.callback_payload or {}).get("canceled")),
        "answered_by": clarify.approved_by,
        "answered_at": updated_at.strftime("%Y-%m-%d %H:%M:%S") if updated_at else None,
    }


# =============================================================================
# 追问恢复
# =============================================================================

def _resume_config(
    clarify: ApprovalRequest,
    queue,
    db,
    clarify_id: str,
    resume_value: dict,
) -> dict:
    """
    构造恢复图执行的运行配置（阶段 4：统一携带递归上限与 LangSmith 元数据）。

    §20 F3.0：`clarify_resume` 必须随 config 下发。
    主图的 `Command(resume=...)` 只能恢复**主图自己**的中断；`ask_user` 追问实际
    中断在内层任务 Agent（`agent/nodes/task_node.py`，独立 thread + checkpointer），
    主图的 resume 值到不了那里。task_node 重入时会读
    `configurable.clarify_resume` 并转交内层 Agent，否则答案/取消信号被静默丢弃
    → 内层 Agent 重新追问（二次中断），用户永远等不到结果。

    :param clarify: 追问单对象
    :param queue: SSE 队列（可能为 None）
    :param db: 数据库会话
    :param clarify_id: 追问单 ID（LangSmith 检索维度）
    :param resume_value: 透传给内层 Agent 的恢复载荷（answers / canceled）
    """
    return {
        "configurable": {
            "thread_id": clarify.conversation_id,
            "sse_queue": queue,
            "db": db,
            "trace_id": get_trace_id(),
            "clarify_resume": resume_value,
            "clarify_id": clarify_id,
        },
        "recursion_limit": settings.agent_graph_recursion_limit,
        "metadata": run_metadata(
            conversation_id=clarify.conversation_id,
            extra={"clarify_id": clarify_id, "biz_type": BIZ_TYPE_CLARIFY},
        ),
        "tags": run_tags(),
    }


async def resume_clarify(
    clarify_id: str,
    answers: list[str],
    db,
    *,
    answered_by: str = "",
) -> None:
    """
    前台提交批量答案后恢复图执行（幂等）。

    状态机守卫：仅当追问单（biz_type=clarification）为 pending 时抢占流转，
    重复提交被拦截。恢复后任务节点重入，ask_user 工具的 interrupt() 返回
    答案列表，Agent 依据补充信息继续处理，结果由节点经 SSE 推送。

    :param clarify_id: 追问单 ID
    :param answers: 用户批量回答（与 questions 一一对应）
    :param db: 数据库会话（后台任务独立会话）
    :param answered_by: 回答人（JWT username）
    """
    answers = [str(a).strip()[:_ANSWER_MAX_LENGTH] for a in (answers or []) if a is not None and str(a).strip()]
    if not answers:
        logger.error(f"[Clarify] 追问答案为空: clarify_id={clarify_id}")
        return

    # 1. 原子状态抢占（幂等守卫），并回填答案/回答人（审计）
    claimed = await _claim_clarify(db, clarify_id, answers, answered_by=answered_by)
    if not claimed:
        logger.info(f"[Clarify] 追问单 {clarify_id} 已被处理，忽略重复提交")
        return

    # 2. 加载追问单
    clarify = await _get_clarify(db, clarify_id)
    if clarify is None:
        return

    # 3. 推送回答回显（前端翻转卡片状态并展示答案；连接已断开则跳过）
    queue = approval_service.get_queue(clarify_id)
    if queue is not None:
        await put_status(
            queue,
            "clarification_answered",
            clarify_id=clarify_id,
            answers=answers,
            answered_by=answered_by,
        )

    # 4. 恢复同一 thread 的图执行（ask_user 工具的 interrupt() 返回答案列表）
    #    阶段 4 加固：补整体超时 + 递归上限（同审批链路，避免后台任务永久挂起）
    #    §20 F3.0：resume 值经 configurable.clarify_resume 透传给内层 Agent
    resume_value = {"answers": answers}
    config = _resume_config(clarify, queue, db, clarify_id, resume_value)
    graph = get_graph()
    try:
        result = await asyncio.wait_for(
            graph.ainvoke(Command(resume=resume_value), config),
            timeout=settings.agent_resume_timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.error(
            f"[Clarify] 追问恢复执行超时"
            f"（>{settings.agent_resume_timeout_seconds}s）: "
            f"clarify_id={clarify_id} trace_id={get_trace_id()}"
        )
        metrics.incr(K_CLARIFY_PREFIX.format("resume_timeout"))
        if queue is not None:
            await put_content(
                queue, "追问已受理，但恢复执行超时，请稍后查看会话结果", branch="task"
            )
        return
    except Exception as e:
        logger.exception(
            f"[Clarify] 追问恢复图执行失败: clarify_id={clarify_id} "
            f"trace_id={get_trace_id()}"
        )
        if queue is not None:
            await put_content(
                queue, f"追问已处理，但恢复执行失败：{str(e)[:120]}", branch="task"
            )
        return

    # ---- §20 F3.1（追问链路）：恢复后再次中断不得误报成功 ----
    # LangGraph 用返回值表达中断，`ainvoke` 仍正常返回；若内层 Agent 未消费到
    # 答案（透传失效）则会再次追问，此处必须显式识别并提示，避免静默假成功。
    result_dict = result if isinstance(result, dict) else {}
    if result_dict.get("__interrupt__"):
        logger.error(
            f"[Clarify] 恢复后再次中断，本次回答未生效: clarify_id={clarify_id}, "
            f"中断数={len(result_dict.get('__interrupt__') or [])}, "
            f"trace_id={get_trace_id()}"
        )
        metrics.incr(K_CLARIFY_PREFIX.format("resume_interrupted"))
        if queue is not None:
            await put_content(
                queue,
                "回答已提交，但执行未生效（恢复后再次触发追问），请重新发起该操作",
                branch="task",
            )
        return

    metrics.incr(K_CLARIFY_PREFIX.format("resumed"))
    logger.info(
        f"[Clarify] 追问恢复完成: clarify_id={clarify_id}, "
        f"answered_by={answered_by}, trace_id={get_trace_id()}"
    )


async def cancel_clarify(
    clarify_id: str,
    db,
    *,
    canceled_by: str = "",
) -> None:
    """
    前台取消追问后恢复图执行并终止任务（幂等）。

    状态机守卫：仅当追问单（biz_type=clarification）为 pending 时抢占流转，
    重复取消被拦截。恢复后 ask_user 工具的 interrupt() 返回 {"canceled": True}，
    LLM 据此如实告知用户"已取消提供相关信息"并给出修复建议；任务执行记录
    标记为 canceled（终止态）。

    :param clarify_id: 追问单 ID
    :param db: 数据库会话（后台任务独立会话）
    :param canceled_by: 取消人（JWT username）
    """
    # 1. 原子状态抢占（幂等守卫）：pending -> canceled
    claimed = await _claim_clarify_canceled(db, clarify_id, canceled_by=canceled_by)
    if not claimed:
        logger.info(f"[Clarify] 追问单 {clarify_id} 已被处理，忽略重复取消")
        return

    # 2. 加载追问单
    clarify = await _get_clarify(db, clarify_id)
    if clarify is None:
        return

    # 3. 推送取消回显（前端翻转卡片为已取消；连接已断开则跳过）
    queue = approval_service.get_queue(clarify_id)
    if queue is not None:
        await put_status(
            queue,
            "clarification_canceled",
            clarify_id=clarify_id,
            canceled_by=canceled_by,
        )

    # 4. 恢复同一 thread 的图执行（ask_user 工具收到取消信号）
    #    §20 F3.0：resume 值经 configurable.clarify_resume 透传给内层 Agent
    resume_value = {"canceled": True}
    config = _resume_config(clarify, queue, db, clarify_id, resume_value)
    graph = get_graph()
    try:
        result = await asyncio.wait_for(
            graph.ainvoke(Command(resume=resume_value), config),
            timeout=settings.agent_resume_timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.error(
            f"[Clarify] 追问取消恢复执行超时"
            f"（>{settings.agent_resume_timeout_seconds}s）: "
            f"clarify_id={clarify_id} trace_id={get_trace_id()}"
        )
        metrics.incr(K_CLARIFY_PREFIX.format("resume_timeout"))
        if queue is not None:
            await put_content(
                queue, "追问已取消，但恢复执行超时，请稍后查看会话结果", branch="task"
            )
        return
    except Exception as e:
        logger.exception(
            f"[Clarify] 追问取消恢复图执行失败: clarify_id={clarify_id} "
            f"trace_id={get_trace_id()}"
        )
        if queue is not None:
            await put_content(
                queue, f"追问已取消，但恢复执行失败：{str(e)[:120]}", branch="task"
            )
        return

    # ---- §20 F3.1（追问链路）：取消后再次中断不得误报成功 ----
    result_dict = result if isinstance(result, dict) else {}
    if result_dict.get("__interrupt__"):
        logger.error(
            f"[Clarify] 取消后再次中断，取消信号未生效: clarify_id={clarify_id}, "
            f"trace_id={get_trace_id()}"
        )
        metrics.incr(K_CLARIFY_PREFIX.format("resume_interrupted"))
        if queue is not None:
            await put_content(
                queue,
                "取消已提交，但执行未生效（恢复后再次触发追问），请重新发起该操作",
                branch="task",
            )
        # 即便二次中断，用户取消意图明确 → 任务仍标记 canceled（终止态）
        await _mark_task_canceled(db, clarify)
        return

    # 5. 任务终止标记（用户主动取消追问语义）
    await _mark_task_canceled(db, clarify)

    metrics.incr(K_CLARIFY_PREFIX.format("canceled_resumed"))
    logger.info(
        f"[Clarify] 追问取消完成: clarify_id={clarify_id}, "
        f"canceled_by={canceled_by}, trace_id={get_trace_id()}"
    )


async def _resume_in_background(
    clarify_id: str,
    answers: list[str],
    answered_by: str,
) -> None:
    """后台恢复图执行（独立数据库会话，避免与请求会话生命周期冲突）。"""
    try:
        async with AsyncSessionLocal() as bg_db:
            await resume_clarify(clarify_id, answers, bg_db, answered_by=answered_by)
    except Exception:
        logger.exception(f"[Clarify] 后台恢复失败: clarify_id={clarify_id}")
    finally:
        # 通知 SSE 等待方：恢复完成（含最终结果推送）
        approval_service.set_completed(clarify_id)


async def _cancel_in_background(
    clarify_id: str,
    canceled_by: str,
) -> None:
    """后台取消追问（独立数据库会话，避免与请求会话生命周期冲突）。"""
    try:
        async with AsyncSessionLocal() as bg_db:
            await cancel_clarify(clarify_id, bg_db, canceled_by=canceled_by)
    except Exception:
        logger.exception(f"[Clarify] 后台取消失败: clarify_id={clarify_id}")
    finally:
        approval_service.set_completed(clarify_id)


def submit_answer(clarify_id: str, answers: list[str], answered_by: str = "") -> None:
    """
    受理前台提交的追问答案：以后台任务恢复图执行（幂等由状态机守卫）。

    立即返回，恢复结果经原 SSE 连接推送（断连时前端可轮询兜底）。
    """
    approval_service.spawn_background(
        _resume_in_background(clarify_id, answers, answered_by)
    )


def submit_cancel(clarify_id: str, canceled_by: str = "") -> None:
    """
    受理前台取消追问：以后台任务恢复图执行并终止任务（幂等由状态机守卫）。

    立即返回，取消说明与修复建议经原 SSE 连接推送。
    """
    approval_service.spawn_background(_cancel_in_background(clarify_id, canceled_by))


# =============================================================================
# 会话追问列表（刷新后重建追问卡片）
# =============================================================================

async def list_conversation_clarifies(
    db,
    conversation_id: str,
    requester_id: str,
    limit: int = 50,
) -> list[ApprovalRequest]:
    """按会话查询追问单列表（仅发起人本人，按创建时间倒序）。"""
    result = await db.execute(
        select(ApprovalRequest)
        .where(
            ApprovalRequest.conversation_id == conversation_id,
            ApprovalRequest.biz_type == BIZ_TYPE_CLARIFY,
            ApprovalRequest.requester_id == requester_id,
        )
        .order_by(ApprovalRequest.created_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


# =============================================================================
# 数据访问辅助
# =============================================================================

async def _find_running_task(db, conversation_id: str) -> Optional[TaskExecution]:
    """
    查找会话最近的 running 状态任务记录（追问单关联复用）。

    阶段 4 加固：同审批链路，DB 抖动时回滚并返回 None，避免整段追问单落库失败。
    """
    try:
        result = await db.execute(
            select(TaskExecution)
            .where(
                TaskExecution.conversation_id == conversation_id,
                TaskExecution.status == "running",
            )
            .order_by(TaskExecution.created_at.desc())
            .limit(1)
        )
        return result.scalars().first()
    except Exception as e:
        try:
            await db.rollback()
        except Exception:
            pass
        logger.warning(f"[Clarify] 查询任务记录失败（非致命）: {e}")
        return None


async def _claim_clarify(
    db,
    clarify_id: str,
    answers: list[str],
    *,
    answered_by: str = "",
) -> bool:
    """
    原子抢占追问单：pending（biz_type=clarification）-> answered，成功返回 True。

    批量回答存 callback_payload={"answers": [...]}；answer 列保留首个回答
    （仅兼容旧单问题展示，列表数据以 callback_payload.answers 为准）。
    """
    result = await db.execute(
        update(ApprovalRequest)
        .where(
            ApprovalRequest.id == clarify_id,
            ApprovalRequest.biz_type == BIZ_TYPE_CLARIFY,
            ApprovalRequest.status == "pending",
        )
        .values(
            status="answered",
            answer=answers[0][:_ANSWER_MAX_LENGTH],
            callback_payload={"answers": answers},
            approved_by=answered_by or None,
        )
    )
    await db.commit()
    return result.rowcount > 0


async def _claim_clarify_canceled(
    db,
    clarify_id: str,
    *,
    canceled_by: str = "",
) -> bool:
    """
    原子抢占追问单：pending（biz_type=clarification）-> canceled，成功返回 True。

    取消标记存 callback_payload={"canceled": True}，供卡片载荷与恢复值识别。
    """
    result = await db.execute(
        update(ApprovalRequest)
        .where(
            ApprovalRequest.id == clarify_id,
            ApprovalRequest.biz_type == BIZ_TYPE_CLARIFY,
            ApprovalRequest.status == "pending",
        )
        .values(
            status="canceled",
            callback_payload={"canceled": True},
            approved_by=canceled_by or None,
        )
    )
    await db.commit()
    return result.rowcount > 0


async def _get_task(db, task_execution_id: Optional[str]) -> Optional[TaskExecution]:
    if not task_execution_id:
        return None
    result = await db.execute(
        select(TaskExecution).where(TaskExecution.id == task_execution_id)
    )
    return result.scalars().first()


async def _mark_task_canceled(db, clarify: ApprovalRequest) -> None:
    """用户取消追问：将关联任务执行记录标记为 canceled（终止态）。"""
    task = await _get_task(db, clarify.task_execution_id)
    if task is None:
        return
    task.status = "canceled"
    await db.commit()


async def _get_clarify(db, clarify_id: str) -> Optional[ApprovalRequest]:
    """按 ID 查询追问单（限定 biz_type=clarification，避免误取审批单）。"""
    result = await db.execute(
        select(ApprovalRequest).where(
            ApprovalRequest.id == clarify_id,
            ApprovalRequest.biz_type == BIZ_TYPE_CLARIFY,
        )
    )
    return result.scalars().first()
