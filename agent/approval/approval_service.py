"""
审批服务（执行包装层）

职责（设计文档 §5.6、§7.2、§7.3）：
1. 解析图执行中断（HITLRequest），落库审批单（approval_request, status=pending）；
2. 构造审批卡片载荷（SSE approval_required 事件携带，前台据此渲染审批卡片）；
3. 前台审批决策（POST /api/approval/{id}/decision）受理后，
   以 Command(resume=...) 恢复同一 thread 的图执行；
4. SSE 队列注册表：审批等待期连接存活时，后台恢复任务可向原连接推送结果；
5. 幂等守卫：审批单状态单向流转（pending→approved/rejected/canceled），
   重复提交决策 / 重复恢复被状态机拦截。

多 action 支持（设计文档 §5.6）：
- 一次 HITL 中断可能包含多个待审批写操作（模型平行 tool calling）；
- 单次中断落一张审批单，actions 字段保存全部操作（脱敏），
  approval_type / target_table / tool_params 保存首个 action（卡片头部展示）；
- 决策为批次级：恢复时 decisions 数量与 actions 数量一一对应，
  与 LangChain HITL 中间件的数量校验（ValueError）保持一致；
- 存量单（无 actions）按单元素处理，兼容旧逻辑。
"""
import asyncio
import logging
import time
from typing import Optional

from langgraph.types import Command
from sqlalchemy import select, update

from core.config import settings
from core.database import AsyncSessionLocal
from models.approval_model import ApprovalRequest
from models.task_model import TaskExecution
from agent.streaming import put_content, put_ping, put_status
from agent.graph_builder import get_graph
from agent.observability import (
    K_APPROVAL_PREFIX,
    get_trace_id,
    metrics,
    run_metadata,
    run_tags,
)

logger = logging.getLogger(__name__)

# 审批等待心跳间隔（秒）：防网关/代理超时
HEARTBEAT_INTERVAL = 15
# 审批等待超时（秒）：超过后 SSE 结束等待，客户端可经轮询接口获取结果
APPROVAL_WAIT_TIMEOUT = settings.approval_wait_timeout

# 工具名 -> 审批类型（approval_type 字段）
_ACTION_TYPE_MAP = {
    "update_data": "update",
    "delete_data": "delete",
    "insert_data": "insert",
}

# 审批单 ID -> SSE 队列（供后台恢复任务推送结果）
_pending_queues: dict[str, asyncio.Queue] = {}
# 审批单 ID -> 完成事件（后台恢复完成后触发，路由据此结束心跳等待）
_pending_events: dict[str, asyncio.Event] = {}
# 后台任务强引用（防止被垃圾回收）
_background_tasks: set = set()


# =============================================================================
# SSE 队列注册表
# =============================================================================

def register_queue(approval_id: str, queue: asyncio.Queue) -> None:
    """注册审批单对应的 SSE 队列与完成事件。"""
    _pending_queues[approval_id] = queue
    _pending_events[approval_id] = asyncio.Event()


def get_queue(approval_id: str) -> Optional[asyncio.Queue]:
    """获取审批单对应的 SSE 队列（连接已断开时返回 None）。"""
    return _pending_queues.get(approval_id)


def unregister_queue(approval_id: str) -> None:
    """注销审批单对应的 SSE 队列与完成事件。"""
    _pending_queues.pop(approval_id, None)
    _pending_events.pop(approval_id, None)


def set_completed(approval_id: str) -> None:
    """标记审批恢复完成（后台任务调用；未注册时无副作用）。"""
    event = _pending_events.get(approval_id)
    if event is not None:
        event.set()


def spawn_background(coro) -> None:
    """启动后台协程并持有强引用（防止被 GC 中断）。"""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


# =============================================================================
# 中断解析与审批单落库
# =============================================================================

def _mask_params(args: Optional[dict]) -> dict:
    """对工具参数脱敏：保留结构，值截断、敏感字段打码（设计文档 §11）。"""
    sensitive_keys = ("password", "secret", "token", "key", "salt", "captcha_text")

    def _mask(key: str, val):
        if isinstance(val, dict):
            return {k: _mask(k, v) for k, v in val.items()}
        if isinstance(val, list):
            return f"<{len(val)} 项>"
        text = str(val)
        if key.lower() in sensitive_keys:
            return "******"
        return text if len(text) <= 32 else text[:16] + "..." + text[-8:]

    return {k: _mask(k, v) for k, v in (args or {}).items()}


async def save_pending_approval(
    db,
    *,
    conversation_id: str,
    requester_id: str,
    action_requests: list[dict],
) -> ApprovalRequest:
    """
    解析 HITL 中断载荷并落库审批单（pending）。

    单次中断可能含多个待审批写操作（模型平行 tool calling）：全部操作
    脱敏后存入 actions 字段，approval_type / target_table / tool_params
    保存首个 action（供卡片头部渲染）；决策数量与 actions 一一对应。

    :param db: 数据库会话
    :param conversation_id: 会话 ID（= thread_id）
    :param requester_id: 发起人标识
    :param action_requests: HITLRequest.action_requests
    :return: 审批单对象（供卡片载荷构造）
    """
    if not action_requests:
        raise ValueError("中断载荷缺少 action_requests")
    if len(action_requests) > 1:
        logger.info(
            f"单次中断包含 {len(action_requests)} 个待审批操作，"
            f"合并为一张审批单（批次级决策）"
        )

    # 全部操作脱敏归一化：[{approval_type, target_table, tool_params}]
    actions = [
        {
            "approval_type": _ACTION_TYPE_MAP.get(
                act.get("name", ""), act.get("name", "")
            ),
            "target_table": str((act.get("args") or {}).get("table", ""))[:64],
            "tool_params": _mask_params(act.get("args") or {}),
        }
        for act in action_requests
    ]
    first = actions[0]

    # 审批单关联任务记录（中断前任务节点已创建 status=running 的记录）
    task = await _find_running_task(db, conversation_id)

    approval = ApprovalRequest(
        task_execution_id=task.id if task else None,
        conversation_id=conversation_id,
        requester_id=requester_id,
        approval_type=first["approval_type"],
        target_table=first["target_table"],
        tool_params=first["tool_params"],
        actions=actions,
        status="pending",
    )
    db.add(approval)
    await db.commit()
    await db.refresh(approval)
    metrics.incr(K_APPROVAL_PREFIX.format("created"))
    logger.info(
        f"[Approval] 审批单创建: id={approval.id}, "
        f"type={approval.approval_type}, table={approval.target_table}, "
        f"actions={len(actions)}, trace_id={get_trace_id()}"
    )
    return approval


# =============================================================================
# 审批卡片载荷（前台可视化，设计文档 §7.5）
# =============================================================================

def build_card_payload(approval: ApprovalRequest) -> dict:
    """
    构造前台审批卡片载荷（SSE approval_required 事件携带）。

    tool_params / actions 已在落库时脱敏（_mask_params）；datetime 转字符串，
    避免 SSE JSON 编码失败。
    """
    created_at = approval.created_at
    return {
        "approval_type": approval.approval_type,
        "target_table": approval.target_table,
        "tool_params": approval.tool_params or {},
        "actions": approval.actions or [],
        "requester_id": approval.requester_id,
        "created_at": created_at.strftime("%Y-%m-%d %H:%M:%S") if created_at else None,
        "status": approval.status,
    }


# =============================================================================
# 审批恢复
# =============================================================================

def _build_resume_value(decision: str, reason: str = "", count: int = 1) -> dict:
    """
    构造 HITL 恢复载荷。

    LangChain HITL 中间件校验 decisions 数量与中断挂起工具数必须一致
    （不一致抛 ValueError）；单次中断可能含多个写操作，故按 count 批量生成。
    """
    if count < 1:
        count = 1
    if decision == "approved":
        return {"decisions": [{"type": "approve"} for _ in range(count)]}
    if decision == "rejected":
        message = reason or "审批人拒绝执行该写操作"
        return {"decisions": [{"type": "reject", "message": message} for _ in range(count)]}
    # canceled 视为拒绝：写操作不执行
    return {
        "decisions": [
            {"type": "reject", "message": "审批已取消，写操作未执行"}
            for _ in range(count)
        ]
    }


async def resume_graph(
    approval_id: str,
    decision: str,
    db,
    *,
    approved_by: str = "",
    reason: str = "",
) -> None:
    """
    前台审批决策受理后恢复图执行（幂等）。

    状态机守卫：仅当审批单为 pending 时抢占并流转，重复提交被拦截。

    :param approval_id: 审批单 ID
    :param decision: approved / rejected / canceled
    :param db: 数据库会话（后台任务独立会话）
    :param approved_by: 审批人（JWT username）
    :param reason: 审批意见（可选）
    """
    if decision not in ("approved", "rejected", "canceled"):
        logger.error(f"[Approval] 非法审批决策: {decision}")
        return

    # 1. 原子状态抢占（幂等守卫），并回填审批人/意见（审计）
    claimed = await _claim_approval(
        db, approval_id, decision, approved_by=approved_by, reason=reason
    )
    if not claimed:
        logger.info(f"[Approval] 审批单 {approval_id} 已被处理，忽略重复提交")
        return

    # 2. 加载审批单
    approval = await _get_approval(db, approval_id)
    if approval is None:
        return

    # 挂起工具数（多 action 批次决策）；存量单无 actions 按 1 个处理
    action_count = len(approval.actions) if approval.actions else 1

    # 3. 推送审批决策回显（前端翻转卡片状态；连接已断开则跳过）
    queue = get_queue(approval_id)
    if queue is not None:
        await put_status(
            queue,
            "approval_decided",
            approval_id=approval_id,
            decision=decision,
            approved_by=approved_by,
        )

    # 4. 恢复同一 thread 的图执行
    #    阶段 4 加固：恢复路径此前**无任何超时保护**（落库/LLM 挂死会永久挂起后台任务，
    #    前端只能等审批等待超时），此处补整体超时 + 主图递归上限。
    config = {
        "configurable": {
            "thread_id": approval.conversation_id,
            "sse_queue": queue,
            "db": db,
            "trace_id": get_trace_id(),
        },
        "recursion_limit": settings.agent_graph_recursion_limit,
        "metadata": run_metadata(
            conversation_id=approval.conversation_id,
            extra={"approval_id": approval_id, "biz_type": "write_approval"},
        ),
        "tags": run_tags(),
    }
    graph = get_graph()
    try:
        result = await asyncio.wait_for(
            graph.ainvoke(
                Command(resume=_build_resume_value(decision, reason, action_count)),
                config,
            ),
            timeout=settings.agent_resume_timeout_seconds,
        )
    except asyncio.TimeoutError:
        error = f"审批恢复执行超时（>{settings.agent_resume_timeout_seconds}s）"
        logger.error(f"[Approval] {error}: approval_id={approval_id} trace_id={get_trace_id()}")
        metrics.incr(K_APPROVAL_PREFIX.format("resume_timeout"))
        await _mark_task_error(db, approval, error)
        if queue is not None:
            await put_content(queue, "审批已处理，但恢复执行超时，请稍后查看任务结果")
        return
    except Exception as e:
        logger.exception(f"[Approval] 审批恢复图执行失败: approval_id={approval_id}")
        await _mark_task_error(db, approval, f"审批恢复执行失败：{str(e)[:120]}")
        if queue is not None:
            await put_content(queue, f"审批已处理，但恢复执行失败：{str(e)[:120]}")
        return

    result_dict = result if isinstance(result, dict) else {}
    final_response = result_dict.get("final_response")
    task_answer = result_dict.get("task_answer")

    # 4. 更新任务状态与最终结果
    await _update_task_after_resume(db, approval, decision, final_response)

    # 5. 推送最终结果（去重：与任务分支结果相同时任务节点已推送）
    if final_response and final_response != task_answer:
        if queue is not None:
            await put_content(queue, final_response)
    elif not final_response and task_answer:
        if queue is not None:
            await put_content(queue, task_answer)

    metrics.incr(K_APPROVAL_PREFIX.format(decision))
    logger.info(
        f"[Approval] 审批恢复完成: approval_id={approval_id}, decision={decision}, "
        f"trace_id={get_trace_id()}"
    )


async def _resume_in_background(
    approval_id: str,
    decision: str,
    approved_by: str,
    reason: str,
) -> None:
    """后台恢复图执行（独立数据库会话，避免与请求会话生命周期冲突）。"""
    try:
        async with AsyncSessionLocal() as bg_db:
            await resume_graph(
                approval_id, decision, bg_db, approved_by=approved_by, reason=reason
            )
    except Exception:
        logger.exception(f"[Approval] 后台恢复失败: approval_id={approval_id}")
    finally:
        # 通知 SSE 等待方：恢复完成（含最终结果推送）
        set_completed(approval_id)


def submit_decision(
    approval_id: str,
    decision: str,
    approved_by: str,
    reason: str = "",
) -> None:
    """
    受理前台审批决策：以后台任务恢复图执行（幂等由审批单状态机守卫）。

    立即返回，恢复结果经原 SSE 连接推送（断连时前端可轮询兜底）。
    """
    spawn_background(_resume_in_background(approval_id, decision, approved_by, reason))


# =============================================================================
# 等待与轮询
# =============================================================================

async def wait_for_final(approval_id: str, queue) -> None:
    """
    等待审批恢复完成（事件驱动 + 心跳保活）。

    后台恢复任务完成（含最终结果推送）后触发完成事件，路由据此结束等待；
    超时（APPROVAL_WAIT_TIMEOUT）返回，客户端可经 GET /api/agent/tasks/{id}
    或 GET /api/approval/{id} 轮询兜底。
    """
    event = _pending_events.get(approval_id)
    if event is None:
        return
    deadline = time.monotonic() + APPROVAL_WAIT_TIMEOUT
    while True:
        if event.is_set():
            return
        if time.monotonic() > deadline:
            logger.warning(f"[Approval] 审批等待超时: approval_id={approval_id}")
            return
        await put_ping(queue)
        try:
            await asyncio.wait_for(event.wait(), timeout=HEARTBEAT_INTERVAL)
        except asyncio.TimeoutError:
            continue


# =============================================================================
# 数据访问辅助
# =============================================================================

async def _find_running_task(db, conversation_id: str) -> Optional[TaskExecution]:
    """
    查找会话最近的 running 状态任务记录。

    阶段 4 加固：原实现无异常兜底，DB 抖动会让「落审批单/写审计」整段失败
    （审批单创建失败 = 写操作无审批凭证）。这里改为失败返回 None 并回滚会话，
    保证审批单仍能创建（仅丢失任务关联），不影响审批主链路。
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
        logger.warning(f"[Approval] 查询任务记录失败（非致命）: {e}")
        return None


async def _get_approval(db, approval_id: str) -> Optional[ApprovalRequest]:
    result = await db.execute(
        select(ApprovalRequest).where(ApprovalRequest.id == approval_id)
    )
    return result.scalars().first()


async def _get_task(db, task_execution_id: Optional[str]) -> Optional[TaskExecution]:
    if not task_execution_id:
        return None
    result = await db.execute(
        select(TaskExecution).where(TaskExecution.id == task_execution_id)
    )
    return result.scalars().first()


async def _claim_approval(
    db,
    approval_id: str,
    decision: str,
    *,
    approved_by: str = "",
    reason: str = "",
) -> bool:
    """原子抢占审批单：仅 pending -> 目标状态并回填审计字段，成功返回 True。"""
    target = decision if decision in ("approved", "rejected", "canceled") else "rejected"
    result = await db.execute(
        update(ApprovalRequest)
        .where(
            ApprovalRequest.id == approval_id,
            ApprovalRequest.status == "pending",
        )
        .values(
            status=target,
            approved_by=approved_by or None,
            decision_reason=reason or None,
        )
    )
    await db.commit()
    return result.rowcount > 0


async def _update_task_after_resume(
    db,
    approval: ApprovalRequest,
    decision: str,
    final_response: Optional[str],
) -> None:
    """审批恢复后更新任务记录状态与最终结果。"""
    task = await _get_task(db, approval.task_execution_id)
    if task is None:
        return
    if decision == "rejected":
        task.status = "rejected"
    elif decision == "canceled":
        task.status = "canceled"
    if final_response:
        task.result = final_response
    await db.commit()


async def _mark_task_error(
    db,
    approval: ApprovalRequest,
    error: str,
) -> None:
    """审批恢复失败时标记任务失败。"""
    task = await _get_task(db, approval.task_execution_id)
    if task is None:
        return
    task.status = "failed"
    task.error = error[:500]
    await db.commit()
