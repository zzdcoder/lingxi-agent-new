"""
审批路由

- POST /api/approval/{approval_id}/decision：前台审批决策（受理后后台恢复图执行）
- GET  /api/approval?conversation_id=xxx：按会话查询审批单列表（刷新后重建审批卡片）
- GET  /api/approval/{approval_id}：查询审批单状态（SSE 断连轮询兜底）
"""
import logging

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.exceptions import ConversationException
from models.approval_model import ApprovalRequest
from models.approval_schema import ApprovalDecisionIn, ApprovalOut
from models.user_model import User
from utils.auth import get_current_user
from agent.approval import approval_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/approval", tags=["Approval"])


# ==========================================
# 前台审批决策（受理后后台恢复图执行）
# ==========================================

@router.post("/{approval_id}/decision")
async def submit_decision(
    approval_id: str,
    body: ApprovalDecisionIn,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    前台审批决策接口（设计文档 §7.3）。

    1. 越权校验：仅发起人本人可审批（requester_id 比对，越权 403）；
    2. 原子抢占与图恢复由 approval_service.submit_decision 幂等完成
       （后台任务使用独立数据库会话）；
    3. 立即返回受理结果，决策回显与执行结果经原 SSE 连接推送
       （断连时前端可轮询兜底）。

    :param approval_id: 审批单 ID
    :param body: 决策请求体（approved/rejected + 可选意见）
    :param db: 数据库会话
    :param current_user: 当前用户（审批人）
    """
    try:
        result = await db.execute(
            select(ApprovalRequest).where(ApprovalRequest.id == approval_id)
        )
        approval = result.scalars().first()
        if not approval:
            raise ConversationException(f"审批单不存在: {approval_id}")
        if approval.requester_id and approval.requester_id != current_user.username:
            raise ConversationException("仅发起人本人可审批该操作", code=403)

        approval_service.submit_decision(
            approval_id,
            body.decision.value,
            approved_by=current_user.username,
            reason=body.reason or "",
        )
        logger.info(
            f"[Approval] 前台审批决策已受理: approval_id={approval_id}, "
            f"decision={body.decision.value}, approved_by={current_user.username}"
        )
        return {
            "code": 0,
            "approval_id": approval_id,
            "status": body.decision.value,
            "message": "审批已受理，结果将经会话流式推送",
        }
    except ConversationException:
        raise
    except Exception as e:
        logger.error(f"提交审批决策失败: {e}")
        raise ConversationException(f"提交审批决策失败: {str(e)}")


# ==========================================
# 会话审批列表（刷新后重建审批卡片）
# ==========================================

@router.get("", response_model=list[ApprovalOut])
async def list_conversation_approvals(
    conversation_id: str = Query(..., description="会话ID"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    按会话查询审批单列表（仅返回发起人本人的审批单，前端据此重建审批卡片：
    pending 渲染可交互卡片，终态渲染只读回显卡片）。

    :param conversation_id: 会话 ID
    :param db: 数据库会话
    :param current_user: 当前用户
    :return: 审批单列表（按创建时间倒序）
    """
    try:
        result = await db.execute(
            select(ApprovalRequest)
            .where(
                ApprovalRequest.conversation_id == conversation_id,
                ApprovalRequest.requester_id == current_user.username,
            )
            .order_by(ApprovalRequest.created_at.desc())
            .limit(50)
        )
        return result.scalars().all()
    except Exception as e:
        logger.error(f"查询会话审批列表失败: {e}")
        raise ConversationException(f"查询会话审批列表失败: {str(e)}")


# ==========================================
# 审批单状态查询（SSE 断连轮询兜底）
# ==========================================

@router.get("/{approval_id}", response_model=ApprovalOut)
async def get_approval_status(
    approval_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    查询审批单状态（前端轮询兜底：SSE 断连后可经此接口获取最新状态）。

    :param approval_id: 审批单 ID
    :param db: 数据库会话
    :param current_user: 当前用户
    :return: 审批单信息
    """
    try:
        result = await db.execute(
            select(ApprovalRequest).where(ApprovalRequest.id == approval_id)
        )
        approval = result.scalars().first()
        if not approval:
            raise ConversationException(f"审批单不存在: {approval_id}")

        # 越权校验：仅发起人可见
        if approval.requester_id and approval.requester_id != current_user.username:
            raise ConversationException("无权查看该审批单", code=403)

        return approval
    except ConversationException:
        raise
    except Exception as e:
        logger.error(f"查询审批单失败: {e}")
        raise ConversationException(f"查询审批单失败: {str(e)}")
