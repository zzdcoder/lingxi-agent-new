"""
审批路由

- POST /api/approval/callback：飞书事件回调（无 JWT，由飞书调用，验签+解密）
- GET  /api/approval/{approval_id}：查询审批单状态（JWT，仅发起人可见）
"""
import logging

from fastapi import APIRouter, Depends, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.exceptions import ConversationException
from models.approval_model import ApprovalRequest
from models.approval_schema import ApprovalOut
from models.user_model import User
from utils.auth import get_current_user
from agent.approval.callback import verify_and_decrypt, handle_approval_event

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/approval", tags=["Approval"])


# ==========================================
# 飞书事件回调（验签 + 解密 + 恢复图执行）
# ==========================================

@router.post("/callback")
async def approval_callback(request: Request):
    """
    飞书事件回调接口。

    由飞书开放平台调用，携带 verification_token 与 encrypt 载荷；
    验签解密后解析审批状态（APPROVED/REJECTED/CANCELED）并触发后台图恢复。
    """
    payload = await request.json()
    event = verify_and_decrypt(payload)
    if event is None:
        # 验签失败：静默返回，避免暴露内部信息
        return {"code": 1}

    # 订阅配置阶段：应答 challenge
    if "challenge" in event:
        return {"challenge": event["challenge"]}

    # 处理审批事件（幂等由审批单状态机守卫）
    await handle_approval_event(event)
    return {"code": 0}


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
