"""
审批单相关 Pydantic Schema

用于前台审批决策提交、审批单状态查询与会话审批列表
"""
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class ApprovalStatus(str, Enum):
    """审批单状态"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    CANCELED = "canceled"


class ApprovalDecision(str, Enum):
    """前台审批决策（仅允许通过与拒绝，取消由服务端状态机处理）"""
    APPROVED = "approved"
    REJECTED = "rejected"


class ApprovalDecisionIn(BaseModel):
    """前台审批决策请求体"""
    decision: ApprovalDecision = Field(..., description="审批决策: approved/rejected")
    reason: Optional[str] = Field(default=None, max_length=255, description="审批意见（可选）")


class ApprovalOut(BaseModel):
    """审批单查询响应（含审批卡片渲染所需字段）"""
    model_config = ConfigDict(from_attributes=True)

    id: str
    task_execution_id: Optional[str] = None
    conversation_id: Optional[str] = None
    requester_id: Optional[str] = None
    approval_type: Optional[str] = None
    target_table: Optional[str] = None
    tool_params: Optional[dict] = None
    actions: Optional[list] = None
    status: str
    approved_by: Optional[str] = None
    decision_reason: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
