"""
审批单相关 Pydantic Schema

用于审批单状态查询与回调处理
"""
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, ConfigDict


class ApprovalStatus(str, Enum):
    """审批单状态"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    CANCELED = "canceled"


class ApprovalOut(BaseModel):
    """审批单查询响应"""
    model_config = ConfigDict(from_attributes=True)

    id: str
    task_execution_id: Optional[str] = None
    conversation_id: Optional[str] = None
    approval_type: Optional[str] = None
    target_table: Optional[str] = None
    status: str
    approved_by: Optional[str] = None
    feishu_instance_code: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
