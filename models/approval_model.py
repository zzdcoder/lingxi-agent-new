import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import String, DateTime, Text, JSON, Index, func
from sqlalchemy.orm import Mapped, mapped_column

from core import Base


def _generate_uuid() -> str:
    return str(uuid.uuid4())


class ApprovalRequest(Base):
    """审批单表 - 存储 Agent 数据库写操作的人工审批记录

    字段对齐设计文档 §8.2：
    - tool_params：待执行工具参数（恢复时使用，脱敏存储）；
    - callback_payload：飞书回调原文（审计）。
    """
    __tablename__ = "approval_request"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_generate_uuid)
    task_execution_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, comment="关联任务执行ID")
    conversation_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, comment="关联会话ID（=thread_id）")
    requester_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, comment="发起人（飞书 user_id/open_id）")
    approval_type: Mapped[Optional[str]] = mapped_column(String(32), nullable=True, comment="审批类型: update/delete/insert")
    target_table: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, comment="目标表")
    tool_params: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True, comment="待执行工具参数（脱敏）")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending", comment="状态: pending/approved/rejected/canceled")
    feishu_instance_code: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, comment="飞书审批实例编码")
    approved_by: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, comment="审批人")
    callback_payload: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True, comment="飞书回调原文（审计）")
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=func.current_timestamp(), comment="创建时间")
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=func.current_timestamp(), onupdate=func.current_timestamp(), comment="更新时间")

    __table_args__ = (
        Index("idx_appr_instance", "feishu_instance_code"),
        Index("idx_appr_conv", "conversation_id", "status"),
    )
