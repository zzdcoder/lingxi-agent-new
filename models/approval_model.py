import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import String, DateTime, JSON, Index, func
from sqlalchemy.orm import Mapped, mapped_column

from core import Base


def _generate_uuid() -> str:
    return str(uuid.uuid4())


class ApprovalRequest(Base):
    """审批单表 - 存储 Agent 数据库写操作的人工审批记录（前台审批，设计文档 §8.2）

    - tool_params：首个待执行工具参数（脱敏存储，供审批卡片头部渲染）；
    - actions：一次中断的全部待审批操作列表（多 action 场景，含各自
      approval_type / target_table / tool_params，均已脱敏；
      单 action 场景为单元素列表，供决策数量与卡片明细渲染）；
    - decision_reason：前台审批意见（可选）；
    - callback_payload：审计载荷（前台模式存决策请求摘要）；
    - feishu_instance_code：历史遗留列（飞书审批已下线，保留避免存量库迁移，不再写入）。
    """
    __tablename__ = "approval_request"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_generate_uuid)
    task_execution_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, comment="关联任务执行ID")
    conversation_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, comment="关联会话ID（=thread_id）")
    requester_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, comment="发起人（系统用户名，JWT username）")
    approval_type: Mapped[Optional[str]] = mapped_column(String(32), nullable=True, comment="审批类型: update/delete/insert（首个 action）")
    target_table: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, comment="目标表（首个 action）")
    tool_params: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True, comment="首个待执行工具参数（脱敏，供审批卡片渲染）")
    actions: Mapped[Optional[list]] = mapped_column(JSON, nullable=True, comment="全部待审批操作列表（脱敏，供决策数量与明细渲染）")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending", comment="状态: pending/approved/rejected/canceled")
    feishu_instance_code: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, comment="已废弃（飞书审批下线，保留列避免迁移）")
    approved_by: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, comment="审批人（前台提交决策的用户名）")
    decision_reason: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, comment="审批意见（前台可选填写）")
    callback_payload: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True, comment="审计载荷（前台模式存决策请求摘要）")
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=func.current_timestamp(), comment="创建时间")
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=func.current_timestamp(), onupdate=func.current_timestamp(), comment="更新时间")

    __table_args__ = (
        Index("idx_appr_instance", "feishu_instance_code"),
        Index("idx_appr_conv", "conversation_id", "status"),
    )
