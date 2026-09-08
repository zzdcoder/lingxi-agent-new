import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import String, DateTime, Text, Integer, Index, func, JSON
from sqlalchemy.orm import Mapped, mapped_column

from core import Base


def _generate_uuid() -> str:
    return str(uuid.uuid4())


class TaskExecution(Base):
    """任务执行记录表 - 存储 Agent 任务执行的完整审计信息"""
    __tablename__ = "task_execution"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_generate_uuid)
    conversation_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, comment="关联会话ID")
    user_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, comment="发起用户ID")
    intent: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, comment="本次路由意图")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="running", comment="状态: running/completed/rejected/failed/canceled")
    tool_calls: Mapped[Optional[list]] = mapped_column(JSON, nullable=True, comment="工具调用序列（审计）")
    rag_answer: Mapped[Optional[str]] = mapped_column(Text, nullable=True, comment="知识库分支回答")
    task_answer: Mapped[Optional[str]] = mapped_column(Text, nullable=True, comment="任务分支回答")
    result: Mapped[Optional[str]] = mapped_column(Text, nullable=True, comment="最终合并结果")
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True, comment="错误信息")
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=func.current_timestamp(), comment="创建时间")
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=func.current_timestamp(), onupdate=func.current_timestamp(), comment="更新时间")

    __table_args__ = (
        Index("idx_task_conv_id", "conversation_id"),
        Index("idx_task_user_status", "user_id", "status"),
        Index("idx_task_created_at", "created_at"),
    )
