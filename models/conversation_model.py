import uuid
import json
from datetime import datetime
from typing import Optional

from sqlalchemy import String, DateTime, Boolean, Text, Integer, Index, func, JSON
from sqlalchemy.orm import Mapped, mapped_column

from core import Base


def _generate_uuid() -> str:
    return str(uuid.uuid4())


class ConversationDefinition(Base):
    """会话定义表 - 存储会话的基本信息"""
    __tablename__ = "conversation"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_generate_uuid)
    user_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, comment="关联用户ID")
    title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, comment="会话标题")
    model: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, default="qwen-turbo", comment="使用的模型名称")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active", comment="会话状态: active, archived, deleted")
    deleted: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="逻辑删除: 0-正常, 1-删除")
    # 创建时间
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=func.current_timestamp(), comment="创建时间")
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=func.current_timestamp(), onupdate=func.current_timestamp(), comment="更新时间")

    __table_args__ = (
        Index("idx_conv_user_id_status", "user_id", "status"),
        Index("idx_conv_created_at", "created_at"),
    )


class ConversationMessage(Base):
    """会话消息表 - 存储对话历史消息（LangChain Memory）"""
    __tablename__ = "conversation_message"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_generate_uuid)
    conversation_id: Mapped[str] = mapped_column(String(36), nullable=False, comment="关联会话ID")
    
    # 消息角色: human, ai, system, function, tool
    message_type: Mapped[str] = mapped_column(String(32), nullable=False, comment="消息类型")
    
    # 消息内容（支持JSON存储复杂结构）
    content: Mapped[str] = mapped_column(Text, nullable=False, comment="消息内容")
    
    # 可选的附加数据（如工具调用、函数调用等）
    additional_kwargs: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True, comment="附加参数")
    
    # 消息顺序（用于排序）
    sequence_number: Mapped[int] = mapped_column(Integer, nullable=False, comment="消息序号")
    
    # Token 统计
    token_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, default=0, comment="Token数量")
    
    # 时间戳
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=func.current_timestamp(), comment="创建时间")

    __table_args__ = (
        Index("idx_msg_conv_id_seq", "conversation_id", "sequence_number"),
        Index("idx_msg_conv_id_type", "conversation_id", "message_type"),
        Index("idx_msg_created_at", "created_at"),
    )


class ConversationSummary(Base):
    """会话摘要表 - 存储对话摘要（用于长对话压缩）"""
    __tablename__ = "conversation_summary"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_generate_uuid)
    conversation_id: Mapped[str] = mapped_column(String(36), nullable=False, unique=True, comment="关联会话ID")
    
    # 摘要内容
    summary: Mapped[str] = mapped_column(Text, nullable=False, comment="对话摘要内容")
    
    # 摘要元数据
    message_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="已总结的消息数量")
    last_message_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, comment="最后一条被总结的消息ID")
    
    # Token 统计
    summary_token_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, comment="摘要Token数量")
    
    # 时间戳
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=func.current_timestamp(), comment="创建时间")
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=func.current_timestamp(), onupdate=func.current_timestamp(), comment="更新时间")

    __table_args__ = (
        Index("idx_summary_conv_id", "conversation_id"),
    )
