"""
元数据定义 ORM 模型

维护知识库自定义元数据字段的定义
"""

import uuid
from datetime import datetime

from sqlalchemy import String, DateTime, Boolean
from sqlalchemy.orm import Mapped, mapped_column

from core.database import Base


def _generate_uuid() -> str:
    return str(uuid.uuid4())


class MetadataDefinition(Base):
    """元数据字段定义"""
    __tablename__ = "metadata_definitions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_generate_uuid)
    # 字段名称（英文标识）
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    # 显示名称
    display_name: Mapped[str] = mapped_column(String(128), nullable=False)
    # 字段类型：string / number / time
    field_type: Mapped[str] = mapped_column(String(16), nullable=False)
    # 是否内置（内置字段不可删除）
    is_builtin: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    # 是否启用
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    # 创建时间
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    # 更新时间
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
