"""
文件 ORM 模型

维护上传文件与腾讯云 COS 对象之间的映射关系
"""

import uuid
from datetime import datetime

from sqlalchemy import String, Integer, DateTime, Text
from sqlalchemy.orm import Mapped, mapped_column

from core.database import Base


def _generate_uuid() -> str:
    return str(uuid.uuid4())


class UploadedFile(Base):
    """已上传文件记录"""
    __tablename__ = "uploaded_files"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_generate_uuid)
    # 用户原始文件名
    original_name: Mapped[str] = mapped_column(String(255), nullable=False)
    # 存储到 COS 时的对象名（通常加随机前缀防止冲突）
    storage_name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    # MIME 类型
    content_type: Mapped[str] = mapped_column(String(128), nullable=False, default="application/octet-stream")
    # 文件大小（字节）
    size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # COS 所属存储桶
    cos_bucket: Mapped[str] = mapped_column(String(128), nullable=False)
    # COS 对象 Key
    cos_key: Mapped[str] = mapped_column(Text, nullable=False)
    # COS 访问 URL（有时效性或永久访问链接）
    cos_url: Mapped[str] = mapped_column(Text, nullable=False)
    # 关联的对话 ID（可选）
    conversation_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    # 上传时间
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
