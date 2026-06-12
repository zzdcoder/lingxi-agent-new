"""
附件上传路由

提供文件上传接口，接收 multipart/form-data，
优先将文件持久化到腾讯云 COS，配置无效时自动回退到本地文件系统存储，
并在数据库中记录映射关系。
"""

import asyncio

from fastapi import APIRouter, UploadFile, File, Form, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.exceptions import FileUploadException
from models.file_model import UploadedFile
from models.file_schema import AttachmentOut
from ingestion.cos_service import get_cos_service
from utils.auth import get_current_user
from models.user_model import User

router = APIRouter(prefix="/attachments", tags=["附件管理"])


@router.post(
    "",
    response_model=AttachmentOut,
    status_code=status.HTTP_201_CREATED,
    summary="上传附件",
    description="接收文件并通过 multipart/form-data 上传，优先 COS，回退本地存储，返回文件元信息",
)
async def upload_attachment(
    file: UploadFile = File(..., description="待上传的文件"),
    conversation_id: str | None = Form(default=None, description="关联对话 ID"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> UploadedFile:
    if not file.filename:
        raise FileUploadException("文件名不能为空")

    storage_service = get_cos_service()

    # 1. 上传至存储（同步操作，在线程池中运行）
    storage_info = await asyncio.to_thread(storage_service.upload_file, file)

    # 2. 写入数据库
    db_file = UploadedFile(
        original_name=storage_info["original_name"],
        storage_name=storage_info["storage_name"],
        content_type=storage_info["content_type"],
        size=storage_info["size"],
        cos_bucket=storage_info["bucket"],
        cos_key=storage_info["key"],
        cos_url=storage_info["url"],
        conversation_id=conversation_id,
    )
    db.add(db_file)
    await db.commit()
    await db.refresh(db_file)

    return db_file
