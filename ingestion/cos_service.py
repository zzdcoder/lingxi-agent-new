"""
文件存储服务

优先使用腾讯云 COS，配置无效时自动回退到本地文件系统存储。
本地存储文件通过 /uploads 静态资源路由对外暴露。
支持文件上传和下载功能。
"""

import os
import uuid
import mimetypes
import logging
from datetime import datetime
from functools import lru_cache
from typing import Optional
from pathlib import Path

from fastapi import UploadFile

from core.config import settings
from core.exceptions import FileUploadException, FileDownloadException

logger = logging.getLogger(__name__)

# 本地存储根目录
LOCAL_UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")


class FileStorageService:
    """统一文件存储服务（COS + 本地回退）"""

    def __init__(self) -> None:
        self._cos_enabled = False
        self._cos_client = None
        self._bucket = None
        self._domain = None

        # 尝试初始化 COS
        try:
            from qcloud_cos import CosConfig, CosS3Client

            if all([settings.cos_secret_id, settings.cos_secret_key, settings.cos_bucket]):
                config = CosConfig(
                    Region=settings.cos_region,
                    SecretId=settings.cos_secret_id,
                    SecretKey=settings.cos_secret_key,
                )
                self._cos_client = CosS3Client(config)
                self._bucket = settings.cos_bucket
                self._domain = settings.cos_domain or f"https://{settings.cos_bucket}.cos.{settings.cos_region}.myqcloud.com"
                self._cos_enabled = True
        except Exception:
            pass

        # 确保本地存储目录存在
        os.makedirs(LOCAL_UPLOAD_DIR, exist_ok=True)

    def _generate_key(self, original_name: str) -> str:
        """生成存储对象 Key：uploads/年月/随机UUID_原文件名"""
        date_prefix = datetime.utcnow().strftime("%Y%m")
        unique_name = f"{uuid.uuid4().hex}_{original_name}"
        return f"uploads/{date_prefix}/{unique_name}"

    def upload_file(self, file: UploadFile) -> dict:
        """
        上传文件：优先 COS，失败或不可用时回退本地存储

        :param file: FastAPI UploadFile
        :return: 包含 bucket, key, url, size, content_type 的字典
        """
        original_name = file.filename or "unnamed"
        content_type = file.content_type or mimetypes.guess_type(original_name)[0] or "application/octet-stream"
        key = self._generate_key(original_name)

        try:
            file_bytes = file.file.read()
        finally:
            file.file.close()

        # 优先尝试 COS
        if self._cos_enabled:
            try:
                self._cos_client.put_object(
                    Bucket=self._bucket,
                    Body=file_bytes,
                    Key=key,
                    ContentType=content_type,
                )
                url = f"{self._domain}/{key}"
                return {
                    "bucket": self._bucket,
                    "key": key,
                    "url": url,
                    "size": len(file_bytes),
                    "content_type": content_type,
                    "original_name": original_name,
                    "storage_name": key.split("/")[-1],
                }
            except Exception:
                # COS 上传失败，回退到本地存储
                pass

        # 本地存储回退
        local_path = os.path.join(LOCAL_UPLOAD_DIR, key)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(file_bytes)

        # 返回可通过静态资源访问的 URL
        local_url = f"/uploads/{key}"
        return {
            "bucket": "local",
            "key": key,
            "url": local_url,
            "size": len(file_bytes),
            "content_type": content_type,
            "original_name": original_name,
            "storage_name": key.split("/")[-1],
        }

    def delete_file(self, key: str) -> None:
        """删除文件：优先 COS，回退本地"""
        if self._cos_enabled:
            try:
                self._cos_client.delete_object(Bucket=self._bucket, Key=key)
                return
            except Exception:
                pass

        # 本地删除
        local_path = os.path.join(LOCAL_UPLOAD_DIR, key)
        if os.path.exists(local_path):
            os.remove(local_path)

    def download_file(self, key: str, local_path: Optional[str] = None) -> str:
        """
        从COS下载文件到本地临时目录

        :param key: COS对象键
        :param local_path: 指定本地保存路径（可选，默认保存到临时目录）
        :return: 本地文件路径
        """
        if not self._cos_enabled:
            # 本地存储模式，直接返回本地路径
            local_file_path = os.path.join(LOCAL_UPLOAD_DIR, key)
            if os.path.exists(local_file_path):
                return local_file_path
            raise FileDownloadException(f"文件不存在: {key}")

        try:
            # 确定保存路径
            if local_path is None:
                temp_dir = getattr(settings, 'file_temp_dir', './temp')
                os.makedirs(temp_dir, exist_ok=True)
                unique_name = f"{uuid.uuid4().hex}_{os.path.basename(key)}"
                local_path = os.path.join(temp_dir, unique_name)

            logger.info(f"开始下载文件: {key} -> {local_path}")

            # 从COS下载文件
            response = self._cos_client.get_object(
                Bucket=self._bucket,
                Key=key,
            )

            # 确保目录存在
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            # 写入本地文件
            with open(local_path, 'wb') as f:
                for chunk in response['Body'].get_raw_stream():
                    f.write(chunk)

            logger.info(f"文件下载成功: {local_path}")
            return local_path

        except Exception as e:
            logger.error(f"文件下载失败: {key}, 错误: {e}")
            raise FileDownloadException(f"文件下载失败: {str(e)}")

    def download_file_bytes(self, key: str) -> bytes:
        """
        从COS下载文件为字节流

        :param key: COS对象键
        :return: 文件字节内容
        """
        if not self._cos_enabled:
            # 本地存储模式
            local_file_path = os.path.join(LOCAL_UPLOAD_DIR, key)
            if os.path.exists(local_file_path):
                with open(local_file_path, 'rb') as f:
                    return f.read()
            raise FileDownloadException(f"文件不存在: {key}")

        try:
            logger.info(f"开始下载文件到内存: {key}")

            response = self._cos_client.get_object(
                Bucket=self._bucket,
                Key=key,
            )

            # 读取所有内容到内存
            content = response['Body'].read()

            logger.info(f"文件下载到内存成功，大小: {len(content)} bytes")
            return content

        except Exception as e:
            logger.error(f"文件下载失败: {key}, 错误: {e}")
            raise FileDownloadException(f"文件下载失败: {str(e)}")

    def get_file_url(self, key: str, expires: int = 3600) -> str:
        """
        获取文件的访问URL

        :param key: COS对象键
        :param expires: 签名URL过期时间（秒），默认1小时
        :return: 文件访问URL
        """
        if not self._cos_enabled:
            # 本地文件
            return f"/uploads/{key}"

        try:
            # 生成带签名的临时访问URL
            url = self._cos_client.get_presigned_download_url(
                Bucket=self._bucket,
                Key=key,
                ExpiredSeconds=expires
            )
            return url
        except Exception as e:
            logger.warning(f"生成签名URL失败: {e}, 返回普通URL")
            return f"{self._domain}/{key}"


@lru_cache
def get_cos_service() -> FileStorageService:
    """延迟初始化文件存储服务单例"""
    return FileStorageService()
