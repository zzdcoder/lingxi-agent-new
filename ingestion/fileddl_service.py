"""
文件清洗处理服务

企业级文件清洗和分割服务，支持多种文件格式的解析、
文本清洗和智能分块处理。
"""

import os
import asyncio
import logging
import time
from typing import Optional, List, Dict, Any
from pathlib import Path

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.documents import Document

from core import get_db
from core.config import settings
from core.exceptions import FileProcessingException, FileDownloadException
from models import User, UserOut
from models.file_schema import FileDDLAndSplitInput, FileDDLAndSplitOutput, ChunkResult

# 导入相关模块
from ingestion.cos_service import get_cos_service, FileStorageService
from ingestion.file_parser import get_file_parser, FileParserRegistry
from ingestion.text_cleaner import TextCleaner, TextCleanOptions, create_cleaner_from_options
from ingestion.text_splitter import get_text_splitter, IntelligentTextSplitter, create_split_config_from_input
from utils import get_current_user

logger = logging.getLogger(__name__)


class FileDDLSplitService:
    """
    文件清洗和分割服务

    完整流程：
    1. 从COS下载文件到本地临时目录
    2. 根据文件类型解析文件内容
    3. 执行文本清洗和预处理
    4. 智能分块处理
    5. 返回分割后的文档块
    """

    # 领域分类中英文映射（保留中文，额外写入英文到元数据）
    DOMAIN_MAP = {
        "技术类": "technical",
        "运维类": "operations",
        "人事类": "hr",
        "客户类": "customer",
    }

    def __init__(
        self,
        storage_service: FileStorageService,
        parser_registry: FileParserRegistry,
    ):
        self.storage_service = storage_service
        self.parser_registry = parser_registry
        self.text_splitter = get_text_splitter()

    async def process(
        self,
        input_data: FileDDLAndSplitInput,
        db: AsyncSession,
        user_name: str=''
    ) -> FileDDLAndSplitOutput:
        """
        处理文件清洗和分割

        :param input_data: 输入参数
        :param db: 数据库会话
        :return: 分割结果
        """
        local_file_path = None
        total_start = time.time()
        try:
            # Step 1: 获取文件信息并下载
            logger.info(f"开始处理文件: {input_data.file_id}")
            step_start = time.time()
            file_info = await self._get_file_info(input_data.file_id, db)

            # Step 2: 下载文件到本地
            cos_key = file_info['cos_key']
            original_name = file_info['original_name']
            local_file_path = await asyncio.to_thread(
                self.storage_service.download_file, cos_key
            )
            logger.info(f"文件下载成功: {local_file_path}, 耗时: {time.time() - step_start:.2f}s")

            # Step 3: 解析文件内容（可能涉及IO和CPU，放线程池）
            step_start = time.time()
            content = await asyncio.to_thread(
                self.parser_registry.parse_file, local_file_path
            )
            logger.info(f"文件解析成功，内容长度: {len(content)}, 耗时: {time.time() - step_start:.2f}s")

            # Step 4: 文本清洗（CPU密集型，放线程池）
            step_start = time.time()
            cleaned_content = await asyncio.to_thread(
                self._clean_text,
                content,
                input_data.ddl_option1,
                input_data.ddl_option2
            )
            logger.info(f"文本清洗完成，原长度: {len(content)}, 清洗后: {len(cleaned_content)}, 耗时: {time.time() - step_start:.2f}s")

            # Step 5: 文本分割（CPU密集型，放线程池）
            step_start = time.time()
            merged_meta = {}
            for meta in input_data.metadata_list:
                merged_meta.update(meta)
            documents = await asyncio.to_thread(
                self._split_text,
                cleaned_content,
                input_data.file_type,
                input_data.separators,
                input_data.chunk_size,
                input_data.chunk_overlap,
                merged_meta|{
                    "create_username":user_name,
                    'file_id': input_data.file_id,
                    'original_name': original_name,
                    'auth_option': input_data.auth_option,
                    'belong_domain': input_data.belong_demain,
                    'belong_domain_en': [
                        self.DOMAIN_MAP.get(d, d)
                        for d in input_data.belong_demain
                    ],
                }
            )
            logger.info(f"文本分割完成，生成 {len(documents)} 个块, 耗时: {time.time() - step_start:.2f}s")

            # 转换为 ChunkResult 列表
            chunk_results = [
                ChunkResult(page_content=doc.page_content, metadata=doc.metadata)
                for doc in documents
            ]

            logger.info(f"文件处理总耗时: {time.time() - total_start:.2f}s")
            return FileDDLAndSplitOutput(deal_result=chunk_results)

        except Exception as e:
            logger.error(f"文件处理失败: {e}")
            raise FileProcessingException(f"文件处理失败: {str(e)}")

        finally:
            # 清理临时文件
            if local_file_path and os.path.exists(local_file_path):
                try:
                    os.remove(local_file_path)
                    logger.debug(f"临时文件已清理: {local_file_path}")
                except Exception as e:
                    logger.warning(f"清理临时文件失败: {local_file_path}, {e}")

    async def _get_file_info(
        self,
        file_id: str,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """
        从数据库获取文件信息

        :param file_id: 文件ID
        :param db: 数据库会话
        :return: 文件信息字典
        """
        from models.file_model import UploadedFile

        result = await db.get(UploadedFile, file_id)

        if not result:
            raise FileProcessingException(f"文件不存在: {file_id}")

        return {
            'id': result.id,
            'original_name': result.original_name,
            'cos_key': result.cos_key,
            'cos_url': result.cos_url,
            'content_type': result.content_type,
            'size': result.size,
        }

    def _clean_text(
        self,
        content: str,
        ddl_option1: bool,
        ddl_option2: bool
    ) -> str:
        """
        清洗文本

        :param content: 原始文本
        :param ddl_option1: 是否移除额外空白
        :param ddl_option2: 是否移除URL和邮箱
        :return: 清洗后的文本
        """
        options = TextCleanOptions(
            remove_extra_whitespace=ddl_option1,
            remove_urls=ddl_option2,
            remove_emails=ddl_option2,
        )
        cleaner = TextCleaner(options)
        return cleaner.clean(content)

    def _split_text(
        self,
        content: str,
        file_type: str,
        separators: str,
        chunk_size: int,
        chunk_overlap: int,
        metadata: Dict[str, Any]
    ) -> List[Document]:
        """
        分割文本

        :param content: 文本内容
        :param file_type: 文件类型
        :param separators: 分隔符
        :param chunk_size: 块大小
        :param chunk_overlap: 块重叠
        :param metadata: 元数据
        :return: Document列表
        """
        # 处理默认分隔符
        if not separators or separators.strip() == "":
            separators = settings.file_default_separators

        return self.text_splitter.split_text(
            text=content,
            file_type=file_type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators_str=separators,
            metadata=metadata
        )

    async def process_simple(
        self,
        input_data: FileDDLAndSplitInput,
        file_content: bytes
    ) -> FileDDLAndSplitOutput:
        """
        简化处理：直接处理文件内容（不经过COS下载）

        :param input_data: 输入参数
        :param file_content: 文件字节内容
        :return: 分割结果
        """
        temp_file_path = None
        try:
            # 保存到临时文件
            temp_dir = getattr(settings, 'file_temp_dir', './temp')
            os.makedirs(temp_dir, exist_ok=True)

            import uuid
            temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4().hex}.{input_data.file_type}")

            with open(temp_file_path, 'wb') as f:
                f.write(file_content)

            # 解析文件
            content = self.parser_registry.parse_file(temp_file_path)

            # 清洗和分割
            cleaned_content = self._clean_text(
                content,
                input_data.ddl_option1,
                input_data.ddl_option2
            )

            documents = self._split_text(
                cleaned_content,
                input_data.file_type,
                input_data.separators,
                input_data.chunk_size,
                input_data.chunk_overlap,
                {
                    'file_id': input_data.file_id,
                    'metadata_list': input_data.metadata_list,
                    'auth_option': input_data.auth_option,
                    'belong_domain': input_data.belong_demain,
                    'belong_domain_en': [
                        self.DOMAIN_MAP.get(d, d)
                        for d in input_data.belong_demain
                    ],
                }
            )

            return FileDDLAndSplitOutput(deal_result=[
                ChunkResult(page_content=doc.page_content, metadata=doc.metadata)
                for doc in documents
            ])

        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception:
                    pass


# 服务实例工厂
_storage_service: Optional[FileStorageService] = None
_parser_registry: Optional[FileParserRegistry] = None


def get_file_ddl_split_service() -> FileDDLSplitService:
    """获取文件DDL分割服务实例"""
    global _storage_service, _parser_registry

    if _storage_service is None:
        _storage_service = get_cos_service()

    if _parser_registry is None:
        _parser_registry = get_file_parser()

    return FileDDLSplitService(
        storage_service=_storage_service,
        parser_registry=_parser_registry
    )


def file_ddl_split_service(
    db: Depends(get_db),
    input_data: FileDDLAndSplitInput
) -> FileDDLAndSplitOutput:
    """
    文件清洗处理服务入口函数

    :param db: 数据库会话依赖
    :param input_data: 输入参数
    :return: 分割结果
    """
    service = get_file_ddl_split_service()

    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(service.process(input_data, db))