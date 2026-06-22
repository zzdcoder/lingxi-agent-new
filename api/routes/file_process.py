"""
文件处理路由

提供文件清洗和分块预览API
"""

import logging

from fastapi import APIRouter, Depends

from sqlalchemy.ext.asyncio import AsyncSession

from core import get_db, settings
from core.exceptions import FileProcessingException
from models import User
import os
from models.file_schema import FileDDLAndSplitInput, FileDDLAndSplitOutput
from ingestion.fileddl_service import get_file_ddl_split_service
from embeddings import embedding_deal
from rag.rag_conversation_service import rebuild_hybrid_index
from utils import get_current_user


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/file", tags=["文件处理"])


@router.post("/process", response_model=FileDDLAndSplitOutput)
async def process_file(
    input_data: FileDDLAndSplitInput,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    预览文件清洗和分块结果

    :param input_data: 文件处理参数
    :param db: 数据库会话
    :return: 分块后的文档列表
    """
    try:
        logger.info(f"开始处理文件: {input_data.file_id}, 类型: {input_data.file_type}")

        service = get_file_ddl_split_service()
        result = await service.process(input_data, db,current_user.username)

        logger.info(f"文件处理完成: {input_data.file_id}, 生成 {len(result.deal_result)} 个块")

        if input_data.save_embedding:
            logger.info(f"开始向量存储: file_id={input_data.file_id}, 块数={len(result.deal_result)}")

            # 步骤 1: 向量存储
            try:
                handler = embedding_deal.EmbeddingHandler(settings.api_key)

                # 从 chunk 元数据中提取 doc_id（如果用户传了的话）
                doc_id = None
                if result.deal_result and result.deal_result[0].metadata:
                    doc_id = result.deal_result[0].metadata.get("doc_id")
                    if doc_id:
                        logger.info(f"检测到 doc_id={doc_id}，将执行覆盖更新（先删后插）")

                await handler.save_to_vectors(result.deal_result, doc_id=doc_id)
                logger.info(f"向量存储成功: file_id={input_data.file_id}, doc_id={doc_id}")
            except Exception as e:
                logger.error(f"向量存储失败: file_id={input_data.file_id}, 错误: {e}")
                raise FileProcessingException(f"向量存储失败: {str(e)}")

            # 步骤 2: BM25 索引全量重建（与向量存储解耦，失败不影响文件处理结果）
            try:
                await rebuild_hybrid_index()
                logger.info(f"BM25 索引重建成功: file_id={input_data.file_id}")
            except Exception as e:
                logger.error(f"BM25 索引重建失败: file_id={input_data.file_id}, 错误: {e}")
                # BM25 重建失败不阻断文件处理，但抛出异常让调用方知晓
                raise FileProcessingException(f"BM25 索引重建失败: {str(e)}")

        return result

    except FileProcessingException:
        raise
    except Exception as e:
        logger.error(f"文件处理失败: {e}")
        raise FileProcessingException(f"文件处理失败: {str(e)}")
