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
from models.file_schema import FileDDLAndSplitInput, FileDDLAndSplitOutput
from ingestion.fileddl_service import get_file_ddl_split_service
from embeddings import embedding_deal
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
            try:
                handler = embedding_deal.EmbeddingHandler(settings.api_key)
                await handler.save_to_vectors(result.deal_result)
                logger.info(f"向量存储成功: file_id={input_data.file_id}")
            except Exception as e:
                logger.error(f"向量存储失败: file_id={input_data.file_id}, 错误: {e}")
                raise FileProcessingException(f"向量存储失败: {str(e)}")

        return result

    except FileProcessingException:
        raise
    except Exception as e:
        logger.error(f"文件处理失败: {e}")
        raise FileProcessingException(f"文件处理失败: {str(e)}")
