"""
对话路由

提供会话管理和对话接口
"""

import logging
from typing import Optional
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.exceptions import ConversationException
from models.conversation_schema import (
    ConversationCreate,
    ConversationOut,
    MessageOut,
    ChatRequest,
)
from models.user_model import User
from rag.conversation_service import ConversationService, get_conversation_service
from rag.rag_conversation_service import RAGConversationService, get_rag_conversation_service
from utils.auth import get_current_user_id
from utils.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/conversations", tags=["对话管理"])


# ==========================================
# 会话管理接口
# ==========================================

@router.get("", response_model=list[ConversationOut])
async def list_conversations(
    db: AsyncSession = Depends(get_db),
    current_user_id: Optional[str] = Depends(get_current_user_id)
):
    """
    获取会话列表
    
    返回当前用户的所有活跃会话，按更新时间倒序排列
    """
    try:
        logger.info(f"获取会话列表: user_id={current_user_id}")
        
        service = get_conversation_service(db)
        conversations = await service.list_conversations(user_id=current_user_id)
        
        # 转换为响应模型
        result = []
        for conv in conversations:
            result.append(ConversationOut(
                id=conv.id,
                user_id=conv.user_id,
                title=conv.title,
                status=conv.status,
                model=conv.model,
                created_at=conv.created_at,
                updated_at=conv.updated_at
            ))
        
        logger.info(f"获取会话列表成功: count={len(result)}")
        return result
        
    except ConversationException:
        raise
    except Exception as e:
        logger.error(f"获取会话列表失败: {e}")
        raise ConversationException(f"获取会话列表失败: {str(e)}")


@router.post("", response_model=ConversationOut)
async def create_conversation(
    input_data: ConversationCreate = Depends(),
    db: AsyncSession = Depends(get_db),
    current_user_id: Optional[str] = Depends(get_current_user_id)
):
    """
    创建新会话
    
    :param input_data: 会话创建参数（通过 Query 传递）
    :param db: 数据库会话
    :param current_user_id: 当前用户ID
    :return: 创建的会话
    """
    try:
        logger.info(f"创建会话: title={input_data.title}, user_id={current_user_id}")
        
        service = get_conversation_service(db)
        conversation = await service.create_conversation(
            user_id=current_user_id,
            title=input_data.title,
            model=input_data.model
        )
        
        logger.info(f"创建会话成功: id={conversation.id}")
        
        return ConversationOut(
            id=conversation.id,
            user_id=conversation.user_id,
            title=conversation.title,
            status=conversation.status,
            model=conversation.model,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at
        )
        
    except ConversationException:
        raise
    except Exception as e:
        logger.error(f"创建会话失败: {e}")
        raise ConversationException(f"创建会话失败: {str(e)}")


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
    current_user_id: Optional[str] = Depends(get_current_user_id)
):
    """
    删除会话（逻辑删除）
    
    :param conversation_id: 会话ID
    :param db: 数据库会话
    :param current_user_id: 当前用户ID
    :return: 删除结果
    """
    try:
        logger.info(f"删除会话: id={conversation_id}, user_id={current_user_id}")
        
        service = get_conversation_service(db)
        success = await service.delete_conversation(conversation_id)
        
        if not success:
            raise ConversationException(f"会话不存在: {conversation_id}")
        
        logger.info(f"删除会话成功: id={conversation_id}")
        return {"message": "会话已删除"}
        
    except ConversationException:
        raise
    except Exception as e:
        logger.error(f"删除会话失败: {e}")
        raise ConversationException(f"删除会话失败: {str(e)}")


# ==========================================
# 消息管理接口
# ==========================================

@router.get("/{conversation_id}/messages", response_model=list[MessageOut])
async def list_messages(
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
    current_user_id: Optional[str] = Depends(get_current_user_id)
):
    """
    获取会话的消息历史
    
    :param conversation_id: 会话ID
    :param db: 数据库会话
    :param current_user_id: 当前用户ID
    :return: 消息列表
    """
    try:
        logger.info(f"获取消息历史: conversation_id={conversation_id}")
        
        service = get_conversation_service(db)
        messages = await service.get_messages(conversation_id)
        
        # 转换为响应模型
        result = []
        for msg in messages:
            result.append(MessageOut(
                id=msg.id,
                conversation_id=msg.conversation_id,
                message_type=msg.message_type,
                content=msg.content,
                additional_kwargs=msg.additional_kwargs,
                sequence_number=msg.sequence_number,
                token_count=msg.token_count,
                created_at=msg.created_at
            ))
        
        logger.info(f"获取消息历史成功: count={len(result)}")
        return result
        
    except ConversationException:
        raise
    except Exception as e:
        logger.error(f"获取消息历史失败: {e}")
        raise ConversationException(f"获取消息历史失败: {str(e)}")


# ==========================================
# 对话接口（流式）
# ==========================================

@router.post("/chat")
async def chat(
    chat_request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    流式对话接口
    
    发送消息并接收流式响应（SSE）
    支持普通对话和 RAG 知识检索对话
    
    :param chat_request: 对话请求
    :param db: 数据库会话
    :param current_user_id: 当前用户ID
    :return: SSE 流式响应
    """
    try:
        logger.info(
            f"开始流式对话: conversation_id={chat_request.conversation_id}, "
            f"message_count={len(chat_request.messages)}, model={chat_request.model}, "
            f"use_rag={chat_request.use_rag}"
        )
        
        # 根据 use_rag 参数选择服务
        if chat_request.use_rag:
            # 使用 RAG 对话服务
            service = get_rag_conversation_service(db)
            
            async def event_generator():
                """SSE 事件生成器"""
                async for chunk in service.chat_with_rag(
                    conversation_id=chat_request.conversation_id,
                    messages=chat_request.messages,
                    model=chat_request.model,
                    login_username=current_user.username
                ):
                    yield chunk
        else:
            # 使用普通对话服务
            service = get_conversation_service(db)
            
            async def event_generator():
                """SSE 事件生成器"""
                async for chunk in service.chat_with_memory(
                    conversation_id=chat_request.conversation_id,
                    messages=chat_request.messages,
                    model=chat_request.model
                ):
                    yield chunk
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
            }
        )
        
    except ConversationException:
        raise
    except Exception as e:
        logger.error(f"流式对话失败: {e}")
        raise ConversationException(f"流式对话失败: {str(e)}")
