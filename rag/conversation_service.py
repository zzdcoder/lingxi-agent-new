"""
对话服务层

提供会话管理、消息存储、LLM调用和RAG检索的完整服务
使用 langchain_core + langchain_openai 直接实现，不依赖 langchain 主包
"""

import logging
import time
from typing import List, Optional, Dict, Any, AsyncGenerator

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from models.conversation_model import ConversationDefinition, ConversationMessage
from rag.memory_mysql import MySQLChatMessageHistory
from core.exceptions import ConversationException
from core.config import settings

logger = logging.getLogger(__name__)


class ConversationService:
    """
    对话服务
    
    负责：
    1. 会话的 CRUD 操作
    2. 消息的存储和检索
    3. 与 LLM 的交互（带 Memory）
    4. RAG 知识检索集成
    """

    def __init__(self, db_session: AsyncSession):
        """
        初始化对话服务
        
        :param db_session: SQLAlchemy 异步会话
        """
        self.db = db_session

    # ==========================================
    # 会话管理
    # ==========================================

    async def create_conversation(
        self,
        user_id: Optional[str],
        title: str = "新对话",
        model: str = "qwen-turbo"
    ) -> ConversationDefinition:
        """
        创建新会话
        
        :param user_id: 用户ID
        :param title: 会话标题
        :param model: 使用的模型
        :return: 会话对象
        """
        try:
            conversation = ConversationDefinition(
                user_id=user_id,
                title=title,
                model=model,
                status="active"
            )
            
            self.db.add(conversation)
            await self.db.flush()
            await self.db.refresh(conversation)
            
            logger.info(f"创建会话成功: id={conversation.id}, user_id={user_id}, title={title}")
            return conversation
            
        except Exception as e:
            logger.error(f"创建会话失败: {e}")
            await self.db.rollback()
            raise ConversationException(f"创建会话失败: {str(e)}")

    async def get_conversation(self, conversation_id: str) -> Optional[ConversationDefinition]:
        """
        获取会话详情
        
        :param conversation_id: 会话ID
        :return: 会话对象
        """
        try:
            stmt = select(ConversationDefinition).where(
                ConversationDefinition.id == conversation_id,
                ConversationDefinition.deleted == 0
            )
            result = await self.db.execute(stmt)
            return result.scalars().first()
            
        except Exception as e:
            logger.error(f"获取会话失败: {e}")
            raise ConversationException(f"获取会话失败: {str(e)}")

    async def update_conversation_model(self, conversation_id: str, model: str) -> bool:
        """
        更新会话使用的模型
        
        :param conversation_id: 会话ID
        :param model: 模型名称
        :return: 是否更新成功
        """
        try:
            stmt = select(ConversationDefinition).where(
                ConversationDefinition.id == conversation_id,
                ConversationDefinition.deleted == 0
            )
            result = await self.db.execute(stmt)
            conversation = result.scalars().first()
            
            if not conversation:
                return False
            
            conversation.model = model
            await self.db.flush()
            
            logger.info(f"更新会话模型成功: id={conversation_id}, model={model}")
            return True
            
        except Exception as e:
            logger.error(f"更新会话模型失败: {e}")
            await self.db.rollback()
            raise ConversationException(f"更新会话模型失败: {str(e)}")

    async def list_conversations(self, user_id: Optional[str] = None) -> List[ConversationDefinition]:
        """
        列出用户的会话
        
        :param user_id: 用户ID（可选）
        :return: 会话列表
        """
        try:
            stmt = (
                select(ConversationDefinition)
                .where(
                    ConversationDefinition.deleted == 0,
                    ConversationDefinition.status == "active"
                )
                .order_by(ConversationDefinition.updated_at.desc())
            )
            
            if user_id:
                stmt = stmt.where(ConversationDefinition.user_id == user_id)
            
            result = await self.db.execute(stmt)
            conversations = result.scalars().all()
            
            logger.debug(f"获取会话列表成功: count={len(conversations)}")
            return conversations
            
        except Exception as e:
            logger.error(f"获取会话列表失败: {e}")
            raise ConversationException(f"获取会话列表失败: {str(e)}")

    async def delete_conversation(self, conversation_id: str) -> bool:
        """
        删除会话（逻辑删除）
        
        :param conversation_id: 会话ID
        :return: 是否删除成功
        """
        try:
            stmt = select(ConversationDefinition).where(
                ConversationDefinition.id == conversation_id
            )
            result = await self.db.execute(stmt)
            conversation = result.scalars().first()
            
            if not conversation:
                return False
            
            # 逻辑删除
            conversation.deleted = 1
            conversation.status = "deleted"
            await self.db.flush()
            
            logger.info(f"删除会话成功: id={conversation_id}")
            return True
            
        except Exception as e:
            logger.error(f"删除会话失败: {e}")
            await self.db.rollback()
            raise ConversationException(f"删除会话失败: {str(e)}")

    # ==========================================
    # 消息管理
    # ==========================================

    async def get_messages(self, conversation_id: str) -> List[ConversationMessage]:
        """
        获取会话的消息历史
        
        :param conversation_id: 会话ID
        :return: 消息列表（按序号排序）
        """
        try:
            stmt = (
                select(ConversationMessage)
                .where(ConversationMessage.conversation_id == conversation_id)
                .order_by(ConversationMessage.sequence_number.asc())
            )
            result = await self.db.execute(stmt)
            messages = result.scalars().all()
            
            logger.debug(f"获取消息历史成功: conversation_id={conversation_id}, count={len(messages)}")
            return messages
            
        except Exception as e:
            logger.error(f"获取消息历史失败: {e}")
            raise ConversationException(f"获取消息历史失败: {str(e)}")

    # ==========================================
    # 对话服务（带 Memory）
    # ==========================================

    async def chat_with_memory(
        self,
        conversation_id: str,
        messages: List[Dict[str, Any]],
        model: str = "qwen-turbo"
    ) -> AsyncGenerator[bytes, None]:
        """
        带 Memory 的对话（流式响应）
        
        使用 langchain_core + langchain_openai 直接实现，不依赖 langchain 主包
        
        :param conversation_id: 会话ID
        :param messages: 消息列表（包含历史消息和当前消息）
        :param model: 使用的模型
        :yield: SSE 格式的响应流（字节）
        """
        start_time = time.time()
        logger.info(f"开始对话: conversation_id={conversation_id}, message_count={len(messages)}")
        
        try:
            # 1. 创建 MySQL ChatMessageHistory
            mysql_history = MySQLChatMessageHistory(
                session=self.db,
                conversation_id=conversation_id
            )
            
            # 2. 提取最后一条用户消息
            last_message = messages[-1]
            user_input = last_message.get("content", "")
            
            # 3. 加载历史消息并构建消息列表
            history_messages = await mysql_history.aget_messages()
            
            # 4. 创建 LLM
            llm = self._create_llm(model)
            
            # 5. 构建 Prompt（包含系统提示和历史消息）
            prompt_messages = [
                ("system", "你是一个专业的智能助手。请根据用户的提问提供准确、有帮助的回答。"),
            ]
            
            # 添加历史消息
            for msg in history_messages[-10:]:  # 只取最近10条
                if isinstance(msg, HumanMessage):
                    prompt_messages.append(("human", msg.content))
                elif isinstance(msg, AIMessage):
                    prompt_messages.append(("ai", msg.content))
            
            # 添加当前用户消息
            prompt_messages.append(("human", "{input}"))
            
            prompt = ChatPromptTemplate.from_messages(prompt_messages)
            
            # 6. 构建链
            chain = prompt | llm
            
            # 7. 执行对话（流式）
            full_response = ""
            async for chunk in chain.astream({"input": user_input}):
                if hasattr(chunk, 'content'):
                    content = chunk.content
                    full_response += content
                    yield f'data: {{"content": "{self._escape_json(content)}"}}\n\n'.encode('utf-8')
            
            # 8. 保存消息到数据库
            await mysql_history.add_message(HumanMessage(content=user_input))
            await mysql_history.add_message(AIMessage(content=full_response))
            
            # 显式提交事务，确保消息持久化（StreamingResponse 场景下 get_db 的自动 commit 可能不生效）
            await self.db.commit()
            
            # 发送完成标记
            yield b'data: {"done": true}\n\n'
            
            elapsed = time.time() - start_time
            logger.info(f"对话完成: conversation_id={conversation_id}, 耗时={elapsed:.2f}s")
            
        except Exception as e:
            logger.error(f"对话失败: {e}")
            error_data = f'data: {{"content": "对话失败：{str(e)}", "done": true}}\n\n'
            yield error_data.encode('utf-8')

    def _create_llm(self, model: str) -> ChatOpenAI:
        """
        创建 LLM 实例
        
        :param model: 模型名称
        :return: LangChain LLM 对象
        """
        # 这里使用 OpenAI 兼容接口（支持通义千问等）
        return ChatOpenAI(
            model=model,
            openai_api_key=settings.api_key,
            openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 通义千问
            temperature=0.7,
            streaming=True,
        )

    def _escape_json(self, text: str) -> str:
        """转义 JSON 特殊字符"""
        return text.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')


def get_conversation_service(db: AsyncSession) -> ConversationService:
    """
    获取对话服务实例（FastAPI 依赖注入）
    
    :param db: 数据库会话
    :return: 对话服务实例
    """
    return ConversationService(db_session=db)
