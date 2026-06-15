"""
LangChain Memory MySQL 适配器

提供基于 MySQL 的对话历史存储，支持 LangChain 的 BaseChatMessageHistory 接口
"""

import json
import logging
from typing import List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import BaseChatMessageHistory
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from models.conversation_model import ConversationMessage, ConversationSummary

logger = logging.getLogger(__name__)


class MySQLChatMessageHistory(BaseChatMessageHistory):
    """
    MySQL 实现的 LangChain ChatMessageHistory
    
    用法:
        history = MySQLChatMessageHistory(session=db_session, conversation_id="xxx")
        
        # 添加消息
        history.add_user_message("你好")
        history.add_ai_message("你好！有什么可以帮助你的？")
        
        # 获取消息
        messages = history.messages
        
        # 清空消息
        history.clear()
    """

    def __init__(
        self,
        session: AsyncSession,
        conversation_id: str,
    ):
        """
        初始化 MySQL ChatMessageHistory
        
        :param session: SQLAlchemy 异步会话
        :param conversation_id: 会话ID
        """
        self.session = session
        self.conversation_id = conversation_id
        self._messages: List[BaseMessage] = []
        self._loaded = False

    async def _ensure_loaded(self):
        """确保消息已加载"""
        if not self._loaded:
            await self._load_messages()
            self._loaded = True

    async def _load_messages(self):
        """从数据库加载消息"""
        try:
            stmt = (
                select(ConversationMessage)
                .where(ConversationMessage.conversation_id == self.conversation_id)
                .order_by(ConversationMessage.sequence_number.asc())
            )
            result = await self.session.execute(stmt)
            messages_db = result.scalars().all()

            self._messages = []
            for msg_db in messages_db:
                message = self._deserialize_message(msg_db)
                if message:
                    self._messages.append(message)

            logger.debug(f"加载消息成功: conversation_id={self.conversation_id}, count={len(self._messages)}")

        except Exception as e:
            logger.error(f"加载消息失败: {e}")
            self._messages = []

    def _deserialize_message(self, msg_db: ConversationMessage) -> Optional[BaseMessage]:
        """
        将数据库记录转换为 LangChain Message
        
        :param msg_db: 数据库消息记录
        :return: LangChain Message 对象
        """
        try:
            additional_kwargs = msg_db.additional_kwargs or {}

            if msg_db.message_type == "human":
                return HumanMessage(content=msg_db.content, additional_kwargs=additional_kwargs)
            elif msg_db.message_type == "ai":
                return AIMessage(content=msg_db.content, additional_kwargs=additional_kwargs)
            elif msg_db.message_type == "system":
                return SystemMessage(content=msg_db.content, additional_kwargs=additional_kwargs)
            else:
                logger.warning(f"未知消息类型: {msg_db.message_type}")
                return None

        except Exception as e:
            logger.error(f"反序列化消息失败: {e}")
            return None

    async def _get_next_sequence_number(self) -> int:
        """获取下一个消息序号"""
        stmt = (
            select(func.max(ConversationMessage.sequence_number))
            .where(ConversationMessage.conversation_id == self.conversation_id)
        )
        result = await self.session.execute(stmt)
        max_seq = result.scalar()
        return (max_seq or 0) + 1

    async def add_message(self, message: BaseMessage) -> None:
        """
        添加消息到历史记录
        
        :param message: LangChain Message 对象
        """
        await self._ensure_loaded()

        try:
            # 确定消息类型
            if isinstance(message, HumanMessage):
                message_type = "human"
            elif isinstance(message, AIMessage):
                message_type = "ai"
            elif isinstance(message, SystemMessage):
                message_type = "system"
            else:
                message_type = message.type

            # 获取下一个序号
            sequence_number = await self._get_next_sequence_number()

            # 创建数据库记录
            msg_db = ConversationMessage(
                conversation_id=self.conversation_id,
                message_type=message_type,
                content=message.content,
                additional_kwargs=message.additional_kwargs if message.additional_kwargs else None,
                sequence_number=sequence_number,
                token_count=None,  # 可以在后续计算
            )

            self.session.add(msg_db)
            await self.session.flush()  # 获取 ID

            # 更新内存缓存
            self._messages.append(message)

            logger.debug(f"添加消息成功: type={message_type}, seq={sequence_number}")

        except Exception as e:
            logger.error(f"添加消息失败: {e}")
            await self.session.rollback()
            raise

    async def aadd_messages(self, messages: List[BaseMessage]) -> None:
        """批量添加消息"""
        for message in messages:
            await self.add_message(message)

    async def aget_messages(self) -> List[BaseMessage]:
        """获取消息列表"""
        await self._ensure_loaded()
        return self._messages

    async def aclear(self) -> None:
        """清空消息历史"""
        try:
            stmt = (
                select(ConversationMessage)
                .where(ConversationMessage.conversation_id == self.conversation_id)
            )
            result = await self.session.execute(stmt)
            messages_db = result.scalars().all()

            for msg_db in messages_db:
                await self.session.delete(msg_db)

            await self.session.flush()

            # 清空内存缓存
            self._messages = []
            self._loaded = True

            logger.info(f"清空消息历史成功: conversation_id={self.conversation_id}")

        except Exception as e:
            logger.error(f"清空消息历史失败: {e}")
            await self.session.rollback()
            raise

    # 同步方法（兼容 LangChain 接口）
    @property
    def messages(self) -> List[BaseMessage]:
        """获取消息列表（同步版本，不推荐使用）"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.aget_messages())

    def add_user_message(self, message: str) -> None:
        """添加用户消息（同步版本）"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        loop.run_until_complete(self.add_message(HumanMessage(content=message)))

    def add_ai_message(self, message: str) -> None:
        """添加 AI 消息（同步版本）"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        loop.run_until_complete(self.add_message(AIMessage(content=message)))

    def clear(self) -> None:
        """清空消息（同步版本）"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        loop.run_until_complete(self.aclear())


class MySQLConversationSummaryMemory:
    """
    MySQL 会话摘要管理
    
    用于存储和检索对话摘要，支持长对话压缩
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_summary(self, conversation_id: str) -> Optional[str]:
        """
        获取会话摘要
        
        :param conversation_id: 会话ID
        :return: 摘要内容，如果不存在返回 None
        """
        try:
            stmt = select(ConversationSummary).where(
                ConversationSummary.conversation_id == conversation_id
            )
            result = await self.session.execute(stmt)
            summary_db = result.scalars().first()

            if summary_db:
                logger.debug(f"获取会话摘要成功: conversation_id={conversation_id}")
                return summary_db.summary

            return None

        except Exception as e:
            logger.error(f"获取会话摘要失败: {e}")
            return None

    async def save_summary(
        self,
        conversation_id: str,
        summary: str,
        message_count: int = 0,
        last_message_id: Optional[str] = None,
        summary_token_count: Optional[int] = None,
    ) -> ConversationSummary:
        """
        保存或更新会话摘要
        
        :param conversation_id: 会话ID
        :param summary: 摘要内容
        :param message_count: 已总结的消息数量
        :param last_message_id: 最后一条被总结的消息ID
        :param summary_token_count: 摘要Token数量
        :return: 摘要记录
        """
        try:
            # 查找是否已存在
            stmt = select(ConversationSummary).where(
                ConversationSummary.conversation_id == conversation_id
            )
            result = await self.session.execute(stmt)
            summary_db = result.scalars().first()

            if summary_db:
                # 更新
                summary_db.summary = summary
                summary_db.message_count = message_count
                summary_db.last_message_id = last_message_id
                summary_db.summary_token_count = summary_token_count
            else:
                # 创建
                summary_db = ConversationSummary(
                    conversation_id=conversation_id,
                    summary=summary,
                    message_count=message_count,
                    last_message_id=last_message_id,
                    summary_token_count=summary_token_count,
                )
                self.session.add(summary_db)

            await self.session.flush()

            logger.info(f"保存会话摘要成功: conversation_id={conversation_id}, messages={message_count}")
            return summary_db

        except Exception as e:
            logger.error(f"保存会话摘要失败: {e}")
            await self.session.rollback()
            raise

    async def delete_summary(self, conversation_id: str) -> bool:
        """
        删除会话摘要
        
        :param conversation_id: 会话ID
        :return: 是否删除成功
        """
        try:
            stmt = select(ConversationSummary).where(
                ConversationSummary.conversation_id == conversation_id
            )
            result = await self.session.execute(stmt)
            summary_db = result.scalars().first()

            if summary_db:
                await self.session.delete(summary_db)
                await self.session.flush()
                logger.info(f"删除会话摘要成功: conversation_id={conversation_id}")
                return True

            return False

        except Exception as e:
            logger.error(f"删除会话摘要失败: {e}")
            await self.session.rollback()
            return False
