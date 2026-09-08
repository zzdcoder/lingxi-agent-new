"""
知识服务薄适配层

为 Agent 图的知识库分支提供完整流程封装，复用现有 RAG 组件（HybridRetriever、
SemanticCache、MySQL 记忆），不重写检索链路，仅做编排适配。
"""
import asyncio
import logging
from typing import Any, List, Optional

from langchain_core.messages import HumanMessage, AIMessage
from qdrant_client.models import SparseVector
from sqlalchemy.ext.asyncio import AsyncSession

from rag.rag_conversation_service import RAGConversationService
from rag.semantic_cache import get_semantic_cache
from embeddings.embedding_deal import DashScopeEmbedding
from core.config import settings

logger = logging.getLogger(__name__)


class KnowledgeService:
    """
    知识库问答服务（Agent 图节点复用）。

    职责：
    1. 预计算 query embedding（全链路复用）；
    2. 语义缓存命中检查与写入；
    3. 混合检索 + 上下文格式化（委托现有 RAGConversationService）；
    4. 压缩历史获取与消息落库。
    """

    def __init__(
        self,
        db_session: AsyncSession,
        rag_service: Optional[RAGConversationService] = None,
    ):
        self.db = db_session
        self.rag = rag_service or RAGConversationService(db_session=db_session)
        self.embeddings = DashScopeEmbedding(api_key=settings.api_key)

    async def compute_embedding(self, text: str) -> Optional[List[float]]:
        """
        预计算文本向量（失败时返回 None，不影响主流程）。

        :param text: 查询文本
        :return: 向量列表或 None
        """
        try:
            return await asyncio.to_thread(self.embeddings.embed_query, text)
        except Exception as e:
            logger.warning(f"预计算 query embedding 失败: {e}")
            return None

    async def compute_embedding_with_sparse(
        self, text: str
    ) -> tuple[Optional[List[float]], Optional[SparseVector]]:
        """
        一次调用生成稠密 + 稀疏双向量（text-embedding-v4 双输出，全链路复用）。

        :param text: 查询文本
        :return: (稠密向量, 稀疏向量)，任一失败时对应为 None，不影响主流程
        """
        try:
            return await asyncio.to_thread(
                self.embeddings.embed_query_with_sparse, text
            )
        except Exception as e:
            logger.warning(f"预计算 query 双向量失败: {e}")
            return None, None

    async def get_cached(
        self,
        user_input: str,
        username: Optional[str],
        embedding: Optional[List[float]],
    ) -> Optional[Any]:
        """
        语义缓存命中检查。

        :param user_input: 用户输入
        :param username: 用户名（权限隔离）
        :param embedding: 预计算向量
        :return: CacheEntry 或 None
        """
        cache = get_semantic_cache()
        if not cache:
            return None
        try:
            return await cache.get(
                user_input, username, precomputed_embedding=embedding
            )
        except Exception as e:
            logger.warning(f"语义缓存查询异常（降级为无缓存）: {e}")
            return None

    async def retrieve(
        self,
        user_input: str,
        username: Optional[str],
        embedding: Optional[List[float]],
        sparse: Optional[SparseVector] = None,
        k: int = 4,
    ) -> tuple[List[Any], str]:
        """
        混合检索并格式化上下文。

        :param user_input: 用户输入
        :param username: 用户名（权限过滤）
        :param embedding: 预计算向量
        :param sparse: 预计算稀疏向量（与稠密向量同一次调用生成）
        :param k: 返回文档数
        :return: (文档列表, 格式化上下文字符串)
        """
        try:
            docs = await self.rag.retrieve_context_public(
                user_input, k=k, login_username=username,
                precomputed_embedding=embedding,
                precomputed_sparse=sparse,
            )
            return docs, self.rag.format_context_public(docs)
        except Exception as e:
            logger.error(f"Agent 知识检索失败: {e}")
            return [], ""

    async def get_history(self, conversation_id: str) -> List[Any]:
        """
        获取压缩后的对话历史（复用现有摘要压缩逻辑）。

        :param conversation_id: 会话 ID
        :return: 消息列表
        """
        try:
            return await self.rag.get_compressed_history_public(conversation_id)
        except Exception as e:
            logger.error(f"获取压缩历史失败: {e}")
            return []

    async def save_answer(
        self,
        conversation_id: str,
        user_input: str,
        answer: str,
    ) -> None:
        """
        将问答写入会话历史（消息落库）。

        :param conversation_id: 会话 ID
        :param user_input: 用户输入
        :param answer: 回答内容
        """
        try:
            await self.rag.save_messages_public(
                conversation_id, user_input, answer
            )
        except Exception as e:
            logger.warning(f"消息落库失败（非致命）: {e}")

    async def put_cache(
        self,
        user_input: str,
        embedding: List[float],
        answer: str,
        context_docs: Optional[List[Any]],
        username: Optional[str],
    ) -> None:
        """
        写入语义缓存（best-effort，失败不影响主流程）。

        :param user_input: 用户输入
        :param embedding: 预计算向量
        :param answer: 完整回答
        :param context_docs: 检索文档
        :param username: 用户名
        """
        cache = get_semantic_cache()
        if not cache:
            return
        try:
            await cache.put(
                query=user_input,
                query_embedding=embedding,
                answer=answer,
                context_docs=context_docs,
                login_username=username,
            )
        except Exception as e:
            logger.warning(f"写入语义缓存失败（非致命）: {e}")
