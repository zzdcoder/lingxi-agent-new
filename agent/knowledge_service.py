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
from rag.retrieval_cache import get_retrieval_cache
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
        intent: Optional[str] = None,
        model: Optional[str] = None,
        ttl_override: Optional[int] = None,
    ) -> Optional[Any]:
        """
        语义缓存命中检查（§18：带缓存维度）。

        :param user_input: 用户输入
        :param username: 用户名（权限隔离）
        :param embedding: 预计算向量
        :param intent: 本轮意图（与条目的 intent 不一致即 miss）
        :param model: 生成模型名
        :param ttl_override: 本轮 TTL（由意图准入策略给出）
        :return: CacheEntry 或 None
        """
        cache = get_semantic_cache()
        if not cache:
            return None
        try:
            return await cache.get(
                user_input,
                username,
                precomputed_embedding=embedding,
                intent=intent,
                model=model,
                kb_version=_cache_kb_version(),
                prompt_version=_cache_prompt_version(),
                ttl_override=ttl_override,
            )
        except Exception as e:
            logger.warning(f"语义缓存查询异常（降级为无缓存）: {e}")
            return None

    def _retrieve_enabled(self) -> bool:
        """
        检索缓存总开关（§18.13）。

        两道开关：`cache_enabled`（缓存总闸）与 `cache_retrieve_enabled`
        （检索缓存独立回滚开关）。任一关闭即回到「每次都真实检索」的旧行为。
        配置缺失时（老部署）按 getattr 缺省处理，不抛异常。
        """
        try:
            return bool(settings.cache_enabled) and bool(
                getattr(settings, "cache_retrieve_enabled", True)
            )
        except Exception:
            return False

    async def retrieve(
        self,
        user_input: str,
        username: Optional[str],
        embedding: Optional[List[float]],
        sparse: Optional[SparseVector] = None,
        k: int = 4,
    ) -> tuple[List[Any], str]:
        """
        混合检索并格式化上下文（§18.13：前置检索缓存层）。

        命中检索缓存时复用上次的 `docs + context_text`，**但仍然照常走 LLM 生成**
        —— 这是与答案缓存的根本区别：省掉向量检索 / RRF 融合 / 重排，
        保住引用溯源、审计落痕与分支语义。

        任一步异常都降级为「真实检索」，不会因为缓存而丢失检索能力。

        :param user_input: 用户输入
        :param username: 用户名（权限过滤）
        :param embedding: 预计算向量
        :param sparse: 预计算稀疏向量（与稠密向量同一次调用生成）
        :param k: 返回文档数
        :return: (文档列表, 格式化上下文字符串)
        """
        cache = get_retrieval_cache() if self._retrieve_enabled() else None
        kb_version = _cache_kb_version()

        # 1. 检索缓存查询（需要 query 向量；入口已预取，通常不额外付费）
        if cache is not None and embedding is not None:
            try:
                hit = await cache.get(
                    user_input,
                    username,
                    precomputed_embedding=embedding,
                    kb_version=kb_version,
                    top_k=k,
                )
                if hit is not None:
                    logger.info(
                        f"[检索缓存] 命中: score={hit.score:.4f}, "
                        f"docs={len(hit.docs)}（仍走 LLM 生成）"
                    )
                    return hit.docs, hit.context_text
            except Exception as e:
                logger.warning(f"检索缓存查询异常（降级为真实检索）: {e}")

        # 2. 真实检索
        try:
            docs = await self.rag.retrieve_context_public(
                user_input, k=k, login_username=username,
                precomputed_embedding=embedding,
                precomputed_sparse=sparse,
            )
        except Exception as e:
            logger.error(f"Agent 知识检索失败: {e}")
            return [], ""
        context_text = self.rag.format_context_public(docs)

        # 3. 回写检索缓存（best-effort；空结果 / 超长文档不写，保证命中即等价）
        if cache is not None and embedding is not None and docs:
            try:
                await cache.put(
                    user_input,
                    embedding,
                    docs,
                    context_text,
                    login_username=username,
                    kb_version=kb_version,
                    top_k=k,
                )
            except Exception as e:
                logger.debug(f"检索缓存写入失败（非致命）: {e}")

        return docs, context_text

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
        intent: Optional[str] = None,
        model: Optional[str] = None,
        ttl_class: Optional[int] = None,
    ) -> None:
        """
        写入语义缓存（best-effort，失败不影响主流程）。

        :param user_input: 用户输入
        :param embedding: 预计算向量
        :param answer: 完整回答
        :param context_docs: 检索文档
        :param username: 用户名
        :param intent: 本轮意图（写入维度，查询时按此过滤）
        :param model: 生成模型名
        :param ttl_class: TTL（由意图准入策略给出）
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
                intent=intent,
                model=model,
                kb_version=_cache_kb_version(),
                prompt_version=_cache_prompt_version(),
                ttl_class=ttl_class,
            )
        except Exception as e:
            logger.warning(f"写入语义缓存失败（非致命）: {e}")


def _cache_kb_version() -> str:
    """知识库内容版本（§18；P1 改为真实指纹前由配置给出，bump 即逻辑失效）。"""
    try:
        return str(settings.cache_kb_version or "v1")
    except Exception:
        return "v1"


def _cache_prompt_version() -> str:
    """提示词版本（§18；提示词变更时 bump 即逻辑失效）。"""
    try:
        return str(settings.cache_prompt_version or "1")
    except Exception:
        return "1"
