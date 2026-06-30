"""
语义缓存模块

基于 Qdrant 实现语义级问答缓存：
- 将用户问题 embedding 写入独立缓存集合
- 相似问题命中时直接返回缓存回答，跳过检索 + LLM 全流程
- 支持按用户隔离（login_username）和 TTL 过期

设计原则：
    1. 复用已有的 QdrantClient 单例和 DashScopeEmbedding，零新增依赖
    2. 缓存初始化失败不影响主流程（降级为无缓存模式）
    3. 文档变更时通过 invalidate_all() 全量清除，保证一致性
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, List

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from qdrant_client.models import PointIdsList

from core.config import settings
from embeddings.embedding_deal import DashScopeEmbedding, get_qdrant_client

logger = logging.getLogger(__name__)


# =============================================================================
# 数据结构
# =============================================================================


@dataclass
class CacheEntry:
    """缓存命中时返回的结构"""
    query_text: str       # 原始缓存问题文本
    answer: str           # LLM 完整回答
    score: float          # 余弦相似度得分（0~1）
    login_username: Optional[str]
    created_at: str
    hit_count: int


# =============================================================================
# 全局单例
# =============================================================================

_semantic_cache: Optional["SemanticCache"] = None


def get_semantic_cache() -> Optional["SemanticCache"]:
    """获取全局语义缓存实例，未初始化时返回 None。"""
    return _semantic_cache


def init_semantic_cache() -> "SemanticCache":
    """
    初始化全局语义缓存实例。

    由 app/main.py lifespan 在启动时调用。
    若 cache_enabled=False，返回一个 disabled 的实例（所有操作为 no-op）。
    """
    global _semantic_cache

    if not settings.cache_enabled:
        logger.info("语义缓存已通过配置禁用（cache_enabled=False）")
        _semantic_cache = SemanticCache(
            qdrant_client=get_qdrant_client(),
            embedding_model=DashScopeEmbedding(api_key=settings.api_key),
            collection_name=settings.cache_collection_name,
            similarity_threshold=settings.cache_similarity_threshold,
            ttl_seconds=settings.cache_ttl_seconds,
            max_entries=settings.cache_max_entries,
            enabled=False,
        )
        return _semantic_cache

    _semantic_cache = SemanticCache(
        qdrant_client=get_qdrant_client(),
        embedding_model=DashScopeEmbedding(api_key=settings.api_key),
        collection_name=settings.cache_collection_name,
        similarity_threshold=settings.cache_similarity_threshold,
        ttl_seconds=settings.cache_ttl_seconds,
        max_entries=settings.cache_max_entries,
    )
    _semantic_cache.initialize()
    logger.info(
        f"语义缓存初始化完成: collection={settings.cache_collection_name}, "
        f"threshold={settings.cache_similarity_threshold}, "
        f"ttl={settings.cache_ttl_seconds}s"
    )
    return _semantic_cache


# =============================================================================
# 核心类
# =============================================================================


class SemanticCache:
    """
    基于 Qdrant 的语义问答缓存。

    工作流程：
        1. get(): 计算 query embedding → 在缓存集合中搜索相似问题
           → 命中则返回 CacheEntry（跳过 LLM）
        2. put(): LLM 回答完成后，将 query embedding + answer 写入缓存
        3. invalidate_all(): 文档变更时全量清除缓存
    """

    def __init__(
        self,
        qdrant_client: QdrantClient,
        embedding_model: DashScopeEmbedding,
        collection_name: str,
        similarity_threshold: float = 0.92,
        ttl_seconds: int = 86400,
        max_entries: int = 10000,
        vector_dim: int = 1024,
        enabled: bool = True,
    ):
        self.client = qdrant_client
        self.embedding = embedding_model
        self.collection_name = collection_name
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.vector_dim = vector_dim
        self.enabled = enabled

    def initialize(self) -> None:
        """创建缓存集合（若不存在），使用 COSINE 距离。"""
        if not self.enabled:
            return

        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]

            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_dim,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"语义缓存集合已创建: {self.collection_name}")
            else:
                info = self.client.get_collection(self.collection_name)
                logger.info(
                    f"语义缓存集合已存在: {self.collection_name}, "
                    f"points_count={info.points_count}"
                )
        except Exception as e:
            logger.error(f"语义缓存集合初始化失败: {e}")
            self.enabled = False

    # -------------------------------------------------------------------------
    # 核心操作：查询缓存
    # -------------------------------------------------------------------------

    async def get(
        self,
        query: str,
        login_username: Optional[str] = None,
        precomputed_embedding: Optional[List[float]] = None,
    ) -> Optional[CacheEntry]:
        """
        语义缓存查询。

        流程：
            1. 调用 embedding 模型计算 query 向量（或使用预计算向量）
            2. 在缓存集合中搜索最相似的条目（top_k=1）
            3. 判断 score >= 阈值
            4. 权限校验：缓存的 login_username 必须匹配
            5. 命中则更新 hit_count

        :param query: 用户问题
        :param login_username: 当前登录用户名
        :param precomputed_embedding: 可选的预计算 query embedding，避免重复计算
        :return: 命中时返回 CacheEntry，否则返回 None
        """
        if not self.enabled:
            return None

        try:
            # 1. 获取 query embedding（优先使用预计算向量）
            import asyncio

            if precomputed_embedding is not None:
                query_embedding = precomputed_embedding
            else:
                query_embedding = await asyncio.to_thread(
                    self.embedding.embed_query, query
                )

            # 2. 构建权限过滤条件
            #    允许命中：(public 且无用户) OR (当前用户名匹配)
            effective_user = login_username or "__anonymous__"
            filter_obj = Filter(
                should=[
                    # 允许命中自己创建的缓存
                    FieldCondition(
                        key="login_username",
                        match=MatchValue(value=effective_user),
                    ),
                    # 允许命中公共缓存（无特定用户）
                    FieldCondition(
                        key="login_username",
                        match=MatchValue(value="__anonymous__"),
                    ),
                ]
            )

            # 3. 搜索缓存集合（qdrant-client >= 1.7 使用 query_points 替代 search）
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                query_filter=filter_obj,
                limit=1,
                with_payload=True,
            )
            results = response.points

            if not results:
                logger.debug("语义缓存未命中（无结果）")
                return None

            best_match = results[0]

            # 4. 判断相似度阈值
            if best_match.score < self.similarity_threshold:
                logger.debug(
                    f"语义缓存未命中（score={best_match.score:.4f} "
                    f"< threshold={self.similarity_threshold}）"
                )
                return None

            # 5. 检查 TTL 过期
            payload = best_match.payload or {}
            created_at_str = payload.get("created_at", "")
            if created_at_str:
                try:
                    created_at = datetime.fromisoformat(created_at_str)
                    if datetime.now(timezone.utc) - created_at > timedelta(
                        seconds=self.ttl_seconds
                    ):
                        logger.debug("语义缓存未命中（已过期）")
                        # 异步删除过期条目
                        try:
                            self.client.delete(
                                collection_name=self.collection_name,
                                points_selector=PointIdsList(
                                    points=[best_match.id]
                                ),
                            )
                        except Exception:
                            pass
                        return None
                except (ValueError, TypeError):
                    pass

            # 6. 更新 hit_count（best-effort，不阻塞返回）
            try:
                old_hit_count = payload.get("hit_count", 0)
                self.client.set_payload(
                    collection_name=self.collection_name,
                    payload={"hit_count": old_hit_count + 1},
                    points=[best_match.id],
                )
            except Exception as e:
                logger.debug(f"更新 hit_count 失败（非致命）: {e}")

            entry = CacheEntry(
                query_text=payload.get("query_text", ""),
                answer=payload.get("answer", ""),
                score=best_match.score,
                login_username=payload.get("login_username"),
                created_at=created_at_str,
                hit_count=payload.get("hit_count", 0) + 1,
            )
            logger.info(
                f"语义缓存命中: score={best_match.score:.4f}, "
                f"hit_count={entry.hit_count}, "
                f"cached_query='{entry.query_text[:40]}...'"
            )
            return entry

        except Exception as e:
            logger.warning(f"语义缓存查询异常（降级为无缓存）: {e}")
            return None

    # -------------------------------------------------------------------------
    # 核心操作：写入缓存
    # -------------------------------------------------------------------------

    async def put(
        self,
        query: str,
        query_embedding: List[float],
        answer: str,
        context_docs: Optional[list] = None,
        login_username: Optional[str] = None,
    ) -> None:
        """
        将问答对写入语义缓存。

        :param query: 用户问题原文
        :param query_embedding: 问题 embedding 向量（可复用检索阶段已计算的向量）
        :param answer: LLM 完整回答
        :param context_docs: 检索到的文档（仅存摘要，用于调试）
        :param login_username: 当前用户名（用于权限隔离）
        """
        if not self.enabled:
            return

        try:
            effective_user = login_username or "__anonymous__"
            point_id = str(uuid.uuid4())

            # 构建 context 摘要（仅保留前 100 字，避免 payload 过大）
            context_preview = ""
            if context_docs:
                context_preview = "; ".join(
                    doc.page_content[:100] for doc in context_docs[:3]
                )

            payload = {
                "query_text": query,
                "answer": answer,
                "login_username": effective_user,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "hit_count": 0,
                "context_preview": context_preview,
            }

            import asyncio
            await asyncio.to_thread(
                self.client.upsert,
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=query_embedding,
                        payload=payload,
                    )
                ],
            )

            logger.info(
                f"语义缓存写入: user={effective_user}, "
                f"query='{query[:40]}...', answer_len={len(answer)}"
            )

            # 检查容量上限，触发淘汰（best-effort）
            await self._evict_if_needed()

        except Exception as e:
            logger.warning(f"语义缓存写入异常（非致命）: {e}")

    # -------------------------------------------------------------------------
    # 缓存管理
    # -------------------------------------------------------------------------

    async def invalidate_all(self) -> int:
        """
        全量清除缓存集合。

        实现策略：删除并重建集合（比逐条删除更高效和可靠）。

        :return: 清除前的条目数
        """
        if not self.enabled:
            return 0

        try:
            info = self.client.get_collection(self.collection_name)
            old_count = info.points_count or 0

            # 删除并重建
            self.client.delete_collection(self.collection_name)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_dim,
                    distance=Distance.COSINE,
                ),
            )

            logger.info(f"语义缓存已全量清除: 原条目数={old_count}")
            return old_count

        except Exception as e:
            logger.error(f"语义缓存清除失败: {e}")
            return 0

    async def cleanup_expired(self) -> int:
        """
        清理已过期的缓存条目。

        通过 scroll 遍历 + 时间戳判断实现（Qdrant 无原生 TTL 支持）。
        建议在 lifespan 中作为定时任务执行。

        :return: 清理的条目数
        """
        if not self.enabled:
            return 0

        try:
            cutoff = (
                datetime.now(timezone.utc) - timedelta(seconds=self.ttl_seconds)
            ).isoformat()

            # Qdrant 无原生 TTL 支持，通过 scroll 遍历 + Python 侧时间戳比较实现过期清理
            all_points, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=self.max_entries,
                with_payload=True,
                with_vectors=False,
            )

            expired_ids = []
            for point in all_points:
                created_at_str = (point.payload or {}).get("created_at", "")
                if created_at_str and created_at_str < cutoff:
                    expired_ids.append(point.id)

            if expired_ids:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=PointIdsList(points=expired_ids),
                )
                logger.info(f"清理了 {len(expired_ids)} 条过期缓存")

            return len(expired_ids)

        except Exception as e:
            logger.warning(f"清理过期缓存失败（非致命）: {e}")
            return 0

    async def get_stats(self) -> dict:
        """
        获取缓存统计信息。

        :return: {total_entries, total_hits, collection_name, enabled}
        """
        try:
            info = self.client.get_collection(self.collection_name)
            total_entries = info.points_count or 0

            # 汇总 hit_count（遍历 payload）
            total_hits = 0
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=total_entries,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                total_hits += (point.payload or {}).get("hit_count", 0)

            return {
                "collection_name": self.collection_name,
                "enabled": self.enabled,
                "total_entries": total_entries,
                "total_hits": total_hits,
                "similarity_threshold": self.similarity_threshold,
                "ttl_seconds": self.ttl_seconds,
                "max_entries": self.max_entries,
            }
        except Exception as e:
            return {
                "collection_name": self.collection_name,
                "enabled": self.enabled,
                "error": str(e),
            }

    # -------------------------------------------------------------------------
    # 内部方法
    # -------------------------------------------------------------------------

    async def _evict_if_needed(self) -> None:
        """
        当缓存条目超过 max_entries 时，淘汰 hit_count 最低的条目。

        采用简单策略：超过上限时清除 hit_count=0 的旧条目。
        若仍然超限，则清除最早创建的条目（按 created_at 排序）。
        """
        try:
            info = self.client.get_collection(self.collection_name)
            current_count = info.points_count or 0

            if current_count <= self.max_entries:
                return

            logger.info(
                f"缓存条目 ({current_count}) 超过上限 ({self.max_entries})，开始淘汰"
            )

            # 优先清除 hit_count=0 的条目
            zero_hit_filter = Filter(
                must=[
                    FieldCondition(
                        key="hit_count",
                        match=MatchValue(value=0),
                    )
                ]
            )

            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=zero_hit_filter,
                limit=current_count - self.max_entries + 100,
                with_payload=True,
                with_vectors=False,
            )

            if points:
                # 按 created_at 排序，删除最旧的
                sorted_points = sorted(
                    points,
                    key=lambda p: (p.payload or {}).get("created_at", ""),
                )
                to_delete = sorted_points[: current_count - self.max_entries]
                if to_delete:
                    self.client.delete(
                        collection_name=self.collection_name,
                        points_selector=PointIdsList(
                            points=[p.id for p in to_delete]
                        ),
                    )
                    logger.info(f"淘汰了 {len(to_delete)} 条低热度缓存")

        except Exception as e:
            logger.debug(f"缓存淘汰执行异常（非致命）: {e}")
