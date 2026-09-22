"""
检索缓存模块（设计文档 §18.13 / cache_architecture_redesign §3.5）

**定位：默认加速层；答案缓存（rag/semantic_cache.py）降为高档位。**

与答案缓存的本质区别：

| | 答案缓存（SemanticCache） | 检索缓存（本模块） |
|---|---|---|
| 缓存内容 | LLM 最终回答 | `doc_ids + context_text + 完整 docs` |
| 命中后 | **跳过整条链路**（不检索、不生成、不落审计） | **照常走 LLM 生成** |
| 省掉 | 检索 + 生成 | 向量检索 + RRF 融合 + 重排 + 上下文格式化 |
| 保住 | —— | 引用溯源、审计落痕、分支语义、答案与当前上下文一致 |
| 阈值 | 0.95（严） | 0.90（松） |
| 意图准入 | 受 `cache_policy_for` 约束（task 一律 deny） | **不受约束**（见下） |

**为什么阈值可以放宽、且不受意图准入约束？**

因为一次错误命中的代价完全不同：
- 答案缓存错命中 → **直接给出错误答案**（不可接受）；
- 检索缓存错命中 → 只是少了一次「本来该做的检索」，LLM 仍基于当前上下文
  实时生成，**不会产生错误答案**，最坏情况是召回略有偏差。

同理，检索缓存**不需要**按意图禁用：`task` 分支的副作用（写库、调工具）
发生在检索之外，缓存检索结果既不影响副作用执行，也不影响审计落痕。
这正是 §3.5 说的「意图缓存有没有意义」的问题自然消失——
每轮请求照常做意图判定、照常进分支、照常落审计，缓存只把最贵的一段换掉。

**为什么不做匿名共享（`SHARE_ANONYMOUS = False`）？**

`_retrieve_context` 在 `login_username` 为空时**不构造权限过滤**（现有实现），
此时结果里可能含他人私有文档。若沿用答案缓存的「本人 或 匿名」共享口径，
匿名用户写入的条目会被具名用户命中 —— 等于把一次越权检索**固化并放大**。
因此检索缓存采用严格用户隔离：只匹配 `login_username == 本人`。
命中条件**比「等价重跑」更严格**，只会造成 miss（回落到真实检索），
不会造成越权或错误结果。

**保真不变式（重要）**

缓存的 `docs` 必须是**完整**的：`page_content` 超限、metadata 不可 JSON
序列化、文档数超过上限时，一律**放弃写入**而不是截断。
截断会让「命中」与「重跑」产生不一致的上下文，违反 P2。

**当前未缓存 `scores`**：`HybridRetriever` 的 RRF 融合分只用于排序，
未回写到 `Document.metadata`（`hybrid_retriever.py` 里 `scored` 解包后
只取 `doc`）。与其存一串恒为 0 的假分数，这里不存该字段；
溯源靠 `doc_ids`（`chunk_id`）。后续若在 retriever 侧埋点回写分数，
本模块的 `scores` 字段可直接启用，无需改结构。
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

from langchain_core.documents import Document
from qdrant_client import QdrantClient

from core.config import settings
from embeddings.embedding_deal import DashScopeEmbedding, get_qdrant_client
from rag.semantic_cache import SemanticCache, _metrics

logger = logging.getLogger(__name__)


# =============================================================================
# 数据结构
# =============================================================================


@dataclass
class RetrievalEntry:
    """检索缓存命中时返回的结构"""
    query_text: str                 # 原始缓存问题文本
    context_text: str               # 格式化后的上下文（直接喂给 LLM）
    docs: List[Any]                 # 重建的 Document 列表（引用溯源 / 审计）
    doc_ids: List[str]              # chunk_id 列表（溯源主键）
    score: float                    # 余弦相似度得分（0~1）
    login_username: Optional[str]
    created_at: str
    hit_count: int
    kb_version: Optional[str]       # 知识库内容版本
    top_k: Optional[int]            # 检索 top_k（参与强过滤）
    entity_tags: List[str] = field(default_factory=list)


# =============================================================================
# 文档打包 / 解包（保真优先）
# =============================================================================


def _doc_id(doc: Any) -> str:
    """
    取文档稳定标识：`metadata.chunk_id` 优先，缺失时退化为内容哈希。

    与 `rag/hybrid_retriever.py::_compute_doc_id` 口径保持一致（那边缺省用
    内容哈希，这里同样），保证同一 chunk 在缓存与检索链路里是同一个 id。
    """
    try:
        meta = getattr(doc, "metadata", None) or {}
        if isinstance(meta, dict):
            cid = meta.get("chunk_id")
            if cid:
                return str(cid)
    except Exception:
        pass
    content = getattr(doc, "page_content", "") or ""
    return f"hash:{hash(content)}"


def pack_docs(
    docs: List[Any], max_docs: int, max_doc_chars: int
) -> Optional[List[dict]]:
    """
    把 Document 列表打包成可入 Qdrant payload 的结构。

    **保真优先**：任一文档不合规（超长 / metadata 不可序列化 / 缺
    page_content）时返回 `None`，由调用方放弃写入 —— 宁可不缓存，
    也不存被截断的上下文（截断会破坏「命中 == 重跑」的不变式）。

    :param docs: Document 列表
    :param max_docs: 单条目最多缓存文档数
    :param max_doc_chars: 单文档 page_content 字符上限
    :return: 打包后的列表，或 None（表示不应缓存）
    """
    if not docs:
        return None
    if len(docs) > max_docs:
        logger.debug(f"检索缓存跳过写入：文档数 {len(docs)} > {max_docs}")
        return None

    packed: List[dict] = []
    for doc in docs:
        content = getattr(doc, "page_content", None)
        if not isinstance(content, str) or not content:
            return None
        if len(content) > max_doc_chars:
            logger.debug(
                f"检索缓存跳过写入：单文档 {len(content)} 字符 > "
                f"{max_doc_chars}（不截断，保证命中即等价）"
            )
            return None
        meta = getattr(doc, "metadata", None)
        if meta is None:
            meta = {}
        if not isinstance(meta, dict):
            return None
        try:
            # 可行性校验：Qdrant payload 必须是 JSON-safe，不可序列化则不缓存
            json.dumps(meta)
        except (TypeError, ValueError):
            return None
        packed.append(
            {
                "id": _doc_id(doc),
                "content": content,
                "metadata": meta,
            }
        )
    return packed


def unpack_docs(packed: Any) -> Optional[List[Any]]:
    """
    从 payload 还原 Document 列表；结构损坏时返回 None（调用方按 miss 处理）。

    :param packed: `pack_docs` 产出的结构
    :return: Document 列表或 None
    """
    if not isinstance(packed, list) or not packed:
        return None
    docs: List[Any] = []
    for item in packed:
        if not isinstance(item, dict):
            return None
        content = item.get("content")
        if not isinstance(content, str):
            return None
        meta = item.get("metadata")
        docs.append(
            Document(page_content=content, metadata=dict(meta) if meta else {})
        )
    return docs


# =============================================================================
# 全局单例
# =============================================================================

_retrieval_cache: Optional["RetrievalCache"] = None


def get_retrieval_cache() -> Optional["RetrievalCache"]:
    """获取全局检索缓存实例，未初始化或已禁用时返回 None。"""
    return _retrieval_cache


def init_retrieval_cache() -> "RetrievalCache":
    """
    初始化全局检索缓存实例（由 app/main.py lifespan 调用）。

    下列任一情况返回 disabled 实例（所有操作为 no-op，等价于未改动）：
    - `cache_enabled=False`（总开关关闭）
    - `cache_retrieve_enabled=False`（检索缓存回滚开关）
    """
    global _retrieval_cache

    enabled = bool(settings.cache_enabled) and bool(
        getattr(settings, "cache_retrieve_enabled", True)
    )

    _retrieval_cache = RetrievalCache(
        qdrant_client=get_qdrant_client(),
        embedding_model=DashScopeEmbedding(api_key=settings.api_key),
        collection_name=settings.cache_retrieve_collection_name,
        retrieve_threshold=settings.cache_retrieve_threshold,
        ttl_seconds=settings.cache_retrieve_ttl_seconds,
        max_entries=settings.cache_retrieve_max_entries,
        vector_dim=1024,
        enabled=enabled,
        schema_version=settings.cache_schema_version,
        max_doc_chars=settings.cache_retrieve_max_doc_chars,
        max_docs=settings.cache_retrieve_max_docs,
    )
    if not enabled:
        logger.info("检索缓存已通过配置禁用（cache_enabled / cache_retrieve_enabled）")
        return _retrieval_cache

    _retrieval_cache.initialize()
    logger.info(
        f"检索缓存初始化完成: collection={settings.cache_retrieve_collection_name}, "
        f"threshold={settings.cache_retrieve_threshold}, "
        f"ttl={settings.cache_retrieve_ttl_seconds}s, "
        f"max_entries={settings.cache_retrieve_max_entries}, "
        f"max_doc_chars={settings.cache_retrieve_max_doc_chars}"
    )
    return _retrieval_cache


# =============================================================================
# 核心类
# =============================================================================


class RetrievalCache(SemanticCache):
    """
    检索结果缓存（继承 SemanticCache 的 Qdrant 内核，覆盖维度与载荷）。

    相比基类：
    - 维度换为 `kb_version` + `top_k`（检索结果与生成模型 / 提示词无关，
      因此 `model` / `prompt_version` / `intent` **不参与**过滤 —— 这是
      检索缓存命中率天然高于答案缓存的原因之一）；
    - 关闭匿名共享（严格用户隔离，理由见模块 docstring）；
    - 阈值用 `retrieve_threshold`（默认 0.90，宽松档）；
    - 载荷是 `context_text + doc_ids + docs`，而不是 `answer`。
    """

    DIM_KEYS = ("kb_version", "top_k")
    SHARE_ANONYMOUS = False
    METRIC_PREFIX = "cache.retrieve"
    LAYER_LABEL = "检索缓存"
    # top_k 是整数维度，payload 索引需按 INTEGER 建
    INT_INDEX_FIELDS = ("top_k",)

    def __init__(
        self,
        qdrant_client: QdrantClient,
        embedding_model: DashScopeEmbedding,
        collection_name: str,
        retrieve_threshold: float = 0.90,
        ttl_seconds: int = 86400,
        max_entries: int = 5000,
        vector_dim: int = 1024,
        enabled: bool = True,
        schema_version: str = "1",
        max_doc_chars: int = 1200,
        max_docs: int = 8,
    ):
        super().__init__(
            qdrant_client=qdrant_client,
            embedding_model=embedding_model,
            collection_name=collection_name,
            similarity_threshold=retrieve_threshold,
            ttl_seconds=ttl_seconds,
            max_entries=max_entries,
            vector_dim=vector_dim,
            enabled=enabled,
            answer_threshold=retrieve_threshold,
            schema_version=schema_version,
        )
        # 检索缓存档阈值（宽松：错命中只多一次检索，不会产生错误答案）
        self.retrieve_threshold = retrieve_threshold
        self.max_doc_chars = max_doc_chars
        self.max_docs = max_docs

    # -------------------------------------------------------------------------
    # 查询
    # -------------------------------------------------------------------------

    async def get(
        self,
        query: str,
        login_username: Optional[str] = None,
        precomputed_embedding: Optional[List[float]] = None,
        kb_version: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> Optional[RetrievalEntry]:
        """
        查检索缓存。命中则返回可直接使用的 `docs` + `context_text`。

        :param query: 用户问题
        :param login_username: 当前登录用户名（严格隔离，不共享匿名条目）
        :param precomputed_embedding: 预计算向量（入口已预取，通常无需再算）
        :param kb_version: 知识库内容版本（文档变更后 bump 即逻辑失效）
        :param top_k: 检索 top_k（k 不同则结果不同，必须参与过滤）
        :return: RetrievalEntry 或 None（未命中 / 载荷损坏 / 已禁用）
        """
        if not self.enabled:
            return None

        found = await self._lookup(
            query=query,
            login_username=login_username,
            precomputed_embedding=precomputed_embedding,
            dims={"kb_version": kb_version, "top_k": top_k},
            threshold=self.retrieve_threshold,
            ttl_override=None,
            metric_prefix=self.METRIC_PREFIX,
        )
        if not found:
            return None

        payload, score = found
        docs = unpack_docs(payload.get("docs"))
        if docs is None:
            # 载荷损坏（如手工改过集合结构）→ 按 miss 处理，回落真实检索
            logger.warning(
                "检索缓存载荷不可还原，按未命中处理（回落真实检索）"
            )
            m = _metrics()
            if m is not None:
                m.incr(f"{self.METRIC_PREFIX}.reject.corrupt")
            return None

        entry = RetrievalEntry(
            query_text=payload.get("query_text", ""),
            context_text=payload.get("context_text", "") or "",
            docs=docs,
            doc_ids=list(payload.get("doc_ids") or []),
            score=score,
            login_username=payload.get("login_username"),
            created_at=payload.get("created_at", ""),
            hit_count=payload.get("hit_count", 0) + 1,
            kb_version=payload.get("kb_version"),
            top_k=payload.get("top_k"),
            entity_tags=list(payload.get("entity_tags") or []),
        )
        logger.info(
            f"检索缓存命中: score={score:.4f}, docs={len(docs)}, "
            f"kb_version={entry.kb_version}, top_k={entry.top_k}, "
            f"hit_count={entry.hit_count}"
        )
        return entry

    # -------------------------------------------------------------------------
    # 写入
    # -------------------------------------------------------------------------

    async def put(
        self,
        query: str,
        query_embedding: List[float],
        docs: List[Any],
        context_text: str,
        login_username: Optional[str] = None,
        kb_version: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> bool:
        """
        写检索缓存（best-effort）。

        :param docs: 真实检索返回的 Document 列表（空列表不缓存）
        :param context_text: 与 docs 对应的格式化上下文（必须与重跑结果一致）
        :param kb_version: 知识库内容版本
        :param top_k: 检索 top_k
        :return: 是否写入成功
        """
        if not self.enabled:
            return False

        packed = pack_docs(docs, self.max_docs, self.max_doc_chars)
        if packed is None:
            m = _metrics()
            if m is not None:
                m.incr(f"{self.METRIC_PREFIX}.skip.oversize")
            return False

        await self._write(
            query=query,
            query_embedding=query_embedding,
            extra_payload={
                "context_text": context_text,
                "doc_ids": [d["id"] for d in packed],
                "docs": packed,
            },
            login_username=login_username,
            dims={"kb_version": kb_version, "top_k": top_k},
        )
        m = _metrics()
        if m is not None:
            m.incr(f"{self.METRIC_PREFIX}.put")
        logger.debug(
            f"检索缓存写入: user={login_username or '__anonymous__'}, "
            f"docs={len(packed)}, query='{query[:40]}...'"
        )
        return True
