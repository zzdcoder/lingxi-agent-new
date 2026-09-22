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

**§18 治理改造（缓存键升维 + 命中校验）**

命中答案缓存 = **跳过整条执行链路**（不检索、不生成、不执行工具、不留审计），
因此本模块在此前只按「query 向量 + 用户名」匹配是不够的，补齐两道防线：

1. **键升维**：payload 增加 intent / model / prompt_version / kb_version /
   schema_version / entity_tags，查询时用 Qdrant `Filter(must=...)` **强过滤**。
   任一维度不匹配即 miss —— 版本 bump 等价于「逻辑失效」，无需删数据，
   且可回滚（把版本号改回去即可）。
2. **命中校验**：阈值收紧（答案缓存档 0.95）+ 实体一致性校验。短中文文本在
   0.92 阈值下极易漂移（「删除张三」与「删除李四」余弦常在 0.95+），
   仅靠向量相似度不足以证明「命中 == 重跑一次的结果」。

是否允许查缓存由**意图**决定（见 `agent/intent_gate.cache_policy_for`），
本模块只负责「给定维度下能否命中」，不做准入决策。
"""

import difflib
import logging
import re
import uuid
from dataclasses import dataclass, field
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
    # §18 新增维度（用于日志/审计溯源）
    intent: Optional[str] = None        # 产生该答案时的意图
    kb_version: Optional[str] = None    # 知识库内容版本
    entity_tags: List[str] = field(default_factory=list)  # 实体标签（P1 定向失效用）


# =============================================================================
# 实体抽取与一致性校验（§18.3）
# =============================================================================
#
# 目的：拦住「向量很近但不是同一个问题」的命中。
# 典型反例：把用户表里张三这条记录删除 / 把用户表里李四这条记录删除
# —— 余弦相似度常在 0.95 以上，但答案完全不同。

_NUM_RE = re.compile(r"\d+")                      # 阿拉伯数字（数量 / ID / 年份）
_CN_NUM_RE = re.compile(
    r"[零〇一二三四五六七八九十百千万两]+"
)                                                 # 中文数字（十条 / 二十条 / 三年）
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")  # 表名 / 字段名 / 英文实体
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")          # 汉字（用于字符级差异比对）
_TAG_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}|\d{2,}|[\u4e00-\u9fff]{2,4}")

# 中文字符级差异容忍比例：超过则认为换了实体/换了对象
MAX_DIFF_RATIO = 0.12


def extract_entity_tags(text: str, limit: int = 20) -> List[str]:
    """
    抽取用于**定向失效**的实体标签（表名 / 数字 / 中文片段）。

    只做粗抽取（不上 NLP），目的是在 P1「写操作后按实体失效」时提供匹配素材；
    标签不参与命中判定（命中判定用 `entity_consistent`）。
    """
    if not text:
        return []
    tags = {m.group(0).lower() for m in _TAG_RE.finditer(text)}
    return sorted(tags)[:limit]


def entity_consistent(
    query: str, cached_query: str, max_diff_ratio: float = MAX_DIFF_RATIO
) -> bool:
    """
    关键实体一致性校验：确认两条 query 指向**同一个对象**。

    三道判据，任一不通过即判不一致：
    1. 数字集合必须相同（数量 / ID / 年份变了就不是同一个问题）；
    2. 英文 token 集合必须相同（表名 / 字段名 / 英文实体变了同理）；
    3. 汉字字符级差异比例不得超过阈值（换人名、换对象会被拦住）。

    :param query: 当前问题
    :param cached_query: 缓存里的问题原文
    :param max_diff_ratio: 汉字差异容忍比例
    :return: 是否可视为同一个问题
    """
    q, c = query or "", cached_query or ""
    if not q or not c:
        return True

    # 1. 数字集合（阿拉伯数字 + 中文数字）
    #    中文数字单独处理：字符级差异比对分不出「十条」与「二十条」
    #    （只多一个「二」字，差异比例低于阈值），必须按集合比对。
    if set(_NUM_RE.findall(q)) != set(_NUM_RE.findall(c)):
        return False
    if set(_CN_NUM_RE.findall(q)) != set(_CN_NUM_RE.findall(c)):
        return False
    # 2. 英文 token 集合（大小写不敏感）
    if {w.lower() for w in _WORD_RE.findall(q)} != {
        w.lower() for w in _WORD_RE.findall(c)
    }:
        return False
    # 3. 汉字字符级差异
    a, b = _CJK_RE.findall(q), _CJK_RE.findall(c)
    if not a or not b:
        return True
    diff = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        if tag == "equal":
            continue
        diff += max(i2 - i1, j2 - j1)
    return diff / max(1, min(len(a), len(b))) <= max_diff_ratio


def _metrics():
    """惰性获取指标单例（观测不可用时静默返回 None）。"""
    try:
        from agent.observability import metrics

        return metrics
    except Exception:
        return None


def _entity_check_enabled() -> bool:
    """实体一致性校验开关（配置缺失时默认开启）。"""
    try:
        from core.config import settings

        return bool(getattr(settings, "cache_entity_check_enabled", True))
    except Exception:
        return True


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
        answer_threshold=settings.cache_answer_threshold,
        schema_version=settings.cache_schema_version,
    )
    _semantic_cache.initialize()
    logger.info(
        f"语义缓存初始化完成: collection={settings.cache_collection_name}, "
        f"threshold={settings.cache_similarity_threshold}, "
        f"answer_threshold={settings.cache_answer_threshold}, "
        f"schema_version={settings.cache_schema_version}, "
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

    **§18.13 分层**：本类是**答案缓存层**（命中即跳过整条链路，阈值 0.95）。
    检索缓存层（`rag.retrieval_cache.RetrievalCache`）继承本类并覆盖
    `DIM_KEYS` / `SHARE_ANONYMOUS` / 阈值与载荷构造，复用 `_lookup` / `_write` /
    `cleanup_expired` / `_evict_if_needed` 等公共内核。
    """

    # 参与强过滤的维度键（子类可覆盖）
    DIM_KEYS: tuple = ("intent", "model", "kb_version", "prompt_version")
    # 匿名条目是否可被具名用户命中（检索缓存关闭，理由见 _build_match_filter）
    SHARE_ANONYMOUS: bool = True
    # 指标前缀（子类覆盖，用于分层观测）
    METRIC_PREFIX: str = "cache.get"
    # 日志中的层名（子类覆盖）
    LAYER_LABEL: str = "语义缓存"

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
        answer_threshold: float = 0.95,
        schema_version: str = "1",
    ):
        self.client = qdrant_client
        self.embedding = embedding_model
        self.collection_name = collection_name
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.vector_dim = vector_dim
        self.enabled = enabled
        # §18：答案缓存阈值（命中即跳过整条链路，比一般相似度要求更严）
        self.answer_threshold = answer_threshold or similarity_threshold
        # 缓存结构版本：bump 后旧条目自动失配（逻辑失效，可回滚）
        self.schema_version = str(schema_version or "1")

    # 需要建 INTEGER 索引的维度字段（其余按 KEYWORD 处理）
    INT_INDEX_FIELDS: tuple = ()

    def _ensure_payload_indexes(self) -> None:
        """
        为过滤字段建 payload 索引（best-effort）。

        Qdrant 在没有索引的字段上做 `Filter` 会退化为**全量扫描**；
        两层缓存都是「每请求一次带 4~6 个维度条件的近邻查询」，
        没有索引时缓存查询本身会比真实检索还慢。索引已存在时会抛错，
        这里静默忽略（debug 级）。

        字段类型：`INT_INDEX_FIELDS` 中的用 INTEGER（如检索缓存的 `top_k`），
        其余用 KEYWORD。
        """
        try:
            from qdrant_client.http.models import PayloadSchemaType
        except Exception:
            return

        fields = ("schema_version", "login_username", *self.DIM_KEYS)
        for field in fields:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=(
                        PayloadSchemaType.INTEGER
                        if field in self.INT_INDEX_FIELDS
                        else PayloadSchemaType.KEYWORD
                    ),
                )
            except Exception as e:
                logger.debug(f"payload 索引 {field} 已存在或创建失败（忽略）: {e}")

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
                logger.info(f"{self.LAYER_LABEL}集合已创建: {self.collection_name}")
            else:
                info = self.client.get_collection(self.collection_name)
                logger.info(
                    f"{self.LAYER_LABEL}集合已存在: {self.collection_name}, "
                    f"points_count={info.points_count}"
                )

            # 建 payload 索引：无索引时维度过滤退化为全量扫描（best-effort）
            self._ensure_payload_indexes()
        except Exception as e:
            logger.error(f"语义缓存集合初始化失败: {e}")
            self.enabled = False

    # -------------------------------------------------------------------------
    # 核心操作：查询缓存
    # -------------------------------------------------------------------------

    def _build_match_filter(self, effective_user: str, dims: dict) -> Filter:
        """
        构造查询过滤条件（§18.2 键升维 / §18.13 检索缓存复用）。

        结构：`must[schema_version, 各维度..., 权限]`

        - **维度条件**：`schema_version` 恒参与；`self.DIM_KEYS` 中的维度在调用方
          给出时参与。任一不匹配即 miss —— 这就是「版本 bump = 逻辑失效」的实现：
          不必删数据，改版本号即可让旧条目失配，且可回滚。
        - **权限**：
          - `SHARE_ANONYMOUS=True`（答案缓存，沿用历史口径）：本人 **或** 公共
            （`login_username == __anonymous__`），用嵌套 `should` 表达；
          - `SHARE_ANONYMOUS=False`（检索缓存）：只匹配本人，匿名条目不参与共享。
            见 `rag/retrieval_cache.py` 说明 —— 命中比等价更严格只会造成 miss，
            不会造成越权。

        :param effective_user: 归一化后的用户名（匿名固定为 __anonymous__）
        :param dims: 维度字典，键取自 `self.DIM_KEYS`
        """
        must: list = [
            FieldCondition(
                key="schema_version",
                match=MatchValue(value=self.schema_version),
            )
        ]
        for key in self.DIM_KEYS:
            value = dims.get(key)
            if value not in (None, ""):
                must.append(FieldCondition(key=key, match=MatchValue(value=value)))

        if self.SHARE_ANONYMOUS:
            must.append(
                Filter(
                    should=[
                        FieldCondition(
                            key="login_username",
                            match=MatchValue(value=effective_user),
                        ),
                        FieldCondition(
                            key="login_username",
                            match=MatchValue(value="__anonymous__"),
                        ),
                    ]
                )
            )
        else:
            must.append(
                FieldCondition(
                    key="login_username",
                    match=MatchValue(value=effective_user),
                )
            )
        return Filter(must=must)

    # -------------------------------------------------------------------------
    # 公共内核：供答案缓存（本类）与检索缓存（子类）复用
    # -------------------------------------------------------------------------

    async def _lookup(
        self,
        query: str,
        login_username: Optional[str],
        precomputed_embedding: Optional[List[float]],
        dims: dict,
        threshold: float,
        ttl_override: Optional[int],
        metric_prefix: str,
    ) -> Optional[tuple[dict, float]]:
        """
        向量近邻查找 + 阈值 / TTL / 实体一致性三道校验。

        返回 `(payload, score)`；未命中或异常时返回 `None`。
        指标按 `metric_prefix` 分层上报（答案缓存 `cache.get` / 检索缓存
        `cache.retrieve`），便于分别观测两层的命中率与拒绝原因。

        :param dims: 参与强过滤的维度（键取自 `self.DIM_KEYS`）
        :param threshold: 本层命中阈值（答案层 0.95 / 检索层 0.90）
        :param ttl_override: 本轮 TTL（None 时用条目自带 `ttl_class`）
        :param metric_prefix: 指标前缀
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

            # 2. 构建过滤条件（缓存维度 must + 权限）
            effective_user = login_username or "__anonymous__"
            filter_obj = self._build_match_filter(effective_user, dims)

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
                logger.debug(f"{metric_prefix} 未命中（无结果）")
                m = _metrics()
                if m is not None:
                    m.incr(f"{metric_prefix}.miss")
                return None

            best_match = results[0]

            # 4. 判断相似度阈值
            if best_match.score < threshold:
                logger.debug(
                    f"{metric_prefix} 未命中（score={best_match.score:.4f} "
                    f"< threshold={threshold}）"
                )
                m = _metrics()
                if m is not None:
                    m.incr(f"{metric_prefix}.miss")
                    m.incr(f"{metric_prefix}.reject.threshold")
                return None

            # 5. 检查 TTL 过期
            payload = best_match.payload or {}
            created_at_str = payload.get("created_at", "")
            effective_ttl = ttl_override
            if not effective_ttl:
                try:
                    effective_ttl = int(payload.get("ttl_class") or 0) or self.ttl_seconds
                except (TypeError, ValueError):
                    effective_ttl = self.ttl_seconds
            if created_at_str:
                try:
                    created_at = datetime.fromisoformat(created_at_str)
                    if datetime.now(timezone.utc) - created_at > timedelta(
                        seconds=effective_ttl
                    ):
                        logger.debug(f"{metric_prefix} 未命中（已过期）")
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
                        m = _metrics()
                        if m is not None:
                            m.incr(f"{metric_prefix}.miss")
                            m.incr(f"{metric_prefix}.reject.expired")
                        return None
                except (ValueError, TypeError):
                    pass

            # 6. 实体一致性校验（§18.3）：拦住「向量很近但不是同一个问题」
            cached_query = payload.get("query_text", "")
            if _entity_check_enabled() and not entity_consistent(query, cached_query):
                logger.info(
                    f"{metric_prefix} 未命中（实体校验不通过）: "
                    f"current='{query[:30]}' cached='{cached_query[:30]}'"
                )
                m = _metrics()
                if m is not None:
                    m.incr(f"{metric_prefix}.miss")
                    m.incr(f"{metric_prefix}.reject.entity")
                return None

            # 7. 更新 hit_count（best-effort，不阻塞返回）
            try:
                old_hit_count = payload.get("hit_count", 0)
                self.client.set_payload(
                    collection_name=self.collection_name,
                    payload={"hit_count": old_hit_count + 1},
                    points=[best_match.id],
                )
            except Exception as e:
                logger.debug(f"更新 hit_count 失败（非致命）: {e}")

            m = _metrics()
            if m is not None:
                m.incr(f"{metric_prefix}.hit")
            return payload, best_match.score

        except Exception as e:
            logger.warning(f"{metric_prefix} 查询异常（降级为无缓存）: {e}")
            return None

    async def _write(
        self,
        query: str,
        query_embedding: List[float],
        extra_payload: dict,
        login_username: Optional[str],
        dims: dict,
        entity_tags: Optional[List[str]] = None,
        ttl_class: Optional[int] = None,
    ) -> None:
        """
        写入一条缓存条目（公共字段 + 调用方提供的层特有字段）。

        写入的维度必须与查询时的过滤维度一一对应，否则条目永远命中不了。

        :param extra_payload: 层特有字段（答案层：answer / context_preview；
                              检索层：context_text / doc_ids / scores / docs）
        :param dims: 维度字典（键取自 `self.DIM_KEYS`）
        :param entity_tags: 实体标签（P1 定向失效用；缺省时自动抽取）
        :param ttl_class: 本条目的 TTL（秒）
        """
        if not self.enabled:
            return

        try:
            effective_user = login_username or "__anonymous__"
            point_id = str(uuid.uuid4())

            payload = {
                "query_text": query,
                "login_username": effective_user,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "hit_count": 0,
                # ---- §18 缓存维度（查询时强过滤，见 _build_match_filter）----
                "schema_version": self.schema_version,
                # 实体标签：定向失效素材（P1），缺省时从 query 粗抽取
                "entity_tags": list(entity_tags or extract_entity_tags(query)),
                "ttl_class": int(ttl_class or self.ttl_seconds),
            }
            for key in self.DIM_KEYS:
                payload[key] = dims.get(key) if dims.get(key) is not None else ""
            payload.update(extra_payload or {})

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

            logger.debug(
                f"缓存写入[{self.collection_name}]: user={effective_user}, "
                f"query='{query[:40]}...'"
            )

            # 检查容量上限，触发淘汰（best-effort）
            await self._evict_if_needed()

        except Exception as e:
            logger.warning(f"缓存写入异常（非致命）: {e}")

    async def get(
        self,
        query: str,
        login_username: Optional[str] = None,
        precomputed_embedding: Optional[List[float]] = None,
        intent: Optional[str] = None,
        model: Optional[str] = None,
        kb_version: Optional[str] = None,
        prompt_version: Optional[str] = None,
        ttl_override: Optional[int] = None,
    ) -> Optional[CacheEntry]:
        """
        语义缓存查询（§18：键升维 + 阈值收紧 + 实体校验）。

        流程：
            1. 调用 embedding 模型计算 query 向量（或使用预计算向量）
            2. 构造强过滤条件（缓存维度 + 权限），搜索最相似条目（top_k=1）
            3. 判断 score >= 答案缓存阈值（0.95，严于原 0.92）
            4. 检查 TTL 过期（优先用调用方给的 ttl_override，其次条目自带的 ttl_class）
            5. 实体一致性校验（拦住「向量很近但不是同一个问题」）
            6. 命中则更新 hit_count

        :param query: 用户问题
        :param login_username: 当前登录用户名
        :param precomputed_embedding: 可选的预计算 query embedding，避免重复计算
        :param intent: 本轮意图（与缓存条目 intent 不一致即 miss）
        :param model: 生成模型名
        :param kb_version: 知识库内容版本
        :param prompt_version: 提示词版本
        :param ttl_override: 本轮 TTL（由意图准入策略给出；None 时用条目自带值）
        :return: 命中时返回 CacheEntry，否则返回 None
        """
        if not self.enabled:
            return None

        found = await self._lookup(
            query=query,
            login_username=login_username,
            precomputed_embedding=precomputed_embedding,
            dims={
                "intent": intent,
                "model": model,
                "kb_version": kb_version,
                "prompt_version": prompt_version,
            },
            threshold=self.answer_threshold,
            ttl_override=ttl_override,
            metric_prefix=self.METRIC_PREFIX,
        )
        if not found:
            return None

        payload, score = found
        entry = CacheEntry(
            query_text=payload.get("query_text", ""),
            answer=payload.get("answer", ""),
            score=score,
            login_username=payload.get("login_username"),
            created_at=payload.get("created_at", ""),
            hit_count=payload.get("hit_count", 0) + 1,
            intent=payload.get("intent"),
            kb_version=payload.get("kb_version"),
            entity_tags=list(payload.get("entity_tags") or []),
        )
        logger.info(
            f"语义缓存命中: score={score:.4f}, "
            f"intent={entry.intent}, kb_version={entry.kb_version}, "
            f"hit_count={entry.hit_count}, "
            f"cached_query='{entry.query_text[:40]}...'"
        )
        return entry

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
        intent: Optional[str] = None,
        model: Optional[str] = None,
        kb_version: Optional[str] = None,
        prompt_version: Optional[str] = None,
        entity_tags: Optional[List[str]] = None,
        ttl_class: Optional[int] = None,
    ) -> None:
        """
        将问答对写入语义缓存（§18：带上全部缓存维度）。

        写入的维度必须与查询时的过滤维度一一对应，否则条目永远命中不了。

        :param query: 用户问题原文
        :param query_embedding: 问题 embedding 向量（可复用检索阶段已计算的向量）
        :param answer: LLM 完整回答
        :param context_docs: 检索到的文档（仅存摘要，用于调试）
        :param login_username: 当前用户名（用于权限隔离）
        :param intent: 本轮意图（查询时按此过滤）
        :param model: 生成模型名
        :param kb_version: 知识库内容版本（文档变更时 bump 即逻辑失效）
        :param prompt_version: 提示词版本
        :param entity_tags: 实体标签（P1 写操作后定向失效用；缺省时自动抽取）
        :param ttl_class: 本条目的 TTL（秒），由意图准入策略给出
        """
        if not self.enabled:
            return

        # 构建 context 摘要（仅保留前 100 字，避免 payload 过大）
        context_preview = ""
        if context_docs:
            try:
                context_preview = "; ".join(
                    doc.page_content[:100] for doc in context_docs[:3]
                )
            except Exception:
                context_preview = ""

        await self._write(
            query=query,
            query_embedding=query_embedding,
            extra_payload={"answer": answer, "context_preview": context_preview},
            login_username=login_username,
            dims={
                "intent": intent,
                "model": model,
                "kb_version": kb_version,
                "prompt_version": prompt_version,
            },
            entity_tags=entity_tags,
            ttl_class=ttl_class,
        )
        logger.info(
            f"语义缓存写入: user={login_username or '__anonymous__'}, "
            f"query='{query[:40]}...', answer_len={len(answer)}"
        )

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
