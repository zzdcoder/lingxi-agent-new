"""
企业级混合检索模块 (Hybrid Retriever)

设计目标：
    结合 Qdrant 稀疏向量（text-embedding-v4 关键词检索）与稠密向量（语义检索），
    通过 RRF (Reciprocal Rank Fusion) 算法融合多路结果，解决单一检索方式的局限性：
    - 稀疏向量擅长精确匹配关键词（如产品型号、专有名词、身份证号）
    - 稠密向量擅长语义理解（如同义词、近义表达、上下文推理）
    - RRF 融合不依赖分数绝对值，只利用排名的相对位置，天然适配不同检索方式的分数尺度差异

架构组成：
    1. Qdrant 稀疏向量: 由 text-embedding-v4 一次调用双输出生成，写入时随稠密向量
       一并存入 Qdrant（命名向量 "sparse"）；
    2. HybridRetriever: 编排 稀疏 + 稠密 两路检索，执行 RRF 融合，输出最终排序结果。

设计演进（相对旧版内存 BM25 索引）：
    - 不再维护独立的 BM25 索引文件与全量重建流程（索引持久化/同步机制一并移除）；
    - 稀疏向量与稠密向量在文档写入时一并落库，检索完全由 Qdrant 完成；
    - 文档更新时按 doc_id 先删后插（Qdrant 删点即删全部向量），无需重建索引。

依赖说明：
    无需额外分词/稀疏算法依赖，稀疏向量由 text-embedding-v4 直接生成。
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langsmith import traceable
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter
from qdrant_client.models import SparseVector

from embeddings.embedding_deal import SPARSE_VECTOR_NAME

logger = logging.getLogger(__name__)

# =============================================================================
# 常量与默认配置
# =============================================================================

DEFAULT_RRF_K: int = 60
"""RRF 公式中的平滑常数 k，论文推荐值为 60。

k 越大，低排名文档获得的分数衰减越慢，等于给后排文档更多"翻身"机会；
k 越小，排名靠前的文档优势越明显。60 是 Cormack 等人在 TREC 实验中验证的通用最优值。
"""

DEFAULT_SPARSE_TOPK: int = 50
"""稀疏向量单路检索召回数量。由于 RRF 依赖排名而非分数，每路都需要召回足够多
的候选（通常取最终 top_k 的 3~5 倍），否则融合时会因候选池太小而损失精度。"""

DEFAULT_VECTOR_TOPK: int = 50
"""稠密向量单路检索召回数量，理由同上。"""


# =============================================================================
# 工具函数
# =============================================================================

def _compute_doc_id(doc: Document, key: str = "chunk_id") -> str:
    """
    为 Document 计算稳定唯一标识。

    策略（按优先级）：
        1. 如果 metadata 中存在指定的 key（默认 chunk_id），直接复用；
        2. 否则计算 page_content 的 MD5 摘要作为兜底标识。

    为什么用 MD5 而非随机 UUID？
        - 相同内容总是生成相同 ID，天然去重；
        - 在 RRF 融合时，两路检索召回同一内容能正确合并得分。

    注意：如果两份不同文档的内容恰好完全一致（如两个 FAQ 问答答案相同），
    它们会被视为同一个文档。此时建议在写入向量库时显式注入 chunk_id。
    """
    if key in doc.metadata and doc.metadata[key]:
        return str(doc.metadata[key])
    # MD5 生成 32 位十六进制字符串，足够唯一且长度固定
    return hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()


# =============================================================================
# Cross-Encoder 重排序器
# =============================================================================

class CrossEncoderReranker:
    """
    基于 Cross-Encoder 的重排序器（使用 sentence-transformers）。

    设计目标：
        在 稀疏 + 稠密 混合检索完成初筛后，使用精度更高的 Cross-Encoder 模型
        对候选文档进行精细重排，显著提升 Top-K 结果的相关性。

    为什么需要重排序？
        - 稀疏向量和稠密向量都是"双塔"架构：查询和文档分别编码，相似度计算
          发生在向量空间，会损失细粒度交互信息；
        - Cross-Encoder 将查询和文档拼接后一起输入 Transformer，通过
          self-attention 捕捉词级别交互，相关性判断更精准；
        - 业界实践（如 Bing、Google）普遍采用"召回 + 重排"的两阶段架构。

    模型推荐：
        - BAAI/bge-reranker-large（默认）：中文英文均衡，企业知识库场景表现优秀
        - BAAI/bge-reranker-base：速度更快，精度略低，适合延迟敏感场景

    性能边界：
        - 重排序是计算密集型操作，耗时与候选文档数成正比；
        - 建议在 RRF 融合后只对 top 50~100 个候选做重排，而非全量；
        - 首次加载模型时会有显存/内存占用（bge-reranker-large 约 1.3GB）。
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 16,
    ):
        """
        初始化 Cross-Encoder 重排序器。

        Args:
            model_name: HuggingFace 模型名称或本地路径。
                        默认 "BAAI/bge-reranker-base"。
            device: 运行设备，None 时自动选择（cuda > mps > cpu）。
            max_length: 输入最大 token 长度，超过会被截断。
            batch_size: 推理批大小，根据 GPU 显存调整。
        """
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size

        self._model = None

    def _lazy_load(self) -> None:
        """惰性加载 CrossEncoder 模型（首次调用 rerank 时触发）。"""
        if self._model is not None:
            return

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                f"CrossEncoderReranker 导入 sentence_transformers 失败。"
                f"原始错误: {exc}"
            ) from exc

        logger.info(
            f"正在加载 Cross-Encoder 重排序模型: {self.model_name}"
        )
        start_time = time.time()

        # device 为空字符串/None 时，让 CrossEncoder 自动选择
        kwargs = {"max_length": self.max_length}
        if self.device:
            kwargs["device"] = self.device

        self._model = CrossEncoder(
            self.model_name,
            **kwargs,
        )

        elapsed = time.time() - start_time
        logger.info(f"Cross-Encoder 模型加载完成，耗时: {elapsed:.2f}s")

    @traceable(name="cross_encoder_rerank", tags=["rerank", "cross-encoder"])
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
    ) -> List[Document]:
        """
        对候选文档执行 Cross-Encoder 重排序。

        Args:
            query: 用户查询字符串。
            documents: 候选 Document 列表。
            top_k: 重排序后返回的文档数量。None 表示返回全部。

        Returns:
            按 Cross-Encoder 相关性分数降序排列的 Document 列表。
        """
        if not documents:
            return []

        self._lazy_load()
        top_k = top_k or len(documents)

        start_time = time.time()
        logger.info(
            f"Cross-Encoder 重排序开始，候选文档: {len(documents)}, "
            f"batch_size={self.batch_size}"
        )

        # 构建 (query, doc) 对并批量推理
        # 截断文档内容至 128 字符，大幅减少 tokenization 开销（原 256）
        pairs = [[query, doc.page_content[:256]] for doc in documents]
        scores = self._model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        # 按分数降序排列
        scored = list(zip(documents, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        elapsed = time.time() - start_time
        if scored:
            logger.info(
                f"Cross-Encoder 重排序完成，耗时: {elapsed:.3f}s, "
                f"top1_score={scored[0][1]:.4f}"
            )
        else:
            logger.info(f"Cross-Encoder 重排序完成，耗时: {elapsed:.3f}s")

        # 记录前几条结果的分数（方便调试）
        for idx, (doc, score) in enumerate(scored[:5], 1):
            preview = doc.page_content[:60].replace("\n", " ")
            logger.debug(f"  [重排结果 {idx}] score={score:.4f}, text={preview}...")

        return [doc for doc, _ in scored[:top_k]]

    async def arerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
    ) -> List[Document]:
        """
        异步版本的 Cross-Encoder 重排序。

        将同步推理放入线程池执行，避免阻塞事件循环。
        """
        import asyncio
        return await asyncio.to_thread(self.rerank, query, documents, top_k)


# =============================================================================
# 混合检索器 (Hybrid Retriever)
# =============================================================================

class HybridRetriever:
    """
    企业级混合检索器。

    工作流程：
        1. 【并行召回】同时执行 Qdrant 稀疏向量（text-embedding-v4）检索和稠密向量语义检索；
        2. 【RRF 融合】将两路结果按排名位置融合为一个统一排序；
        3. 【去重截断】去除重复文档，截取 top_k 返回。

    RRF 算法原理（Reciprocal Rank Fusion）：
        对于某篇文档 d，它在第 i 路检索中的排名为 r_i(d)（从 1 开始），
        则其 RRF 得分为：

            Score(d) = Σ_i [ 1 / (k + r_i(d)) ]

        其中 k 为平滑常数（默认 60）。

        为什么 RRF 比直接加权求和更好？
            - 不同检索方式的分数尺度完全不同（稀疏向量与稠密向量相似度范围不同），
              直接加权相当于让分数尺度大的一路主导结果；
            - RRF 只关心"排名第几"，天然消除了分数尺度的差异；
            - 实现简单，无需训练，零超参（k=60 是论文通用值）。

    适用场景：
        - 用户查询中包含明确的关键词（如"退款政策"、"2026年预算"），稀疏向量能精准命中；
        - 用户使用口语化、近义词表达（如"怎么退钱"、"今年的花费计划"），稠密向量能捕捉语义；
        - 混合检索在绝大多数真实场景下 Recall@K 显著优于单路检索。
    """

    def __init__(
        self,
        vectorstore: QdrantVectorStore,
        rrf_k: int = DEFAULT_RRF_K,
        sparse_weight: float = 1.0,
        vector_weight: float = 1.0,
        reranker: Optional[CrossEncoderReranker] = None,
        rerank_top_k: Optional[int] = None,
        embedder: Optional[Any] = None,
    ):
        """
        初始化混合检索器。

        Args:
            vectorstore: LangChain 的 QdrantVectorStore 实例，负责稠密向量语义检索，
                         其内部的 QdrantClient 同时承担稀疏向量检索。
            rrf_k: RRF 平滑常数，默认 60。
            sparse_weight: 稀疏路得分的权重乘数（默认 1.0）。
                         如果你发现业务中关键词匹配更重要，可提高到 1.2~1.5；
                         如果语义理解更重要，可降低到 0.8。
            vector_weight: 稠密路得分的权重乘数（默认 1.0）。与 sparse_weight 配合使用。
            reranker: Cross-Encoder 重排序器实例。为 None 时不启用重排序。
            rerank_top_k: 送入重排序器的候选文档数量。
                          默认为 None（取最终 top_k 的 5 倍，或至少 50）。
            embedder: 查询稀疏向量编码器（需提供 embed_query_with_sparse 方法）。
                      为 None 时从 vectorstore.embeddings 兜底获取。
        """
        self._vectorstore: QdrantVectorStore = vectorstore
        self._rrf_k: int = rrf_k
        self._sparse_weight: float = sparse_weight
        self._vector_weight: float = vector_weight
        self._reranker: Optional[CrossEncoderReranker] = reranker
        self._rerank_top_k: Optional[int] = rerank_top_k
        self._embedder: Optional[Any] = embedder or getattr(
            getattr(vectorstore, "embeddings", None), None
        )

    # -------------------------------------------------------------------------
    # 稀疏向量（关键词）检索
    # -------------------------------------------------------------------------

    def _sparse_retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_SPARSE_TOPK,
        filter_obj: Optional[Filter] = None,
        precomputed_sparse: Optional[SparseVector] = None,
    ) -> List[Document]:
        """
        使用 Qdrant 稀疏向量执行关键词召回。

        查询稀疏向量由 text-embedding-v4 生成（与写入侧算法一致），
        交由 Qdrant 原生稀疏索引检索，无需维护任何内存索引。

        Args:
            query: 用户查询字符串。
            top_k: 返回文档数量。建议取最终需求 top_k 的 3~5 倍，为 RRF 融合留出候选空间。
            filter_obj: Qdrant 过滤条件（权限等），与稠密路共用，保证过滤逻辑一致。
            precomputed_sparse: 预计算的查询稀疏向量（由上层一次性生成，避免重复 API 调用）。
                                为 None 时通过 embedder 兜底计算。

        Returns:
            按稀疏向量得分降序排列的 Document 列表。
        """
        query_sparse = precomputed_sparse
        if query_sparse is None:
            # 兜底：未预计算时由 embedder 一次调用生成查询稀疏向量
            if self._embedder is None or not hasattr(self._embedder, "embed_query_with_sparse"):
                logger.warning("无可用稀疏编码器且未提供预计算稀疏向量，跳过稀疏召回")
                return []
            _, query_sparse = self._embedder.embed_query_with_sparse(query)

        if not query_sparse.indices:
            logger.info("稀疏查询为空（无有效 token），跳过稀疏召回")
            return []

        client: QdrantClient = self._vectorstore.client
        response = client.query_points(
            collection_name=self._vectorstore.collection_name,
            query=query_sparse,
            using=SPARSE_VECTOR_NAME,
            query_filter=filter_obj,
            limit=top_k,
            with_payload=True,
        )

        docs = []
        for point in response.points:
            payload = point.payload or {}
            page_content = payload.get("page_content", "")
            # 与 langchain_qdrant 的 Document 构造保持一致：
            # payload 除 page_content 外整体作为 metadata（含嵌套 metadata 字段）
            metadata = {k: v for k, v in payload.items() if k != "page_content"}
            docs.append(Document(page_content=page_content, metadata=metadata))

        logger.info(
            f"稀疏向量召回完成，查询='{query[:50]}...', 召回 {len(docs)} 个"
        )
        return docs

    # -------------------------------------------------------------------------
    # 核心检索接口
    # -------------------------------------------------------------------------

    @traceable(name="hybrid_retrieve", tags=["retrieval", "hybrid"])
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        sparse_top_k: int = DEFAULT_SPARSE_TOPK,
        vector_top_k: int = DEFAULT_VECTOR_TOPK,
        filter_obj: Optional[Filter] = None,
        precomputed_embedding: Optional[List[float]] = None,
        precomputed_sparse: Optional[SparseVector] = None,
    ) -> List[Document]:
        """
        执行混合检索并返回融合后的结果。

        这是本类的主入口，企业级使用时应直接调用此方法。

        Args:
            query: 用户查询字符串。
            top_k: 最终返回的文档数量。
            sparse_top_k: 稀疏向量单路召回数量。建议 >= top_k * 3。
            vector_top_k: 稠密向量单路召回数量。建议 >= top_k * 3。
            filter_obj: Qdrant 过滤条件（权限、状态等），两路共用，
                        使稀疏与稠密检索在过滤逻辑上保持一致。
            precomputed_embedding: 可选的预计算 query embedding，直接传入可跳过
                                   稠密路内部的 embedding 计算。
            precomputed_sparse: 可选的预计算 query 稀疏向量（与稠密向量同一模型同一次
                                调用生成），直接传入可跳过稀疏路的 API 调用。

        Returns:
            按 RRF 得分降序排列的 Document 列表，长度 <= top_k。
        """
        start_time = time.time()
        logger.info(
            f"混合检索开始，查询='{query[:60]}...', top_k={top_k}, "
            f"RRF k={self._rrf_k}"
        )

        # 步骤 1: 并行执行两路检索
        # 注：稀疏路是查询向量生成 + Qdrant 查询，稠密路涉及 embedding 计算与网络 IO。
        # 若提供了 precomputed_embedding / precomputed_sparse，两路均跳过内部向量生成。
        sparse_results = self._sparse_retrieve(
            query, top_k=sparse_top_k, filter_obj=filter_obj,
            precomputed_sparse=precomputed_sparse,
        )
        if precomputed_embedding is not None:
            vector_results = self._vectorstore.similarity_search_by_vector(
                precomputed_embedding, k=vector_top_k, filter=filter_obj
            )
        else:
            vector_results = self._vectorstore.similarity_search(
                query, k=vector_top_k, filter=filter_obj
            )

        logger.info(
            f"两路召回完成: Sparse={len(sparse_results)}, Vector={len(vector_results)}"
        )

        # 步骤 2: RRF 融合
        fused_results = self._rrf_fusion(sparse_results, vector_results)

        # 步骤 3: Cross-Encoder 重排序（若启用）
        if self._reranker and fused_results:
            rerank_candidates = self._rerank_top_k or max(top_k * 3, 20)
            candidates = fused_results[:rerank_candidates]
            fused_results = self._reranker.rerank(
                query=query,
                documents=candidates,
                top_k=top_k,
            )
            logger.info(f"重排序后返回 {len(fused_results)} 个结果")
        else:
            # 未启用重排序时直接截断
            fused_results = fused_results[:top_k]

        final_results = fused_results

        elapsed = time.time() - start_time
        logger.info(
            f"混合检索完成，最终返回 {len(final_results)} 个结果，"
            f"总耗时: {elapsed:.3f}s"
        )

        # 记录最终结果的 ID，方便调试
        for idx, doc in enumerate(final_results, 1):
            doc_id = _compute_doc_id(doc, "chunk_id")
            preview = doc.page_content[:80].replace("\n", " ")
            logger.debug(f"  [融合结果 {idx}] id={doc_id}, text={preview}...")

        return final_results

    # -------------------------------------------------------------------------
    # RRF 融合算法
    # -------------------------------------------------------------------------

    @traceable(name="rrf_fusion", tags=["fusion", "rrf"])
    def _rrf_fusion(
        self,
        sparse_results: List[Document],
        vector_results: List[Document],
    ) -> List[Document]:
        """
        执行 Reciprocal Rank Fusion 融合。

        实现步骤：
            1. 为每路结果中的文档分配排名（从 1 开始）；
            2. 计算每篇文档的 RRF 得分：score = Σ(weight_i / (k + rank_i))；
            3. 按得分降序排列，去重后返回。

        Args:
            sparse_results: 稀疏向量检索结果（已按得分降序排列）。
            vector_results: 稠密向量检索结果（已按相似度降序排列）。

        Returns:
            融合后按 RRF 得分降序排列的 Document 列表（已去重）。
        """
        # 数据结构: {doc_id: {"score": float, "doc": Document, "sources": set}}
        scores: Dict[str, Dict[str, Any]] = {}

        def _register(results: List[Document], weight: float, source_name: str) -> None:
            """将一路检索结果注册到得分表中。"""
            for rank, doc in enumerate(results, start=1):
                doc_id = _compute_doc_id(doc, "chunk_id")
                if doc_id not in scores:
                    scores[doc_id] = {
                        "score": 0.0,
                        "doc": doc,
                        "sources": set(),
                    }
                # RRF 核心公式
                contribution = weight * (1.0 / (self._rrf_k + rank))
                scores[doc_id]["score"] += contribution
                scores[doc_id]["sources"].add(source_name)

        # 注册稀疏路
        if sparse_results:
            _register(sparse_results, self._sparse_weight, "sparse")

        # 注册稠密路
        if vector_results:
            _register(vector_results, self._vector_weight, "vector")

        # 按 RRF 总分降序排列
        sorted_items = sorted(
            scores.items(),
            key=lambda item: item[1]["score"],
            reverse=True,
        )

        # 记录融合详情日志（方便后续调优时分析每篇文档的来源和得分）
        for doc_id, info in sorted_items[:20]:
            logger.debug(
                f"RRF doc_id={doc_id}, score={info['score']:.5f}, "
                f"sources={info['sources']}"
            )

        # 提取去重后的 Document 列表
        return [info["doc"] for _, info in sorted_items]

    # -------------------------------------------------------------------------
    # 异步检索接口（适配 asyncio 环境）
    # -------------------------------------------------------------------------

    @traceable(name="hybrid_retrieve_async", tags=["retrieval", "hybrid", "async"])
    async def aretrieve(
        self,
        query: str,
        top_k: int = 10,
        sparse_top_k: int = DEFAULT_SPARSE_TOPK,
        vector_top_k: int = DEFAULT_VECTOR_TOPK,
        filter_obj: Optional[Filter] = None,
        precomputed_embedding: Optional[List[float]] = None,
        precomputed_sparse: Optional[SparseVector] = None,
    ) -> List[Document]:
        """
        异步版本的混合检索。

        与 retrieve 的区别：
            - 两路检索均放入 asyncio.to_thread 避免阻塞事件循环；
            - 支持传入 precomputed_embedding / precomputed_sparse 跳过两路内部的向量生成。

        如果你的服务使用 FastAPI + Uvicorn，建议调用此方法而非同步的 retrieve。
        """
        import asyncio

        start_time = time.time()
        logger.info(f"异步混合检索开始，查询='{query[:60]}...'")

        # 稀疏路与稠密路并发执行（均放入线程池，避免阻塞主事件循环）
        sparse_task = asyncio.to_thread(
            self._sparse_retrieve, query, sparse_top_k, filter_obj, precomputed_sparse
        )
        if precomputed_embedding is not None:
            vector_task = asyncio.to_thread(
                self._vectorstore.similarity_search_by_vector,
                precomputed_embedding,
                vector_top_k,
                filter=filter_obj,
            )
        else:
            vector_task = asyncio.to_thread(
                self._vectorstore.similarity_search,
                query,
                vector_top_k,
                filter=filter_obj,
            )

        # 并发等待两路结果
        sparse_results, vector_results = await asyncio.gather(
            sparse_task, vector_task
        )

        logger.info(
            f"异步两路召回完成: Sparse={len(sparse_results)}, Vector={len(vector_results)}"
        )

        fused_results = self._rrf_fusion(sparse_results, vector_results)

        # Cross-Encoder 重排序（若启用）
        if self._reranker and fused_results:
            rerank_candidates = self._rerank_top_k or max(top_k * 3, 20)
            candidates = fused_results[:rerank_candidates]
            fused_results = await self._reranker.arerank(
                query=query,
                documents=candidates,
                top_k=top_k,
            )
            logger.info(f"异步重排序后返回 {len(fused_results)} 个结果")
        else:
            fused_results = fused_results[:top_k]

        final_results = fused_results

        elapsed = time.time() - start_time
        logger.info(f"异步混合检索完成，返回 {len(final_results)} 个，耗时: {elapsed:.3f}s")
        return final_results
