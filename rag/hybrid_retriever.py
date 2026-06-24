"""
企业级混合检索模块 (Hybrid Retriever)

设计目标：
    结合 BM25 关键词检索与向量语义检索，通过 RRF (Reciprocal Rank Fusion) 算法融合多路结果，
    解决单一检索方式的局限性：
    - BM25 擅长精确匹配关键词（如产品型号、专有名词、身份证号）
    - 向量检索擅长语义理解（如同义词、近义表达、上下文推理）
    - RRF 融合不依赖分数绝对值，只利用排名的相对位置，天然适配不同检索方式的分数尺度差异

架构组成：
    1. BM25Indexer:   基于 rank-bm25 构建内存索引，支持中文 jieba 分词
    2. HybridRetriever: 编排 BM25 + 向量两路检索，执行 RRF 融合，输出最终排序结果

依赖安装：
    pip install rank-bm25 jieba

企业级特性：
    - 索引持久化（pickle 序列化到磁盘，重启后秒级恢复）
    - 全量同步机制（从 Qdrant 拉取全部数据重建 BM25 索引）
    - 增量同步机制（追加新文档时局部重建）
    - 完善的日志与异常处理
    - 类型注解与文档字符串

作者：AI Assistant
日期：2026-06-17
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import jieba
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# =============================================================================
# 常量与默认配置
# =============================================================================

DEFAULT_RRF_K: int = 60
"""RRF 公式中的平滑常数 k，论文推荐值为 60。

k 越大，低排名文档获得的分数衰减越慢，等于给后排文档更多"翻身"机会；
k 越小，排名靠前的文档优势越明显。60 是 Cormack 等人在 TREC 实验中验证的通用最优值。
"""

DEFAULT_BM25_TOPK: int = 50
"""BM25 单路检索召回数量。由于 RRF 依赖排名而非分数，每路都需要召回足够多
的候选（通常取最终 top_k 的 3~5 倍），否则融合时会因候选池太小而损失精度。"""

DEFAULT_VECTOR_TOPK: int = 50
"""向量检索单路召回数量，理由同上。"""




# =============================================================================
# 工具函数
# =============================================================================

def _default_tokenizer(text: str) -> List[str]:
    """
    默认中文分词器。

    使用 jieba.cut_for_search 而非 jieba.cut，因为：
    - cut_for_search 会对长词进行更细粒度的切分（如"中华人民共和国"会被切成
      "中华"/"人民"/"共和国"/"中华人民共和国"）
    - 检索场景下，细粒度切分能提高召回率，即使查询词只命中长词的一部分也能被检索到
    - 代价是索引体积稍大，但在企业知识库规模（万级文档）下完全可以接受

    如果你的领域有大量英文/数字混合文本（如"Python3.11 异步编程"），
    建议在此函数中增加正则清洗逻辑，去除纯标点符号 token。
    """
    return list(jieba.cut_for_search(text.strip()))


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
# BM25 索引器
# =============================================================================

@dataclass
class BM25IndexState:
    """
    BM25 索引的可序列化状态。

    rank-bm25 库内部使用 numpy 数组存储词频统计，BM25Okapi 对象本身支持 pickle。
    为了后续扩展（如增加自定义字段、版本控制），我们将其包装为 dataclass，
    而非直接序列化裸对象。
    """
    bm25: BM25Okapi
    documents: List[Document] = field(default_factory=list)
    tokenized_corpus: List[List[str]] = field(default_factory=list)
    doc_id_key: str = "chunk_id"
    version: int = 1


class BM25Indexer:
    """
    BM25 关键词检索索引器。

    职责：
        1. 维护一组文档的内存 BM25 索引；
        2. 提供关键词检索接口（retrieve）；
        3. 支持索引的持久化与恢复；
        4. 支持从 Qdrant 向量库全量同步数据。

    线程安全：
        本类**不是线程安全的**。若在生产环境的多线程/多进程场景中使用，
        建议在调用 add_documents / rebuild_index 时加锁，或每个 worker 维护独立索引副本。

    性能边界：
        - 万级文档（< 10 万 chunk）：内存占用 < 500MB，检索延迟 < 50ms，完全适用；
        - 百万级文档：建议迁移到 Elasticsearch / OpenSearch，用其原生 BM25 实现。
    """

    def __init__(
        self,
        documents: Optional[List[Document]] = None,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
        doc_id_key: str = "chunk_id",
    ):
        """
        初始化 BM25 索引器。

        Args:
            documents: 初始文档列表，为 None 时表示空索引，后续通过 add_documents 或
                       from_qdrant 填充。
            tokenizer: 自定义分词函数，默认使用 jieba.cut_for_search。
            doc_id_key: 从 metadata 中提取唯一标识的字段名。
        """
        self._tokenizer: Callable[[str], List[str]] = tokenizer or _default_tokenizer
        self._doc_id_key: str = doc_id_key

        # 内部状态
        self._documents: List[Document] = []
        self._tokenized_corpus: List[List[str]] = []
        self._bm25: Optional[BM25Okapi] = None
        self._doc_id_to_index: Dict[str, int] = {}

        if documents:
            self.rebuild_index(documents)

    # -------------------------------------------------------------------------
    # 核心索引操作
    # -------------------------------------------------------------------------

    def rebuild_index(self, documents: List[Document]) -> None:
        """
        全量重建 BM25 索引。

        这是 BM25 索引器的核心方法。由于 rank-bm25 的 BM25Okapi 在构造时需要
        完整的语料库统计信息（IDF 基于全局词频计算），它不支持真正的"增量插入"
        （不像倒排索引那样可以单独为新文档建 posting list）。

        因此，无论是首次构建还是追加文档，最稳妥的方式都是传入全部文档重新构建。
        在万级文档规模下，重建耗时通常在毫秒级，完全可以接受。

        Args:
            documents: 完整的文档列表（包含旧文档 + 新文档）。
        """
        if not documents:
            logger.warning("BM25 rebuild_index 收到空文档列表，清空当前索引")
            self._documents = []
            self._tokenized_corpus = []
            self._bm25 = None
            self._doc_id_to_index = {}
            return

        start_time = time.time()
        logger.info(f"BM25 开始重建索引，文档数: {len(documents)}")

        # 步骤 1: 去重（以 doc_id 为键，保留最新出现的版本）
        # 去重顺序很重要：如果同一 chunk_id 出现多次，后面的覆盖前面的
        deduped: Dict[str, Document] = {}
        for doc in documents:
            doc_id = _compute_doc_id(doc, self._doc_id_key)
            deduped[doc_id] = doc

        self._documents = list(deduped.values())
        self._doc_id_to_index = {
            _compute_doc_id(doc, self._doc_id_key): idx
            for idx, doc in enumerate(self._documents)
        }

        # 步骤 2: 分词（对每篇文档的 page_content 执行分词）
        self._tokenized_corpus = [
            self._tokenizer(doc.page_content)
            for doc in self._documents
        ]

        # 步骤 3: 构建 BM25Okapi 索引
        # BM25Okapi 构造函数接收分词后的二维列表，内部自动计算词频和 IDF
        self._bm25 = BM25Okapi(self._tokenized_corpus)

        elapsed = time.time() - start_time
        logger.info(
            f"BM25 索引重建完成，去重后文档数: {len(self._documents)}, "
            f"耗时: {elapsed:.3f}s"
        )

    def add_documents(self, documents: List[Document]) -> None:
        """
        增量添加文档。

        实现方式：将新文档追加到现有文档列表，然后调用 rebuild_index 全量重建。
        虽然名字是"增量"，但底层是重建。在中小规模数据下这是最简单可靠的策略。

        Args:
            documents: 要新增的文档列表。
        """
        if not documents:
            return
        logger.info(f"BM25 增量添加 {len(documents)} 个文档")
        combined = self._documents + documents
        self.rebuild_index(combined)

    # -------------------------------------------------------------------------
    # 检索接口
    # -------------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_BM25_TOPK,
        metadata_filter: Optional[Callable[[Document], bool]] = None,
    ) -> List[Document]:
        """
        执行 BM25 关键词检索。

        流程：
            1. 对查询串分词；
            2. 如有 metadata_filter，先筛选符合条件的文档索引；
            3. 调用 BM25Okapi.get_top_n 在候选文档中获取得分最高的索引；
            4. 将索引映射回 Document 对象返回。

        Args:
            query: 用户查询字符串。
            top_k: 返回文档数量。建议取最终需求 top_k 的 3~5 倍，为 RRF 融合留出候选空间。
            metadata_filter: 可选的元数据过滤函数，接收 Document 返回 bool。
                             仅对符合条件的文档执行 BM25 排名。

        Returns:
            按 BM25 得分降序排列的 Document 列表。
        """
        if not self._bm25:
            logger.warning("BM25 索引为空，返回空结果")
            return []

        if not query or not query.strip():
            return []

        start_time = time.time()
        tokenized_query = self._tokenizer(query)

        if not tokenized_query:
            logger.warning("BM25 查询分词结果为空，返回空结果")
            return []

        # 构建候选文档索引列表，支持元数据预过滤
        candidate_indices = list(range(len(self._documents)))
        if metadata_filter:
            candidate_indices = [
                idx for idx in candidate_indices
                if metadata_filter(self._documents[idx])
            ]
            if not candidate_indices:
                logger.info("BM25 元数据过滤后无候选文档，返回空结果")
                return []

        # get_top_n 返回的是 top_k 个文档在 corpus 中的索引位置
        top_indices = self._bm25.get_top_n(
            tokenized_query,
            candidate_indices,
            n=min(top_k, len(candidate_indices)),
        )

        results = [self._documents[idx] for idx in top_indices]

        elapsed = time.time() - start_time
        filter_info = "(已过滤)" if metadata_filter else ""
        logger.info(
            f"BM25 检索完成{filter_info}，查询='{query[:50]}...', "
            f"召回 {len(results)} 个，耗时: {elapsed:.3f}s"
        )
        return results

    # -------------------------------------------------------------------------
    # 持久化与恢复
    # -------------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        """
        将 BM25 索引持久化到磁盘。

        使用 pickle 协议 4（Python 3.4+ 支持），兼容性良好。
        生产环境建议：
            - 将文件存储到对象存储（MinIO / OSS / S3），实现多机共享；
            - 定期备份（如每天一次），防止单点故障；
            - 文件命名带上时间戳（如 bm25_index_20260617.pkl），保留历史版本。

        Args:
            filepath: 存储路径。目录不存在时会自动创建。
        """
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        state = BM25IndexState(
            bm25=self._bm25,
            documents=self._documents,
            tokenized_corpus=self._tokenized_corpus,
            doc_id_key=self._doc_id_key,
        )
        with open(filepath, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"BM25 索引已持久化到: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "BM25Indexer":
        """
        从磁盘加载 BM25 索引。

        注意：如果 pickle 文件是在不同 Python 版本或不同 rank-bm25 版本下生成的，
        加载时可能报兼容性错误。生产环境建议在 CI 中做版本锁定（requirements.txt）。

        Args:
            filepath: 索引文件路径。

        Returns:
            恢复后的 BM25Indexer 实例。
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"BM25 索引文件不存在: {filepath}")

        with open(filepath, "rb") as f:
            state: BM25IndexState = pickle.load(f)

        instance = cls.__new__(cls)
        instance._bm25 = state.bm25
        instance._documents = state.documents
        instance._tokenized_corpus = state.tokenized_corpus
        instance._doc_id_key = state.doc_id_key
        instance._tokenizer = _default_tokenizer
        instance._doc_id_to_index = {
            _compute_doc_id(doc, state.doc_id_key): idx
            for idx, doc in enumerate(state.documents)
        }
        logger.info(f"BM25 索引已从 {filepath} 加载，文档数: {len(instance._documents)}")
        return instance

    # -------------------------------------------------------------------------
    # 从 Qdrant 同步
    # -------------------------------------------------------------------------

    @classmethod
    def from_qdrant(
        cls,
        client: QdrantClient,
        collection_name: str,
        filter_obj: Optional[Filter] = None,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
        doc_id_key: str = "chunk_id",
    ) -> "BM25Indexer":
        """
        从 Qdrant 向量库全量拉取数据，构建 BM25 索引。

        适用场景：
            - 服务首次启动时，从已有向量库同步构建 BM25 索引；
            - 定期全量重建（如每天凌晨），确保 BM25 与向量库数据一致。

        实现细节：
            - 使用 Qdrant 的 scroll API 分页拉取全部数据，避免一次性加载导致内存溢出；
            - 每批次 1000 条，对万级数据通常只需几轮即可拉完。

        Args:
            client: QdrantClient 实例。
            collection_name: Qdrant 集合名称。
            filter_obj: 可选的过滤条件（如只同步 public 文档）。
            tokenizer: 自定义分词器。
            doc_id_key: 文档唯一标识字段。

        Returns:
            构建完成的 BM25Indexer 实例。
        """
        logger.info(
            f"开始从 Qdrant 同步 BM25 索引，集合: {collection_name}, "
            f"过滤条件: {filter_obj is not None}"
        )
        start_time = time.time()

        documents: List[Document] = []
        offset = None
        batch_size = 1000
        total_fetched = 0

        while True:
            # scroll 是 Qdrant 的游标分页 API，适合全量遍历
            records, offset = client.scroll(
                collection_name=collection_name,
                scroll_filter=filter_obj,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,  # BM25 不需要向量，节省带宽和内存
            )

            if not records:
                break

            for record in records:
                payload = record.payload or {}
                # Qdrant 存储时通常将文本放在 page_content 字段
                page_content = payload.get("page_content", "")
                metadata = {k: v for k, v in payload.items() if k != "page_content"}
                documents.append(Document(page_content=page_content, metadata=metadata))

            total_fetched += len(records)
            logger.debug(f"已拉取 {total_fetched} 条记录")

            if offset is None:
                break

        elapsed_fetch = time.time() - start_time
        logger.info(f"Qdrant 数据拉取完成，共 {len(documents)} 条，耗时: {elapsed_fetch:.3f}s")

        instance = cls(
            documents=documents,
            tokenizer=tokenizer,
            doc_id_key=doc_id_key,
        )
        return instance

    # -------------------------------------------------------------------------
    # 属性
    # -------------------------------------------------------------------------

    @property
    def document_count(self) -> int:
        """当前索引中的文档数量。"""
        return len(self._documents)


# =============================================================================
# Cross-Encoder 重排序器
# =============================================================================

class CrossEncoderReranker:
    """
    基于 Cross-Encoder 的重排序器（使用 sentence-transformers）。

    设计目标：
        在 BM25 + 向量检索完成初筛后，使用精度更高的 Cross-Encoder 模型
        对候选文档进行精细重排，显著提升 Top-K 结果的相关性。

    为什么需要重排序？
        - BM25 和向量检索都是"双塔"架构：查询和文档分别编码，相似度计算
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
        batch_size: int = 8,
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
        pairs = [[query, doc.page_content] for doc in documents]
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
        1. 【并行召回】同时执行 BM25 关键词检索和向量语义检索；
        2. 【RRF 融合】将两路结果按排名位置融合为一个统一排序；
        3. 【去重截断】去除重复文档，截取 top_k 返回。

    RRF 算法原理（Reciprocal Rank Fusion）：
        对于某篇文档 d，它在第 i 路检索中的排名为 r_i(d)（从 1 开始），
        则其 RRF 得分为：

            Score(d) = Σ_i [ 1 / (k + r_i(d)) ]

        其中 k 为平滑常数（默认 60）。

        为什么 RRF 比直接加权求和更好？
            - 不同检索方式的分数尺度完全不同（BM25 可能是 10~30，向量相似度是 0~1），
              直接加权相当于让分数尺度大的一路主导结果；
            - RRF 只关心"排名第几"，天然消除了分数尺度的差异；
            - 实现简单，无需训练，零超参（k=60 是论文通用值）。

    适用场景：
        - 用户查询中包含明确的关键词（如"退款政策"、"2026年预算"），BM25 能精准命中；
        - 用户使用口语化、近义词表达（如"怎么退钱"、"今年的花费计划"），向量检索能捕捉语义；
        - 混合检索在绝大多数真实场景下 Recall@K 显著优于单路检索。
    """

    def __init__(
        self,
        vectorstore: QdrantVectorStore,
        bm25_indexer: Optional[BM25Indexer] = None,
        rrf_k: int = DEFAULT_RRF_K,
        bm25_weight: float = 1.0,
        vector_weight: float = 1.0,
        reranker: Optional[CrossEncoderReranker] = None,
        rerank_top_k: Optional[int] = None,
    ):
        """
        初始化混合检索器。

        Args:
            vectorstore: LangChain 的 QdrantVectorStore 实例，负责语义检索。
            bm25_indexer: BM25Indexer 实例，负责关键词检索。为 None 时需要在
                          首次检索前通过 from_qdrant 或 set_indexer 设置。
            rrf_k: RRF 平滑常数，默认 60。
            bm25_weight: BM25 路得分的权重乘数（默认 1.0）。
                         如果你发现业务中关键词匹配更重要，可提高到 1.2~1.5；
                         如果语义理解更重要，可降低到 0.8。
            vector_weight: 向量路得分的权重乘数（默认 1.0）。与 bm25_weight 配合使用。
            reranker: Cross-Encoder 重排序器实例。为 None 时不启用重排序。
            rerank_top_k: 送入重排序器的候选文档数量。
                          默认为 None（取最终 top_k 的 5 倍，或至少 50）。
        """
        self._vectorstore: QdrantVectorStore = vectorstore
        self._bm25_indexer: Optional[BM25Indexer] = bm25_indexer
        self._rrf_k: int = rrf_k
        self._bm25_weight: float = bm25_weight
        self._vector_weight: float = vector_weight
        self._reranker: Optional[CrossEncoderReranker] = reranker
        self._rerank_top_k: Optional[int] = rerank_top_k

    # -------------------------------------------------------------------------
    # 索引管理
    # -------------------------------------------------------------------------

    def set_bm25_indexer(self, indexer: BM25Indexer) -> None:
        """设置或更换 BM25 索引器。"""
        self._bm25_indexer = indexer

    def sync_bm25_from_qdrant(
        self,
        client: QdrantClient,
        collection_name: str,
        filter_obj: Optional[Filter] = None,
    ) -> None:
        """
        从 Qdrant 全量同步数据并重建 BM25 索引。

        建议在以下时机调用：
            - 服务启动时（若本地无缓存索引）；
            - 批量导入文档后；
            - 定时任务（如每天凌晨 3 点）。
        """
        indexer =  BM25Indexer.from_qdrant(
            client=client,
            collection_name=collection_name,
            filter_obj=filter_obj,
        )
        self.set_bm25_indexer(indexer)

    # -------------------------------------------------------------------------
    # 核心检索接口
    # -------------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        bm25_top_k: int = DEFAULT_BM25_TOPK,
        vector_top_k: int = DEFAULT_VECTOR_TOPK,
        filter_obj: Optional[Filter] = None,
        metadata_filter: Optional[Callable[[Document], bool]] = None,
    ) -> List[Document]:
        """
        执行混合检索并返回融合后的结果。

        这是本类的主入口，企业级使用时应直接调用此方法。

        Args:
            query: 用户查询字符串。
            top_k: 最终返回的文档数量。
            bm25_top_k: BM25 单路召回数量。建议 >= top_k * 3。
            vector_top_k: 向量检索单路召回数量。建议 >= top_k * 3。
            filter_obj: Qdrant 过滤条件（权限、状态等），仅传递给向量检索。
            metadata_filter: BM25 元数据过滤函数，接收 Document 返回 bool。
                             仅对符合条件的文档执行 BM25 排名，使 BM25 与向量检索
                             在过滤逻辑上保持一致。

        Returns:
            按 RRF 得分降序排列的 Document 列表，长度 <= top_k。

        Raises:
            RuntimeError: 当 BM25 索引器未初始化时抛出。
        """
        if not self._bm25_indexer:
            raise RuntimeError(
                "BM25 索引器未初始化。请先调用 sync_bm25_from_qdrant() "
                "或 set_bm25_indexer() 设置索引。"
            )

        start_time = time.time()
        logger.info(
            f"混合检索开始，查询='{query[:60]}...', top_k={top_k}, "
            f"RRF k={self._rrf_k}"
        )

        # 步骤 1: 并行执行两路检索
        # 注：BM25 检索是纯 CPU 计算，向量检索涉及网络 IO +  embedding API 调用。
        # 在 asyncio 环境下，BM25 会阻塞事件循环，但由于其耗时极短（< 50ms），
        # 对整体延迟影响可忽略。若严格要求异步，可将 BM25 检索也包进 asyncio.to_thread。
        bm25_results = self._bm25_indexer.retrieve(
            query, top_k=bm25_top_k, metadata_filter=metadata_filter
        )
        vector_results = self._vectorstore.similarity_search(
            query, k=vector_top_k, filter=filter_obj
        )

        logger.info(
            f"两路召回完成: BM25={len(bm25_results)}, Vector={len(vector_results)}"
        )

        # 步骤 2: RRF 融合
        fused_results = self._rrf_fusion(bm25_results, vector_results)

        # 步骤 3: Cross-Encoder 重排序（若启用）
        if self._reranker and fused_results:
            rerank_candidates = self._rerank_top_k or max(top_k * 5, 50)
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

    def _rrf_fusion(
        self,
        bm25_results: List[Document],
        vector_results: List[Document],
    ) -> List[Document]:
        """
        执行 Reciprocal Rank Fusion 融合。

        实现步骤：
            1. 为每路结果中的文档分配排名（从 1 开始）；
            2. 计算每篇文档的 RRF 得分：score = Σ(weight_i / (k + rank_i))；
            3. 按得分降序排列，去重后返回。

        Args:
            bm25_results: BM25 检索结果（已按 BM25 得分降序排列）。
            vector_results: 向量检索结果（已按相似度降序排列）。

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

        # 注册 BM25 路
        if bm25_results:
            _register(bm25_results, self._bm25_weight, "bm25")

        # 注册向量路
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

    async def aretrieve(
        self,
        query: str,
        top_k: int = 10,
        bm25_top_k: int = DEFAULT_BM25_TOPK,
        vector_top_k: int = DEFAULT_VECTOR_TOPK,
        filter_obj: Optional[Filter] = None,
        metadata_filter: Optional[Callable[[Document], bool]] = None,
    ) -> List[Document]:
        """
        异步版本的混合检索。

        与 retrieve 的区别：
            - BM25 检索放入 asyncio.to_thread 避免阻塞事件循环；
            - 向量检索仍使用同步 API（LangChain 的 QdrantVectorStore 未提供原生异步接口）。

        如果你的服务使用 FastAPI + Uvicorn，建议调用此方法而非同步的 retrieve。
        """
        import asyncio

        if not self._bm25_indexer:
            raise RuntimeError("BM25 索引器未初始化")

        start_time = time.time()
        logger.info(f"异步混合检索开始，查询='{query[:60]}...'")

        # BM25 放入线程池执行（避免阻塞主事件循环）
        bm25_task = asyncio.to_thread(
            self._bm25_indexer.retrieve, query, bm25_top_k, metadata_filter
        )
        # 向量检索
        vector_task = asyncio.to_thread(
            self._vectorstore.similarity_search,
            query,
            vector_top_k,
            filter=filter_obj,
        )

        # 并发等待两路结果
        bm25_results, vector_results = await asyncio.gather(
            bm25_task, vector_task
        )

        logger.info(
            f"异步两路召回完成: BM25={len(bm25_results)}, Vector={len(vector_results)}"
        )

        fused_results = self._rrf_fusion(bm25_results, vector_results)

        # Cross-Encoder 重排序（若启用）
        if self._reranker and fused_results:
            rerank_candidates = self._rerank_top_k or max(top_k * 5, 50)
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


