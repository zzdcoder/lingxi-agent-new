import asyncio
import logging
import os
import socket
import uuid
from typing import Optional

import dashscope
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter as QdrantFilter, FieldCondition, MatchValue
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from core import settings
from models.file_schema import ChunkResult

logger = logging.getLogger(__name__)

load_dotenv(".env.dev")

# Qdrant 本地磁盘存储路径
QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_data")
# 集合名称
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "lingxi-agent-collection")
# 向量维度（与 DashScope text-embedding-v4 默认维度一致）
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "1024"))

# 命名稀疏向量名称（与检索侧 HybridRetriever 共用）
SPARSE_VECTOR_NAME = "sparse"

# 全局 QdrantClient 单例（本地磁盘模式不支持多实例并发访问同一存储目录）
_qdrant_client: Optional[QdrantClient] = None


def get_qdrant_client() -> QdrantClient:
    """获取全局 QdrantClient 单例。"""
    global _qdrant_client
    if _qdrant_client is None:
        logger.info(f"创建 QdrantClient 单例 (路径: {QDRANT_PATH})")
        _qdrant_client = QdrantClient(path=QDRANT_PATH)
    return _qdrant_client


def _to_sparse_vector(sparse_items: list[dict]) -> SparseVector:
    """
    将 text-embedding-v4 返回的稀疏项转换为 Qdrant SparseVector。

    v4 返回结构：sparse_embedding: [{index, value, token?}, ...]。
    Qdrant 要求 indices 严格升序且唯一，故合并同 index 后排序输出。

    :param sparse_items: 稀疏项列表
    :return: 升序去重的 Qdrant 稀疏向量
    """
    if not sparse_items:
        return SparseVector(indices=[], values=[])

    dims: dict[int, float] = {}
    for item in sparse_items:
        idx = item.get("index", item.get("token_id"))
        val = item.get("value", item.get("weight"))
        if idx is None or val is None:
            continue
        # 同一维度权重累加（防御性合并，避免重复 index）
        dims[int(idx)] = dims.get(int(idx), 0.0) + float(val)

    indices = sorted(dims)
    values = [round(dims[i], 6) for i in indices]
    return SparseVector(indices=indices, values=values)


class DashScopeEmbedding(Embeddings):
    def __init__(self, api_key: str, model: str = "text-embedding-v4", dimension: int = 1024):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        # DashScope 原生接口（获取稀疏向量必需，OpenAI 兼容模式不支持）
        dashscope.api_key = api_key
        self.model = model
        self.dimensions = dimension  # 向量维度，控制输出向量的长度；向量数据库一旦创建就无法更改维度


    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        batch_size = 10
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
                dimensions=self.dimensions,
                encoding_format="float"
            )
            all_embeddings.extend([item.embedding for item in response.data])

        return all_embeddings


    def embed_query(self, text: str) -> list[float]:
        """嵌入单个查询"""
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
            dimensions=self.dimensions,
            encoding_format="float"
        )
        return response.data[0].embedding

    # -------------------------------------------------------------------------
    # text-embedding-v4 双输出（稠密 + 稀疏）
    # OpenAI 兼容模式不支持稀疏向量，以下方法统一走 DashScope 原生接口
    # -------------------------------------------------------------------------

    def embed_documents_with_sparse(
        self, texts: list[str]
    ) -> list[tuple[list[float], SparseVector]]:
        """
        批量生成稠密 + 稀疏向量（一次调用双输出）。

        文档侧使用 text_type="document"，提升非对称检索（文档 vs 查询）效果。

        :param texts: 文本列表（单次请求最多 10 条）
        :return: [(稠密向量, 稀疏向量), ...]，与输入顺序一致
        """
        results: list[tuple[list[float], SparseVector]] = []
        for i in range(0, len(texts), 10):
            batch = texts[i:i + 10]
            for emb in self._call_v4(batch, text_type="document"):
                results.append(
                    (
                        list(emb.get("embedding") or []),
                        _to_sparse_vector(emb.get("sparse_embedding") or []),
                    )
                )
        return results

    def embed_query_with_sparse(self, text: str) -> tuple[list[float], SparseVector]:
        """
        为单个查询生成稠密 + 稀疏向量（text_type="query"）。

        :param text: 查询文本
        :return: (稠密向量, 稀疏向量)
        """
        emb = self._call_v4([text], text_type="query")[0]
        return (
            list(emb.get("embedding") or []),
            _to_sparse_vector(emb.get("sparse_embedding") or []),
        )

    def _call_v4(self, texts: list[str], text_type: str) -> list[dict]:
        """
        调用 text-embedding-v4 原生接口（dense & sparse 双输出）。

        :param texts: 输入文本列表（<= 10 条）
        :param text_type: "query" 或 "document"
        :return: 每条输入对应的 embedding 结构列表
        """
        from http import HTTPStatus

        resp = dashscope.TextEmbedding.call(
            model=self.model,
            input=texts,
            dimension=self.dimensions,
            output_type="dense&sparse",
            text_type=text_type,
        )
        if resp.status_code != HTTPStatus.OK:
            raise RuntimeError(
                f"text-embedding-v4 调用失败: code={resp.code}, message={resp.message}"
            )
        return resp.output["embeddings"]


class EmbeddingHandler():

    def __init__(self,api_key: str):
        self.api_key = api_key


    def init_vectorstore(self):
        logger.info("EmbeddingHandler 初始化: 开始创建 DashScopeEmbedding")
        embeddings = DashScopeEmbedding(api_key=self.api_key)
        logger.info(f"EmbeddingHandler 初始化: 开始连接 Qdrant (路径: {QDRANT_PATH})")

        # 使用本地磁盘模式，复用全局单例避免并发访问冲突
        client = get_qdrant_client()

        # 检查集合并创建（如果不存在）
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        if QDRANT_COLLECTION not in collection_names:
            logger.info(f"Qdrant 集合不存在，创建新集合: {QDRANT_COLLECTION}")
            client.create_collection(
                collection_name=QDRANT_COLLECTION,
                # 无名稠密向量（与历史数据/检索兼容）+ 命名稀疏向量（text-embedding-v4 关键词召回）
                vectors_config={
                    "": VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
                },
                sparse_vectors_config={SPARSE_VECTOR_NAME: SparseVectorParams()},
            )
            logger.info(f"Qdrant 集合创建成功: {QDRANT_COLLECTION}（dense + sparse）")
        else:
            # 集合已存在：缺少稀疏向量配置时增量补充。
            # 老数据点仅有稠密向量，稀疏路对它们召回为空，直到文档重新上传（先删后插）。
            info = client.get_collection(QDRANT_COLLECTION)
            sparse_names = set(
                (getattr(getattr(info.config, "params", None), "sparse_vectors", None) or {}).keys()
            )
            if SPARSE_VECTOR_NAME not in sparse_names:
                logger.info(f"集合缺少稀疏向量配置，增量添加: {SPARSE_VECTOR_NAME}")
                client.update_collection(
                    collection_name=QDRANT_COLLECTION,
                    sparse_vectors_config={SPARSE_VECTOR_NAME: SparseVectorParams()},
                )
            logger.info(f"Qdrant 集合已存在: {QDRANT_COLLECTION}")

        logger.info("EmbeddingHandler 初始化: Qdrant 连接成功")
        return QdrantVectorStore(
            client=client,
            collection_name=QDRANT_COLLECTION,
            embedding=embeddings,
        )


    def delete_by_doc_id(self, doc_id: str) -> None:
        """
        根据 doc_id 删除向量库中该文档的所有旧 chunk。

        在文档更新场景中，需要先删除旧版本再写入新版本，
        避免向量库中同时存在同一文档的新旧内容导致检索混淆。

        Args:
            doc_id: 文档唯一标识。
        """
        if not doc_id:
            logger.warning("doc_id 为空，跳过删除")
            return

        try:
            client = get_qdrant_client()
            filter_obj = QdrantFilter(
                must=[
                    FieldCondition(
                        key="metadata.doc_id",
                        match=MatchValue(value=doc_id)
                    )
                ]
            )
            client.delete(
                collection_name=QDRANT_COLLECTION,
                points_selector=filter_obj,
            )
            logger.info(f"已删除 doc_id={doc_id} 在向量库中的旧 chunk")
        except Exception as e:
            logger.error(f"删除 doc_id={doc_id} 的旧 chunk 失败: {e}")
            raise

    async def save_to_vectors(self, documents: list[ChunkResult], doc_id: str = None):
        """
        将文档块存入向量数据库（text-embedding-v4 稠密 + 稀疏向量）。

        Args:
            documents: 文档块列表
            doc_id: 可选的文档 ID。如果提供，会先删除该 doc_id 的所有旧 chunk
                    （同时删除稠密与稀疏向量），再写入新 chunk，实现"覆盖更新"语义。
        """
        logger.info(f"开始向量存储（稠密 + 稀疏），文档块数: {len(documents)}")

        if not documents:
            logger.warning("文档列表为空，跳过向量存储")
            return

        # 步骤 1: 如果提供了 doc_id，先删除该文档的旧 chunk（Qdrant 删点即删全部向量）
        if doc_id:
            logger.info(f"检测到 doc_id={doc_id}，先清理向量库中的旧内容")
            await asyncio.to_thread(self.delete_by_doc_id, doc_id)

        # 步骤 2: 确保集合就绪（稠密 + 稀疏向量配置）
        self.init_vectorstore()

        # 步骤 3: 向量化并写入（在线程中执行，避免阻塞事件循环）
        points = await asyncio.to_thread(self._build_and_upsert_points, documents)
        logger.info(f"向量存储完成，写入 {len(points)} 个点（dense + sparse）")

    def _build_and_upsert_points(self, documents: list[ChunkResult]) -> list:
        """
        批量生成双向量点并写入 Qdrant（同步方法，供线程池执行）。

        点 ID 由 chunk_id（或文本内容兜底）稳定派生：同一 chunk 重复写入时
        upsert 覆盖，天然幂等。

        :param documents: 文档块列表
        :return: 写入的点列表
        """
        embeddings = DashScopeEmbedding(api_key=self.api_key)
        texts = [doc.page_content for doc in documents]
        # text-embedding-v4 一次调用同时返回稠密 + 稀疏向量
        dense_sparse_pairs = embeddings.embed_documents_with_sparse(texts)

        client = get_qdrant_client()
        points = []
        for chunk, (dense, sparse) in zip(documents, dense_sparse_pairs):
            metadata = chunk.metadata or {}
            chunk_id = str(metadata.get("chunk_id") or chunk.page_content)
            points.append(
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id)),
                    vector={
                        "": dense,
                        SPARSE_VECTOR_NAME: sparse,
                    },
                    payload={
                        "page_content": chunk.page_content,
                        "metadata": metadata,
                    },
                )
            )

        client.upsert(collection_name=QDRANT_COLLECTION, points=points)
        return points
