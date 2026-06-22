import asyncio
import logging
import os
import socket
from typing import Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter as QdrantFilter, FieldCondition, MatchValue
from qdrant_client.models import Distance, VectorParams

from core import settings
from models.file_schema import ChunkResult

logger = logging.getLogger(__name__)

load_dotenv(".env.dev")

# Qdrant 本地磁盘存储路径
QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_data")
# 集合名称
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "lingxi-agent-collection")
# 向量维度（与 DashScope text-embedding-v3 一致）
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "1024"))

# 全局 QdrantClient 单例（本地磁盘模式不支持多实例并发访问同一存储目录）
_qdrant_client: Optional[QdrantClient] = None


def get_qdrant_client() -> QdrantClient:
    """获取全局 QdrantClient 单例。"""
    global _qdrant_client
    if _qdrant_client is None:
        logger.info(f"创建 QdrantClient 单例 (路径: {QDRANT_PATH})")
        _qdrant_client = QdrantClient(path=QDRANT_PATH)
    return _qdrant_client


class  DashScopeEmbedding(Embeddings):
    def __init__(self,api_key:str,model: str="text-embedding-v3",dimension:int=1024):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = model
        self.dimensions = dimension   #向量维度，控制输出向量的长度，比如1024就是吧一个文本用1024个数字去表示存储
                                      #向量数据库一旦被创建就无法更改维度


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
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE,  # DashScope 嵌入已归一化，余弦相似度最合适
                ),
            )
            logger.info(f"Qdrant 集合创建成功: {QDRANT_COLLECTION}")
        else:
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
        将文档块存入向量数据库。

        Args:
            documents: 文档块列表
            doc_id: 可选的文档 ID。如果提供，会先删除该 doc_id 的所有旧 chunk，
                    再写入新 chunk，实现"覆盖更新"语义。
        """
        logger.info(f"开始向量存储，文档块数: {len(documents)}")

        if not documents:
            logger.warning("文档列表为空，跳过向量存储")
            return

        # 步骤 1: 如果提供了 doc_id，先删除该文档的旧 chunk
        if doc_id:
            logger.info(f"检测到 doc_id={doc_id}，先清理向量库中的旧内容")
            await asyncio.to_thread(self.delete_by_doc_id, doc_id)

        # 步骤 2: 将 ChunkResult 转成 Document 对象
        docs = [
            Document(page_content=doc.page_content, metadata=doc.metadata)
            for doc in documents
        ]
        logger.info(f"文档转换完成，共 {len(docs)} 个 Document 对象")

        # 步骤 3: 向量化写入 Qdrant
        vectorstore = self.init_vectorstore()
        vectorstore.add_documents(docs)
        logger.info(f"向量存储完成，数据已写入 Qdrant 集合: {QDRANT_COLLECTION}")
