import logging
import os
import socket

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
from qdrant_client import QdrantClient
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

        # 使用本地磁盘模式，无需独立服务进程
        client = QdrantClient(path=QDRANT_PATH)

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


    async def save_to_vectors(self, documents: list[ChunkResult]):
        logger.info(f"开始向量存储，文档块数: {len(documents)}")

        if not documents:
            logger.warning("文档列表为空，跳过向量存储")
            return

        # 将 ChunkResult 转成 Document 对象
        docs = [
            Document(page_content=doc.page_content, metadata=doc.metadata)
            for doc in documents
        ]
        logger.info(f"文档转换完成，共 {len(docs)} 个 Document 对象")

        # 向量化写入 Qdrant
        vectorstore = self.init_vectorstore()
        vectorstore.add_documents(docs)
        logger.info(f"向量存储完成，数据已写入 Qdrant 集合: {QDRANT_COLLECTION}")
