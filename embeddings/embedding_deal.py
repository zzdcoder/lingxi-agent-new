import logging
import os
import socket

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from openai import OpenAI

from core import settings
from models.file_schema import ChunkResult

logger = logging.getLogger(__name__)

# ChromaDB 端口（与 app/main.py 保持一致）
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8001"))


def _check_chroma_port(host: str = "localhost", port: int = 8000, timeout: float = 3.0):
    """快速检测 ChromaDB 端口是否开放，避免 HttpClient 无限等待"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        result = sock.connect_ex((host, port))
        if result != 0:
            raise ConnectionError(
                f"ChromaDB 服务未启动或无法连接: {host}:{port} "
                f"(socket error code: {result})"
            )
    finally:
        sock.close()


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

    def __init__(self, documents: list[ChunkResult], api_key: str):
        self.documents = documents
        self.api_key = api_key
        logger.info("EmbeddingHandler 初始化: 开始创建 DashScopeEmbedding")
        self.embeddings = DashScopeEmbedding(api_key=api_key)
        logger.info(f"EmbeddingHandler 初始化: 开始连接 ChromaDB (localhost:{CHROMA_PORT})")
        _check_chroma_port("localhost", CHROMA_PORT)
        self.client = chromadb.HttpClient(
            host="localhost",
            port=CHROMA_PORT,
            settings=chromadb.config.Settings(
                chroma_server_host="localhost",
                chroma_server_http_port=CHROMA_PORT,
                anonymized_telemetry=False,
            )
        )
        logger.info("EmbeddingHandler 初始化: ChromaDB 连接成功")
        self.vectorstore = Chroma(
            client=self.client,
            collection_name="lingxi-agent-collection",  # 进行分组，组名称，数据隔离
            embedding_function=self.embeddings
        )
        logger.info("EmbeddingHandler 初始化: Chroma vectorstore 就绪")


    async def save_to_vectors(self):
        logger.info(f"开始向量存储，文档块数: {len(self.documents)}")

        # 将 ChunkResult 转成 Document 对象
        docs = [
            Document(page_content=doc.page_content, metadata=doc.metadata)
            for doc in self.documents
        ]
        logger.info(f"文档转换完成，共 {len(docs)} 个 Document 对象")

        # 向量化写入
        import asyncio
        await asyncio.to_thread(self.vectorstore.add_documents, docs)
        logger.info("向量存储完成，数据已写入 ChromaDB")

if __name__ == '__main__':
        embeddings = DashScopeEmbedding(api_key=settings.api_key)
        client = chromadb.HttpClient(host="localhost", port=8345)
        vectorstore = Chroma(
             client=client,
             collection_name="lingxi-agent-collection",  # 进行分组，组名称，数据隔离
             embedding_function=embeddings
         )
        retrieve_result= vectorstore.similarity_search("group_IRuleCheckCSV_groupRuleCheck",k=2)
        for e in retrieve_result:
            print(e)