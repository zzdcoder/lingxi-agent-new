"""
BM25 索引企业级管理器

职责：
- 启动时加载磁盘缓存，失败则从 Qdrant 重建
- 全量重建时加锁保护，串行排队
- 原子持久化（临时文件 + replace）
- 重建失败保留旧索引，不中断服务
"""

import asyncio
import logging
import os
from typing import Optional

from qdrant_client import QdrantClient

from rag.hybrid_retriever import BM25Indexer

logger = logging.getLogger(__name__)


class BM25IndexManager:
    """
    BM25 索引的企业级管理器。

    设计目标：
        1. 启动加载：服务启动时自动从磁盘恢复索引，秒级就绪；
        2. 安全重建：全量重建过程加锁，并发请求串行排队；
        3. 原子持久化：先写临时文件，成功后原子替换，防止写坏原文件；
        4. 失败回退：重建或持久化失败时保留旧索引，服务不中断。

    线程安全：
        - rebuild_and_swap() 使用 asyncio.Lock 保护，多个重建请求串行执行；
        - 检索请求（只读）无需加锁，Python 对象引用赋值是原子的。
    """

    def __init__(
        self,
        persist_path: str,
        qdrant_client: QdrantClient,
        collection_name: str,
    ):
        """
        初始化 BM25 索引管理器。

        Args:
            persist_path: 索引持久化文件路径。
            qdrant_client: QdrantClient 实例，用于重建时从向量库拉取数据。
            collection_name: Qdrant 集合名称。
        """
        self._persist_path = persist_path
        self._qdrant_client = qdrant_client
        self._collection_name = collection_name
        self._indexer: Optional[BM25Indexer] = None
        self._lock = asyncio.Lock()

    # -------------------------------------------------------------------------
    # 启动加载（同步上下文调用）
    # -------------------------------------------------------------------------

    def load_sync(self) -> Optional[BM25Indexer]:
        """
        同步方式从磁盘加载 BM25 索引。

        由 init_hybrid_retriever() 在启动时调用。

        Returns:
            加载成功的 BM25Indexer，失败返回 None。
        """
        if not os.path.exists(self._persist_path):
            logger.info(f"本地无 BM25 索引缓存: {self._persist_path}")
            return None

        try:
            indexer = BM25Indexer.load(self._persist_path)
            logger.info(
                f"BM25 索引从磁盘加载成功，文档数: {indexer.document_count}"
            )
            self._indexer = indexer
            return indexer
        except Exception as e:
            logger.warning(f"BM25 索引加载失败: {e}，将重新构建")
            return None

    def build_sync(self) -> Optional[BM25Indexer]:
        """
        同步方式从 Qdrant 全量重建 BM25 索引。

        Returns:
            重建成功的 BM25Indexer，失败返回 None。
        """
        try:
            indexer = BM25Indexer.from_qdrant(
                client=self._qdrant_client,
                collection_name=self._collection_name,
            )
            logger.info(
                f"BM25 索引从 Qdrant 重建完成，文档数: {indexer.document_count}"
            )
            self._indexer = indexer
            self._save_sync()
            return indexer
        except Exception as e:
            logger.error(f"BM25 索引从 Qdrant 重建失败: {e}")
            return None

    def _save_sync(self) -> None:
        """同步方式持久化当前索引到磁盘。"""
        if self._indexer is None:
            return
        try:
            os.makedirs(os.path.dirname(self._persist_path) or ".", exist_ok=True)
            self._indexer.save(self._persist_path)
            logger.info(f"BM25 索引已持久化到: {self._persist_path}")
        except Exception as e:
            logger.error(f"BM25 索引持久化失败: {e}")

    # -------------------------------------------------------------------------
    # 运行时重建（异步，带锁）
    # -------------------------------------------------------------------------

    async def rebuild_and_swap(self) -> int:
        """
        全量重建 BM25 索引并原子替换。

        流程：
            1. 获取锁（防止并发重建）；
            2. 在线程池中从 Qdrant 重建新索引；
            3. 将新索引持久化到临时文件；
            4. 原子重命名替换旧文件；
            5. 替换内存中的 indexer 引用；
            6. 释放锁。

        Returns:
            重建后的文档数量。

        Raises:
            RuntimeError: 当重建或持久化失败时抛出（调用方应捕获）。
        """
        async with self._lock:
            logger.info("开始 BM25 索引全量重建...")
            loop = asyncio.get_event_loop()
            start_time = loop.time()

            # 步骤 1: 在线程池中重建（避免阻塞事件循环）
            new_indexer = await asyncio.to_thread(self._build_indexer_in_thread)
            if new_indexer is None:
                raise RuntimeError("BM25 索引从 Qdrant 重建失败")

            # 步骤 2: 持久化到临时文件
            temp_path = self._persist_path + ".tmp"
            try:
                await asyncio.to_thread(
                    self._save_indexer_in_thread, new_indexer, temp_path
                )
            except Exception as e:
                self._cleanup_temp_file(temp_path)
                raise RuntimeError(f"BM25 索引持久化失败: {e}") from e

            # 步骤 3: 原子替换（临时文件 -> 正式文件）
            try:
                os.replace(temp_path, self._persist_path)
                logger.info(
                    f"BM25 索引文件原子替换完成: {self._persist_path}"
                )
            except Exception as e:
                self._cleanup_temp_file(temp_path)
                raise RuntimeError(f"BM25 索引文件替换失败: {e}") from e

            # 步骤 4: 替换内存引用（Python 引用赋值是原子的）
            old_count = self._indexer.document_count if self._indexer else 0
            self._indexer = new_indexer

            elapsed = loop.time() - start_time
            logger.info(
                f"BM25 索引重建完成，"
                f"旧文档数: {old_count} -> 新文档数: {new_indexer.document_count}, "
                f"耗时: {elapsed:.3f}s"
            )

            return new_indexer.document_count

    def _build_indexer_in_thread(self) -> Optional[BM25Indexer]:
        """在线程中执行的重建逻辑。"""
        try:
            return BM25Indexer.from_qdrant(
                client=self._qdrant_client,
                collection_name=self._collection_name,
            )
        except Exception as e:
            logger.error(f"BM25 索引重建异常: {e}")
            return None

    @staticmethod
    def _save_indexer_in_thread(indexer: BM25Indexer, path: str) -> None:
        """在线程中执行的持久化逻辑。"""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        indexer.save(path)

    @staticmethod
    def _cleanup_temp_file(temp_path: str) -> None:
        """清理临时文件。"""
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # 属性
    # -------------------------------------------------------------------------

    @property
    def indexer(self) -> Optional[BM25Indexer]:
        """当前内存中的 BM25 索引器。"""
        return self._indexer
