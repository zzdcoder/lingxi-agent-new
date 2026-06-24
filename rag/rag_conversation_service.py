"""
RAG 对话服务

集成知识检索的对话服务，支持：
1. 向量数据库检索
2. 上下文增强
3. 带 Memory 的对话
"""
import logging
import os
import time
from typing import List, Optional, Dict, Any, AsyncGenerator
import asyncio

from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from rag.hybrid_retriever import HybridRetriever, CrossEncoderReranker
from rag.bm25_index_manager import BM25IndexManager
from rag.memory_mysql import MySQLChatMessageHistory, MySQLConversationSummaryMemory
from prompt.prompt_storage import (
    RAG_SYSTEM_PROMPT_WITH_CONTEXT,
    RAG_SYSTEM_PROMPT_WITHOUT_CONTEXT,
    SUMMARY_GENERATION_PROMPT,
    CONVERSATION_SUMMARY_PREFIX,
    INCREMENTAL_SUMMARY_PROMPT,
)
from embeddings.embedding_deal import DashScopeEmbedding, QDRANT_PATH, QDRANT_COLLECTION
from core.config import settings
from embeddings.embedding_deal import EmbeddingHandler

logger = logging.getLogger(__name__)

# =============================================================================
# 全局混合检索器与 BM25 管理器（由 app/main.py lifespan 在启动时注入）
# =============================================================================

_hybrid_retriever: Optional[HybridRetriever] = None
_bm25_manager: Optional[BM25IndexManager] = None
"""
全局混合检索器单例与 BM25 索引管理器。

为什么使用模块级全局变量而非依赖注入？
    1. RAGConversationService 通过 get_rag_conversation_service 创建，该函数
       作为 FastAPI 依赖，只能接收 db: AsyncSession 一个参数；
    2. 若要通过 Request 获取 app.state，需要修改所有路由的依赖签名，改动面广；
    3. 混合检索器本身是无状态的纯检索组件，适合作为全局单例共享；
    4. 启动时 lifespan 注入一次，所有请求复用，无需每次重建连接。
"""


def set_hybrid_retriever(retriever: HybridRetriever) -> None:
    """
    设置全局混合检索器实例。

    由外部模块（如 file_process）在索引重建成功后调用。
    """
    global _hybrid_retriever
    _hybrid_retriever = retriever
    logger.info("全局混合检索器已注入 RAGConversationService")


def get_hybrid_retriever() -> Optional[HybridRetriever]:
    """获取全局混合检索器实例。"""
    return _hybrid_retriever


def get_bm25_manager() -> Optional[BM25IndexManager]:
    """获取全局 BM25 索引管理器。"""
    return _bm25_manager


def init_hybrid_retriever() -> HybridRetriever:
    """
    惰性初始化全局混合检索器。

    首次调用时创建 HybridRetriever 实例，并通过 BM25IndexManager 加载/重建索引。
    后续调用直接返回已有实例，避免重复初始化。

    Returns:
        全局 HybridRetriever 实例。
    """
    global _hybrid_retriever, _bm25_manager
    if _hybrid_retriever is not None:
        return _hybrid_retriever

    logger.info("首次初始化混合检索器...")
    vectorstore = EmbeddingHandler(api_key=settings.api_key).init_vectorstore()

    # 初始化 Cross-Encoder 重排序器（若配置了模型路径）
    reranker = None
    if getattr(settings, "reranker_model", None):
        reranker = CrossEncoderReranker(
            model_name=settings.reranker_model,
            device=getattr(settings, "reranker_device", None),
            max_length=getattr(settings, "reranker_max_length", 512),
            batch_size=getattr(settings, "reranker_batch_size", 8),
        )
        logger.info(f"Cross-Encoder 重排序器已启用: {settings.reranker_model}")

    _hybrid_retriever = HybridRetriever(
        vectorstore=vectorstore,
        reranker=reranker,
    )

    # ── 创建 BM25 索引管理器并启动加载 ──
    from embeddings.embedding_deal import get_qdrant_client
    _bm25_manager = BM25IndexManager(
        persist_path="./data/bm25_index.pkl",
        qdrant_client=get_qdrant_client(),
        collection_name=QDRANT_COLLECTION,
    )

    # 尝试从磁盘加载缓存
    indexer = _bm25_manager.load_sync()
    if indexer:
        _hybrid_retriever.set_bm25_indexer(indexer)
    else:
        # 缓存不存在或损坏，从 Qdrant 全量重建
        indexer = _bm25_manager.build_sync()
        if indexer:
            _hybrid_retriever.set_bm25_indexer(indexer)
        else:
            logger.warning(
                "BM25 索引初始化失败，检索将降级为纯向量模式。"
                "请检查 Qdrant 连接并调用 rebuild_hybrid_index()"
            )

    logger.info("混合检索器初始化完成")
    return _hybrid_retriever


async def rebuild_hybrid_index() -> None:
    """
    从 Qdrant 全量同步数据，重建 BM25 索引并原子替换。

    本函数为异步接口，内部通过 BM25IndexManager 实现：
        - 获取锁防止并发重建
        - 在线程池中执行重建和持久化（避免阻塞事件循环）
        - 使用临时文件 + 原子替换保证持久化安全
        - 重建失败保留旧索引

    应在以下场景调用：
        - 向量存储完成新文档写入后（file_process 接口）；
        - 文档被删除或更新后；
        - 定时全量重建（可选）。
    """
    global _hybrid_retriever, _bm25_manager

    # 确保混合检索器壳子已存在
    if _hybrid_retriever is None:
        init_hybrid_retriever()

    if _bm25_manager is None:
        logger.warning("BM25 管理器未初始化，跳过重建")
        return

    try:
        doc_count = await _bm25_manager.rebuild_and_swap()
        # 原子替换 HybridRetriever 中的 indexer 引用
        _hybrid_retriever.set_bm25_indexer(_bm25_manager.indexer)
        logger.info(f"BM25 索引重建完成，当前文档数: {doc_count}")
    except Exception as e:
        logger.error(f"BM25 索引重建失败: {e}，保留旧索引")
        raise


class RAGConversationService:
    """
    RAG 对话服务
    
    结合向量数据库检索和对话记忆，提供知识增强的对话能力
    """

    def __init__(
        self,
        db_session: AsyncSession,
        hybrid_retriever: Optional[HybridRetriever] = None,
    ):
        """
        初始化 RAG 对话服务

        :param db_session: SQLAlchemy 异步会话
        :param hybrid_retriever: 混合检索器实例。若为 None，则回退到模块级
                                 全局变量 _hybrid_retriever（由 lifespan 注入）。
        """
        self.db = db_session
        self.embeddings = DashScopeEmbedding(api_key=settings.api_key)
        self.hybrid_retriever = hybrid_retriever or _hybrid_retriever

        # 复用 hybrid_retriever 内部的 vectorstore，避免重复创建 QdrantClient
        if self.hybrid_retriever:
            self.vectorstore = self.hybrid_retriever._vectorstore
        else:
            self.vectorstore = EmbeddingHandler(api_key=settings.api_key).init_vectorstore()

    async def chat_with_rag(
        self,
        conversation_id: str,
        messages: List[Dict[str, Any]],
        model: str = "qwen-turbo",
        login_username: Optional[str] = None
    ) -> AsyncGenerator[bytes, None]:
        """
        带 RAG 检索的对话（流式响应）
        
        流程：
        1. 检索相关知识
        2. 构建增强 Prompt
        3. 调用 LLM 生成回答
        4. 保存到 Memory
        
        :param conversation_id: 会话ID
        :param messages: 消息列表
        :param model: 使用的模型
        :yield: SSE 格式的响应流
        """
        start_time = time.time()
        logger.info(f"开始 RAG 对话: conversation_id={conversation_id}")
        
        try:
            # 1. 提取用户问题
            user_input = messages[-1].get("content", "")
            logger.info(f"【用户问题】{user_input}")
            
            # 2. 检索相关知识
            context_docs = await self._retrieve_context(user_input, k=4, login_username=login_username)
            context_text = self._format_context(context_docs)
            
            # 3. 创建 MySQL Memory
            mysql_history = MySQLChatMessageHistory(
                session=self.db,
                conversation_id=conversation_id
            )
            
            # 4. 构建 RAG 链
            rag_chain = self._build_rag_chain(model, context_text)
            
            # 5. 执行对话（流式）
            async for chunk in self._stream_rag_response(rag_chain, user_input, context_text, mysql_history):
                yield chunk
            
            elapsed = time.time() - start_time
            logger.info(f"RAG 对话完成: 耗时={elapsed:.2f}s")
            
        except Exception as e:
            logger.error(f"RAG 对话失败: {e}")
            error_data = f'data: {{"content": "对话失败：{str(e)}", "done": true}}\n\n'
            yield error_data.encode('utf-8')

    async def _retrieve_context(
        self,
        query: str,
        k: int = 3,
        login_username: str = None
    ) -> List[Any]:
        """
        检索相关知识（混合检索模式）。

        检索策略优先级：
            1. 若 hybrid_retriever 已初始化，执行 BM25 + 向量混合检索，
               经 RRF 融合后按 created_at 倒序排列（新知识优先）；
            2. 若 hybrid_retriever 未初始化（如启动失败或依赖未安装），
               自动降级为纯向量检索，保证服务可用性。

        :param query: 查询文本
        :param k: 返回文档数量
        :param login_username: 当前登录用户名，用于权限过滤
        :return: 相关文档列表
        """
        if not self.vectorstore:
            logger.warning("向量数据库未初始化，跳过检索")
            return []

        try:
            # -----------------------------------------------------------------
            # 步骤 1: 构建权限过滤条件（与原有逻辑保持一致）
            # -----------------------------------------------------------------
            filter_obj = None
            if login_username:
                filter_obj = Filter(
                    should=[
                        FieldCondition(
                            key="metadata.auth_option",
                            match=MatchValue(value="public")
                        ),
                        Filter(
                            must=[
                                FieldCondition(
                                    key="metadata.auth_option",
                                    match=MatchValue(value="private")
                                ),
                                FieldCondition(
                                    key="metadata.create_username",
                                    match=MatchValue(value=login_username)
                                ),
                            ]
                        ),
                    ]
                )
            if filter_obj:
                logger.info(f"构建的过滤条件为: {filter_obj.model_dump(exclude_none=True)}")
            else:
                logger.info("未构建过滤条件，执行无过滤检索")

            # 构建与 Qdrant filter_obj 逻辑等价的 BM25 元数据过滤函数
            metadata_filter = None
            if login_username:
                def _bm25_auth_filter(doc: Document) -> bool:
                    metadata = doc.metadata or {}
                    auth = metadata.get("auth_option")
                    if auth == "public":
                        return True
                    if auth == "private" and metadata.get("create_username") == login_username:
                        return True
                    return False
                metadata_filter = _bm25_auth_filter

            # -----------------------------------------------------------------
            # 步骤 2: 执行检索（混合检索优先，降级到纯向量检索）
            # -----------------------------------------------------------------
            if self.hybrid_retriever:
                logger.info(f"【混合检索】查询='{query[:60]}...', top_k={k}")
                docs = await self.hybrid_retriever.aretrieve(
                    query=query,
                    top_k=k,
                    bm25_top_k=k * 5,      # 单路召回数取最终需求的 5 倍，为 RRF 预留候选池
                    vector_top_k=k * 5,
                    filter_obj=filter_obj,
                    metadata_filter=metadata_filter,
                )
            else:
                logger.info(f"【纯向量检索-降级模式】查询='{query[:60]}...', top_k={k}")
                docs = await asyncio.to_thread(
                    self.vectorstore.similarity_search,
                    query,
                    k=k,
                    filter=filter_obj
                )


            logger.info(f"检索完成: 查询='{query}', 找到 {len(docs)} 个文档")
            for idx, doc in enumerate(docs, 1):
                content_preview = doc.page_content[:200].replace('\n', ' ')
                logger.info(
                    f"  [检索结果{idx}] "
                    f"{content_preview}{'...' if len(doc.page_content) > 200 else ''}"
                )
            return docs

        except Exception as e:
            logger.error(f"检索失败: {e}")
            return []




    def _format_context(self, docs: List[Any]) -> str:
        """
        格式化检索结果为上下文文本
        
        :param docs: 文档列表
        :return: 格式化的上下文
        """
        if not docs:
            return ""
        
        context_parts = []
        for idx, doc in enumerate(docs, 1):
            context_parts.append(f"[文档{idx}]\n{doc.page_content}")
        
        return "\n\n".join(context_parts)

    # -----------------------------------------------------------------
    # 上下文压缩相关方法
    # -----------------------------------------------------------------

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """
        粗略估算 Token 数量。

        策略：中文字符按 1.5 倍估算，英文单词按 1.2 倍估算，
        其余字符按 0.1 倍估算。此估算偏保守，实际值通常更小。
        """
        import re
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_words = len(re.findall(r'[a-zA-Z]+', text))
        other_chars = len(text) - chinese_chars - len(re.findall(r'[a-zA-Z]', text))
        return int(chinese_chars * 1.5 + english_words * 1.2 + other_chars * 0.1)

    async def _build_compressed_history(
        self,
        mysql_history: MySQLChatMessageHistory,
    ) -> List[Any]:
        """
        构建压缩后的对话历史。

        压缩策略（两阶段）：
            阶段一（硬截断）：消息数 <= 20 时，若超过 10 条则只保留最近 10 条。
            阶段二（摘要压缩）：消息数 > 20 时，生成或复用对话摘要，
                              用 "摘要 + 最近 6 条原始消息" 替代全量历史。

        为什么保留 6 条而非更少？
            长对话中，用户可能正在追问某个细节，只保留 2~3 条可能丢失
            指代消解所需的上下文（如"那它的价格呢？"）。6 条约等于 3 轮
            对话，能覆盖大多数指代场景。
        """
        # 获取全量历史（不包含当前用户输入，因为尚未写入）
        history = await mysql_history.aget_messages()
        total_messages = len(history)

        if total_messages <= 10:
            logger.info(f"历史消息 {total_messages} 条，无需压缩")
            return history

        # 阶段一：硬截断（10 < 消息数 <= 20）
        if total_messages <= 20:
            logger.info(
                f"历史消息 {total_messages} 条，执行硬截断保留最近 10 条"
            )
            return history[-10:]

        # 阶段二：摘要压缩（消息数 > 20）
        logger.info(
            f"历史消息 {total_messages} 条，超过阈值，启用摘要压缩"
        )
        summary_memory = MySQLConversationSummaryMemory(session=self.db)
        summary_record = await self._get_summary_record(
            mysql_history.conversation_id
        )

        if summary_record:
            # 已有摘要，判断是否需要增量更新
            previously_summarized = summary_record.message_count
            new_message_count = total_messages - previously_summarized - 6

            if new_message_count > 10:
                # 新增消息超过阈值，执行增量更新
                logger.info(
                    f"新增消息 {new_message_count} 条，触发增量摘要更新"
                )
                new_messages = history[previously_summarized:-6]
                try:
                    updated_summary = await self._generate_incremental_summary(
                        old_summary=summary_record.summary,
                        new_messages=new_messages,
                    )
                    await summary_memory.save_summary(
                        conversation_id=mysql_history.conversation_id,
                        summary=updated_summary,
                        message_count=total_messages - 6,
                    )
                    logger.info("增量摘要更新完成")
                    summary_msg = SystemMessage(
                        content=f"{CONVERSATION_SUMMARY_PREFIX}\n{updated_summary}"
                    )
                    return [summary_msg] + history[-6:]
                except Exception as e:
                    logger.error(f"增量摘要更新失败: {e}，复用旧摘要")
                    summary_msg = SystemMessage(
                        content=f"{CONVERSATION_SUMMARY_PREFIX}\n{summary_record.summary}"
                    )
                    return [summary_msg] + history[-6:]
            else:
                # 新增消息不多，直接复用旧摘要
                logger.info(
                    f"复用已有对话摘要（已总结 {previously_summarized} 条，"
                    f"新增 {new_message_count} 条未达更新阈值）"
                )
                summary_msg = SystemMessage(
                    content=f"{CONVERSATION_SUMMARY_PREFIX}\n{summary_record.summary}"
                )
                return [summary_msg] + history[-6:]

        # 首次超过阈值，需要生成摘要
        messages_to_summarize = history[:-6]
        recent_messages = history[-6:]

        try:
            summary_text = await self._generate_summary(messages_to_summarize)
            await summary_memory.save_summary(
                conversation_id=mysql_history.conversation_id,
                summary=summary_text,
                message_count=len(messages_to_summarize),
            )
            logger.info(
                f"对话摘要生成完成，已总结 {len(messages_to_summarize)} 条消息"
            )
            summary_msg = SystemMessage(
                content=f"{CONVERSATION_SUMMARY_PREFIX}\n{summary_text}"
            )
            return [summary_msg] + recent_messages
        except Exception as e:
            logger.error(f"摘要生成失败: {e}，降级为硬截断")
            # 降级策略：生成失败时保留最近 10 条
            return history[-10:]

    async def _get_summary_record(
        self,
        conversation_id: str,
    ) -> Optional[Any]:
        """
        获取会话摘要的完整数据库记录。

        用于增量摘要更新时读取 message_count 等元数据。
        """
        from models.conversation_model import ConversationSummary
        from sqlalchemy import select

        try:
            stmt = select(ConversationSummary).where(
                ConversationSummary.conversation_id == conversation_id
            )
            result = await self.db.execute(stmt)
            return result.scalars().first()
        except Exception as e:
            logger.error(f"获取摘要记录失败: {e}")
            return None

    @staticmethod
    def _messages_to_conversation_text(messages: List[Any]) -> str:
        """
        将消息列表转换为对话文本格式。
        """
        lines = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                lines.append(f"用户: {msg.content}")
            elif isinstance(msg, AIMessage):
                lines.append(f"助手: {msg.content}")
            elif isinstance(msg, SystemMessage):
                lines.append(f"系统: {msg.content}")
        return "\n".join(lines)

    async def _generate_summary(
        self,
        messages: List[Any],
    ) -> str:
        """
        调用 LLM 生成对话摘要。

        使用轻量级模型（qwen-turbo）快速生成，控制温度使输出稳定。
        摘要将用于替换被总结的历史消息，因此需要保留：
            - 用户的核心需求和意图
            - 已确认的关键事实和数据
            - 未解决或待跟进的问题
        """
        summary_llm = ChatOpenAI(
            model="qwen-turbo",
            openai_api_key=settings.api_key,
            openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0.3,
            max_tokens=500,
        )

        conversation_text = self._messages_to_conversation_text(messages)
        summary_prompt = SUMMARY_GENERATION_PROMPT.format(
            conversation_text=conversation_text
        )

        response = await summary_llm.ainvoke(summary_prompt)
        return response.content.strip()

    async def _generate_incremental_summary(
        self,
        old_summary: str,
        new_messages: List[Any],
    ) -> str:
        """
        调用 LLM 生成增量对话摘要。

        基于原有摘要和新增消息，生成更新后的摘要。
        """
        summary_llm = ChatOpenAI(
            model="qwen-turbo",
            openai_api_key=settings.api_key,
            openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0.3,
            max_tokens=500,
        )

        new_conversation_text = self._messages_to_conversation_text(new_messages)
        prompt = INCREMENTAL_SUMMARY_PROMPT.format(
            old_summary=old_summary,
            new_conversation_text=new_conversation_text,
        )

        response = await summary_llm.ainvoke(prompt)
        return response.content.strip()

    def _build_rag_chain(
        self,
        model: str,
        context: str
    ) -> Any:
        """
        构建 RAG 对话链
        
        :param model: 模型名称
        :param context: 检索到的上下文
        :return: LangChain 对话链
        """
        # 创建 LLM
        llm = ChatOpenAI(
            model=model,
            openai_api_key=settings.api_key,
            openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0.3,
            streaming=True,
        )
        
        # 构建 Prompt
        if context:
            # 有检索上下文：使用 RAG Prompt
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", RAG_SYSTEM_PROMPT_WITH_CONTEXT),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ])
        else:
            # 无检索上下文：使用普通对话 Prompt
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", RAG_SYSTEM_PROMPT_WITHOUT_CONTEXT),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ])
        
        # 创建对话链
        chain = prompt_template | llm
        
        return chain

    async def _stream_rag_response(
        self,
        chain: Any,
        user_input: str,
        context: str,
        mysql_history: MySQLChatMessageHistory
    ) -> AsyncGenerator[bytes, None]:
        """
        流式执行 RAG 对话
        
        :param chain: RAG 链
        :param user_input: 用户输入
        :param context: 检索到的上下文
        :param mysql_history: MySQL 历史存储
        :yield: SSE 格式的数据
        """
        try:
            # 构建压缩后的对话历史（阶段一 + 阶段二）
            compressed_history = await self._build_compressed_history(
                mysql_history
            )

            # Token 预估日志
            history_text = "\n".join(
                [m.content for m in compressed_history if hasattr(m, "content")]
            )
            est_tokens = self._estimate_tokens(user_input + context + history_text)
            logger.info(
                f"预估上下文 Token 数: {est_tokens}, "
                f"压缩后历史消息数: {len(compressed_history)}"
            )

            inputs = {
                "input": user_input,
                "context": context,
                "history": compressed_history,
            }
            
            # 使用 astream 获取流式响应
            full_response = ""
            async for chunk in chain.astream(inputs):
                # LangChain 的流式输出可能是 AIMessageChunk
                if hasattr(chunk, 'content'):
                    content = chunk.content
                    full_response += content
                    
                    # 发送内容块
                    yield f'data: {{"content": "{self._escape_json(content)}"}}\n\n'.encode('utf-8')
            
            # 保存用户消息
            await mysql_history.add_message(HumanMessage(content=user_input))
            
            # 保存 AI 消息
            await mysql_history.add_message(AIMessage(content=full_response))
            
            # 显式提交事务，确保消息持久化（StreamingResponse 场景下 get_db 的自动 commit 可能不生效）
            await self.db.commit()
            
            # 发送完成标记
            yield b'data: {"done": true}\n\n'
            
        except Exception as e:
            logger.error(f"流式 RAG 对话失败: {e}")
            yield f'data: {{"content": "流式输出失败：{str(e)}", "done": true}}\n\n'.encode('utf-8')

    def _escape_json(self, text: str) -> str:
        """转义 JSON 特殊字符"""
        return text.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')


def get_rag_conversation_service(db: AsyncSession) -> RAGConversationService:
    """
    获取 RAG 对话服务实例（FastAPI 依赖注入）。

    自动将 lifespan 阶段注入的全局混合检索器传入服务实例，
    无需修改路由层的依赖签名。

    :param db: 数据库会话
    :return: RAG 对话服务实例
    """
    return RAGConversationService(
        db_session=db,
        hybrid_retriever=_hybrid_retriever,
    )
