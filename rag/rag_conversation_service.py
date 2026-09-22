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
from qdrant_client.models import SparseVector
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from rag.hybrid_retriever import HybridRetriever, CrossEncoderReranker
from rag.memory_mysql import MySQLChatMessageHistory, MySQLConversationSummaryMemory
from rag.semantic_cache import SemanticCache, get_semantic_cache
from prompt.prompt_storage import (
    RAG_SYSTEM_PROMPT_WITH_CONTEXT,
    RAG_SYSTEM_PROMPT_WITHOUT_CONTEXT,
    SUMMARY_GENERATION_PROMPT,
    CONVERSATION_SUMMARY_PREFIX,
    INCREMENTAL_SUMMARY_PROMPT,
)
from embeddings.embedding_deal import DashScopeEmbedding
from core.config import settings
from embeddings.embedding_deal import EmbeddingHandler

logger = logging.getLogger(__name__)

# =============================================================================
# 全局混合检索器（由 app/main.py lifespan 在启动时注入）
# =============================================================================

_hybrid_retriever: Optional[HybridRetriever] = None
"""
全局混合检索器单例。

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

    由外部模块（如 file_process）在知识库变更后调用。
    """
    global _hybrid_retriever
    _hybrid_retriever = retriever
    logger.info("全局混合检索器已注入 RAGConversationService")


def get_hybrid_retriever() -> Optional[HybridRetriever]:
    """获取全局混合检索器实例。"""
    return _hybrid_retriever


def init_hybrid_retriever() -> HybridRetriever:
    """
    惰性初始化全局混合检索器。

    首次调用时创建 HybridRetriever 实例，并预热 Cross-Encoder 重排序模型。
    后续调用直接返回已有实例，避免重复初始化。

    Returns:
        全局 HybridRetriever 实例。
    """
    global _hybrid_retriever
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
            batch_size=getattr(settings, "reranker_batch_size", 16),
        )
        # 启动时立即加载模型，避免首次请求额外耗时 ~1.8s
        reranker._lazy_load()
        logger.info(f"Cross-Encoder 重排序器已启用并加载完成: {settings.reranker_model}")

    # 稀疏向量（text-embedding-v4）与稠密向量均存储于 Qdrant，无需额外的 BM25 索引管理器
    _hybrid_retriever = HybridRetriever(
        vectorstore=vectorstore,
        reranker=reranker,
        rerank_top_k=12,  # 减少送入重排序器的候选数，大幅降低推理耗时
    )

    logger.info("混合检索器初始化完成")
    return _hybrid_retriever


async def invalidate_kb_semantic_cache() -> None:
    """
    知识库变更后清除语义缓存（保证回答一致性）。

    稀疏向量与稠密向量在文档写入时已一并落库，检索无需任何索引重建；
    本函数仅在知识库数据变更后失效语义缓存，避免旧回答被复用。
    """
    try:
        semantic_cache = get_semantic_cache()
        if semantic_cache:
            cleared = await semantic_cache.invalidate_all()
            logger.info(f"知识库变更，语义缓存已全量清除: {cleared} 条")
    except Exception as e:
        logger.error(f"知识库变更后语义缓存清除失败: {e}")
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
        1. 语义缓存检查（命中则直接返回缓存回答）
        2. 检索相关知识
        3. 构建增强 Prompt
        4. 调用 LLM 生成回答
        5. 保存到 Memory + 写入缓存
        
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

            # 2. 预计算 query embedding（只算一次，全链路复用：稠密 + 稀疏）
            query_embedding = None
            query_sparse = None
            try:
                query_embedding, query_sparse = await asyncio.to_thread(
                    self.embeddings.embed_query_with_sparse, user_input
                )
            except Exception as e:
                logger.warning(f"预计算 query embedding 失败: {e}")

            # 3. 语义缓存检查（复用已计算的 embedding）
            semantic_cache = get_semantic_cache()
            if semantic_cache:
                # §18：与 Agent 链路共用同一缓存集合，必须带上同样的缓存维度，
                # 否则写入的条目因缺少 schema_version 永远命中不了（脏数据）。
                # 本接口是纯知识库问答，意图恒为 knowledge_base，可与 Agent 的
                # knowledge_base 单分支条目互通。
                cached = await semantic_cache.get(
                    user_input, login_username,
                    precomputed_embedding=query_embedding,
                    intent="knowledge_base",
                    model=model,
                    kb_version=str(settings.cache_kb_version or "v1"),
                    prompt_version=str(settings.cache_prompt_version or "1"),
                )
                if cached:
                    logger.info(
                        f"语义缓存命中: score={cached.score:.4f}, "
                        f"cached_query='{cached.query_text[:40]}...'"
                    )
                    async for chunk in self._stream_cached_response(
                        cached.answer, conversation_id, user_input
                    ):
                        yield chunk
                    elapsed = time.time() - start_time
                    logger.info(f"RAG 对话完成（缓存命中）: 耗时={elapsed:.2f}s")
                    return

            # 4. 检索相关知识
            context_docs = await self._retrieve_context(
                user_input, k=10, login_username=login_username,
                precomputed_embedding=query_embedding,
                precomputed_sparse=query_sparse,
            )
            context_text = self._format_context(context_docs)
            
            # 5. 创建 MySQL Memory
            mysql_history = MySQLChatMessageHistory(
                session=self.db,
                conversation_id=conversation_id
            )
            
            # 6. 构建 RAG 链
            rag_chain = self._build_rag_chain(model, context_text)
            
            # 7. 执行对话（流式），收集完整回答用于缓存写入
            # 使用共享容器收集完整回答（由 _stream_rag_response 内部填充）
            response_collector = [""]
            async for chunk in self._stream_rag_response(
                rag_chain, user_input, context_text, mysql_history,
                response_collector=response_collector,
            ):
                yield chunk
            
            full_response = response_collector[0]
            
            # 8. 写入语义缓存（新增）
            if semantic_cache and full_response and query_embedding:
                try:
                    await semantic_cache.put(
                        query=user_input,
                        query_embedding=query_embedding,
                        answer=full_response,
                        context_docs=context_docs,
                        login_username=login_username,
                        intent="knowledge_base",
                        model=model,
                        kb_version=str(settings.cache_kb_version or "v1"),
                        prompt_version=str(settings.cache_prompt_version or "1"),
                        ttl_class=int(settings.cache_ttl_knowledge_seconds or 86400),
                    )
                except Exception as e:
                    logger.warning(f"写入语义缓存失败（非致命）: {e}")
            
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
        login_username: str = None,
        precomputed_embedding: Optional[List[float]] = None,
        precomputed_sparse: Optional[SparseVector] = None,
    ) -> List[Any]:
        """
        检索相关知识（混合检索模式）。

        检索策略优先级：
            1. 若 hybrid_retriever 已初始化，执行稀疏向量 + 稠密向量混合检索，
               经 RRF 融合后按 created_at 倒序排列（新知识优先）；
            2. 若 hybrid_retriever 未初始化（如启动失败或依赖未安装），
               自动降级为纯稠密向量检索，保证服务可用性。

        :param query: 查询文本
        :param k: 返回文档数量
        :param login_username: 当前登录用户名，用于权限过滤
        :param precomputed_embedding: 预计算的 query embedding（供缓存复用）
        :param precomputed_sparse: 预计算的 query 稀疏向量（与稠密向量同一次调用生成）
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

            # -----------------------------------------------------------------
            # 步骤 2: 执行检索（混合检索优先，降级到纯稠密向量检索）
            # 稀疏路与稠密路共用 filter_obj，保证两路权限过滤逻辑一致
            # -----------------------------------------------------------------
            if self.hybrid_retriever:
                logger.info(f"【混合检索】查询='{query[:60]}...', top_k={k}")
                docs = await self.hybrid_retriever.aretrieve(
                    query=query,
                    top_k=k,
                    sparse_top_k=k * 5,   # 稀疏路召回数取最终需求的 5 倍，为 RRF 预留候选池
                    vector_top_k=k * 5,
                    filter_obj=filter_obj,
                    precomputed_embedding=precomputed_embedding,
                    precomputed_sparse=precomputed_sparse,
                )
            else:
                logger.info(f"【纯向量检索-降级模式】查询='{query[:60]}...', top_k={k}")
                if precomputed_embedding is not None:
                    docs = await asyncio.to_thread(
                        self.vectorstore.similarity_search_by_vector,
                        precomputed_embedding,
                        k=k,
                        filter=filter_obj,
                    )
                else:
                    docs = await asyncio.to_thread(
                        self.vectorstore.similarity_search,
                        query,
                        k=k,
                        filter=filter_obj,
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
    # Agent 图节点复用用的公开薄方法
    # 仅委托现有私有方法，不改动既有逻辑（供 agent/knowledge_service.py 调用）
    # -----------------------------------------------------------------

    async def retrieve_context_public(
        self,
        query: str,
        k: int = 4,
        login_username: Optional[str] = None,
        precomputed_embedding: Optional[List[float]] = None,
        precomputed_sparse: Optional[SparseVector] = None,
    ) -> List[Any]:
        """
        Agent 图节点复用：混合检索（委托 _retrieve_context）。

        :param query: 查询文本
        :param k: 返回文档数量
        :param login_username: 当前登录用户名，用于权限过滤
        :param precomputed_embedding: 预计算的 query embedding
        :param precomputed_sparse: 预计算的 query 稀疏向量
        :return: 相关文档列表
        """
        return await self._retrieve_context(
            query, k=k, login_username=login_username,
            precomputed_embedding=precomputed_embedding,
            precomputed_sparse=precomputed_sparse,
        )

    def format_context_public(self, docs: List[Any]) -> str:
        """
        Agent 图节点复用：格式化检索结果（委托 _format_context）。

        :param docs: 文档列表
        :return: 格式化后的上下文字符串
        """
        return self._format_context(docs)

    async def get_compressed_history_public(
        self,
        conversation_id: str,
    ) -> List[Any]:
        """
        Agent 图节点复用：获取压缩后的对话历史（委托 _build_compressed_history）。

        :param conversation_id: 会话 ID
        :return: 压缩/摘要后的消息列表
        """
        mysql_history = MySQLChatMessageHistory(
            session=self.db,
            conversation_id=conversation_id,
        )
        return await self._build_compressed_history(mysql_history)

    async def save_messages_public(
        self,
        conversation_id: str,
        user_input: str,
        answer: str,
    ) -> None:
        """
        Agent 图节点复用：将用户输入与回答写入会话历史并提交事务。

        :param conversation_id: 会话 ID
        :param user_input: 用户输入
        :param answer: 回答内容
        """
        mysql_history = MySQLChatMessageHistory(
            session=self.db,
            conversation_id=conversation_id,
        )
        await mysql_history.add_message(HumanMessage(content=user_input))
        await mysql_history.add_message(AIMessage(content=answer))
        await self.db.commit()

    def build_rag_chain_public(self, model: str, context: str) -> Any:
        """
        Agent 图节点复用：构建 RAG 对话链（委托 _build_rag_chain）。

        :param model: 模型名
        :param context: 检索到的上下文（为空时走无上下文 Prompt）
        :return: LangChain 对话链
        """
        return self._build_rag_chain(model, context)

    # -----------------------------------------------------------------
    # 上下文压缩相关方法
    # -----------------------------------------------------------------

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """
        粗略估算 Token 数量。

        策略：中文字符按 1倍估算，英文单词按 1.3倍估算，
        其余字符按 1 倍估算。此估算偏保守，实际值通常更小。
        """
        import re
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_words = len(re.findall(r'[a-zA-Z]+', text))
        # 标点、数字、空格、其他符号等
        other_chars = len(re.findall(r'[^\u4e00-\u9fffa-zA-Z\s]', text))  # 标点符号
        spaces = len(re.findall(r'\s', text))

        # 中文≈1 token/字，英文单词≈1.3 token/词，标点≈1 token/个，空格≈0.5 token/个
        return int(chinese_chars * 1.0 + english_words * 1.3 + other_chars * 1.0 + spaces * 0.5)

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
            # §19.1：摘要不需要长链推理，关掉思考模式，避免 500 上限被
            # reasoning_tokens 吃满导致摘要被截断成空串（与意图判别同一类隐患）
            extra_body={"enable_thinking": False},
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
            # §19.1：同上，关闭思考模式防止输出上限被 reasoning 吃满
            extra_body={"enable_thinking": False},
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
        # §21.2：走统一入口 —— RAG 分支也要能拿到思考过程。裸 ChatOpenAI 会把
        # `reasoning_content` 静默丢弃（`_convert_delta_to_message_chunk` 只读
        # content/tool_calls），ThinkingChatOpenAI 才把它透传到 additional_kwargs。
        # 对本文件其余 ChatOpenAI（摘要等）**不动**：那些是内部中间产物，
        # 不但不需要思考过程，还刻意 `enable_thinking=False` 防止输出被吃满。
        from core.llm import get_chat_model

        llm = get_chat_model(
            model,
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
        mysql_history: MySQLChatMessageHistory,
        response_collector: Optional[list] = None,
    ) -> AsyncGenerator[bytes, None]:
        """
        流式执行 RAG 对话
        
        :param chain: RAG 链
        :param user_input: 用户输入
        :param context: 检索到的上下文
        :param mysql_history: MySQL 历史存储
        :param response_collector: 可选的共享容器 [str]，用于向调用方传递完整回答
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
            
            # 将完整回答写入共享容器，供 chat_with_rag 缓存写入使用
            if response_collector is not None:
                response_collector[0] = full_response
            
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

    async def _stream_cached_response(
        self,
        cached_answer: str,
        conversation_id: str,
        user_input: str,
    ) -> AsyncGenerator[bytes, None]:
        """
        将缓存的回答以 SSE 流式格式输出。

        模拟流式输出效果：按句子/固定长度分块发送，
        同时保存到对话历史以保持完整性。

        :param cached_answer: 缓存的完整回答文本
        :param conversation_id: 会话 ID
        :param user_input: 用户原始问题
        :yield: SSE 格式的数据块
        """
        try:
            # 按固定长度分块模拟流式输出（每块约 20 个字符）
            chunk_size = 20
            for i in range(0, len(cached_answer), chunk_size):
                chunk = cached_answer[i:i + chunk_size]
                yield f'data: {{"content": "{self._escape_json(chunk)}"}}\n\n'.encode('utf-8')
                await asyncio.sleep(0.01)  # 微小延迟模拟流式效果

            # 保存到对话历史
            mysql_history = MySQLChatMessageHistory(
                session=self.db,
                conversation_id=conversation_id
            )
            await mysql_history.add_message(HumanMessage(content=user_input))
            await mysql_history.add_message(AIMessage(content=cached_answer))
            await self.db.commit()

            # 发送完成标记
            yield b'data: {"done": true}\n\n'

        except Exception as e:
            logger.error(f"缓存流式输出失败: {e}")
            yield f'data: {{"content": "缓存输出失败：{str(e)}", "done": true}}\n\n'.encode('utf-8')


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
