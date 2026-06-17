"""
RAG 对话服务

集成知识检索的对话服务，支持：
1. 向量数据库检索
2. 上下文增强
3. 带 Memory 的对话
"""
import json
import logging
import time
from typing import List, Optional, Dict, Any, AsyncGenerator
import asyncio

from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from rag.memory_mysql import MySQLChatMessageHistory
from embeddings.embedding_deal import DashScopeEmbedding
from core.config import settings
from embeddings.embedding_deal import EmbeddingHandler

logger = logging.getLogger(__name__)


class RAGConversationService:
    """
    RAG 对话服务
    
    结合向量数据库检索和对话记忆，提供知识增强的对话能力
    """

    def __init__(self, db_session: AsyncSession):
        """
        初始化 RAG 对话服务
        
        :param db_session: SQLAlchemy 异步会话
        """
        self.db = db_session
        self.embeddings = DashScopeEmbedding(api_key=settings.api_key)
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
        检索相关知识

        :param query: 查询文本
        :param k: 返回文档数量
        :return: 相关文档列表
        """
        if not self.vectorstore:
            logger.warning("向量数据库未初始化，跳过检索")
            return []

        try:
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
            # 使用异步线程执行同步检索
            docs = await asyncio.to_thread(
                self.vectorstore.similarity_search,
                query,
                k=k,
                filter=filter_obj
            )
            logger.info(f"向量检索完成: 查询='{query}', 找到 {len(docs)} 个文档")
            for idx, doc in enumerate(docs, 1):
                content_preview = doc.page_content[:200].replace('\n', ' ')
                logger.info(f"  [检索结果{idx}] {content_preview}{'...' if len(doc.page_content) > 200 else ''}")
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
                ("system", """你是一个专业的智能助手。基于以下检索到的知识库内容回答用户问题。

检索到的相关知识：
{context}

回答要求：
1. 优先基于检索到的内容回答
2. 如果检索内容不足以回答问题，可以补充你的知识
3. 回答要准确、简洁、有条理
4. 如果不确定，请诚实地告诉用户"""),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ])
        else:
            # 无检索上下文：使用普通对话 Prompt
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", """你是一个专业的智能助手。请根据用户的提问提供准确、有帮助的回答。

回答要求：
1. 回答要准确、简洁、有条理
2. 如果不确定，请诚实地告诉用户
3. 必要时可以举例说明"""),
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
            # 构建输入
            history = await mysql_history.aget_messages()
            inputs = {
                "input": user_input,
                "context": context,
                "history": history,
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
    获取 RAG 对话服务实例（FastAPI 依赖注入）
    
    :param db: 数据库会话
    :return: RAG 对话服务实例
    """
    return RAGConversationService(db_session=db)
