"""
知识库问答节点

复用现有检索流程（混合检索 + 语义缓存 + 压缩历史），通过 KnowledgeService 薄适配层接入。
流程：预计算 embedding → 语义缓存检查 → 混合检索 → 组装 RAG Prompt → LLM 流式生成 → 落库。
产出写入 state["rag_answer"]（并行场景下供 merge 节点合并）。
"""
import asyncio
import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from core.config import settings
from agent.knowledge_service import KnowledgeService
from agent.streaming import put_content, get_sse_queue
from agent.state import AgentState
from agent.observability import node_trace

logger = logging.getLogger(__name__)

# 检索返回文档数（与旧接口一致）
RETRIEVE_TOP_K = 4

# 模拟流式输出时每块字符数（语义缓存命中场景）
CACHE_CHUNK_SIZE = 20


@node_trace("knowledge")
async def knowledge_node(
    state: AgentState, config: RunnableConfig | None = None
) -> dict:
    """
    知识库问答节点。

    :param state: 图状态
    :param config: 运行配置（configurable 携带 db 会话与 sse_queue）
    :return: 状态增量（rag_answer / context_docs / context_text 等）
    """
    queue = get_sse_queue(config)
    db = (config or {}).get("configurable", {}).get("db")

    conversation_id = state.get("conversation_id", "")
    username = state.get("username")
    model = state.get("model", "qwen-turbo")
    user_input = state["messages"][-1].content

    try:
        ks = KnowledgeService(db_session=db)

        # 1. 预计算 query 双向量（稠密 + 稀疏，一次 API 调用全链路复用）
        query_embedding, query_sparse = await ks.compute_embedding_with_sparse(
            user_input
        )
        # §16：把 query 稠密向量回写状态，供意图层的「向量就近判定」复用，
        # 避免为意图分类单独再付一次 embedding 调用（best-effort，不影响主流程）。
        embedding_updates = {"query_embedding": query_embedding} if query_embedding else {}

        # 2. 语义缓存检查（命中直接返回）
        cached = await ks.get_cached(user_input, username, query_embedding)
        if cached:
            logger.info(
                f"[Agent-知识库] 语义缓存命中: score={cached.score:.4f}"
            )
            answer = cached.answer
            # 模拟流式输出缓存回答
            for i in range(0, len(answer), CACHE_CHUNK_SIZE):
                await put_content(queue, answer[i : i + CACHE_CHUNK_SIZE])
            await ks.save_answer(conversation_id, user_input, answer)
            return {
                "rag_answer": answer,
                "cached_answer": answer,
                "context_docs": [],
                "context_text": "",
                **embedding_updates,
            }

        # 3. 混合检索 + 格式化上下文
        docs, context_text = await ks.retrieve(
            user_input, username, query_embedding,
            sparse=query_sparse, k=RETRIEVE_TOP_K,
        )
        logger.info(
            f"[Agent-知识库] 检索完成: {len(docs)} 个文档"
        )

        # 4. 获取压缩历史
        history = await ks.get_history(conversation_id)

        # 5. 组装 RAG Prompt 并流式生成。
        #    阶段 4 加固：rag/ 内部链路不改动（设计文档 §12 最小改动原则），
        #    在此对整段流式生成加整体超时兜底；超时时**保留已生成的部分回答**，
        #    避免 SSE 因模型挂死而永久沉默。
        rag = ks.rag
        chain = rag.build_rag_chain_public(model, context_text)
        inputs = {
            "input": user_input,
            "context": context_text,
            "history": history,
        }

        buffer: list = []

        async def _consume_stream() -> None:
            async for chunk in chain.astream(inputs):
                content = chunk.content if hasattr(chunk, "content") else str(chunk)
                if content:
                    buffer.append(content)
                    await put_content(queue, content)

        from agent.observability import call_with_timeout, get_trace_id

        try:
            await call_with_timeout(
                _consume_stream(),
                settings.agent_llm_timeout_seconds,
                "knowledge_llm",
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"[Agent-知识库] LLM 流式生成超时"
                f"（>{settings.agent_llm_timeout_seconds}s），使用已生成的部分回答 "
                f"trace_id={get_trace_id()}"
            )
        full_response = "".join(buffer)

        # 6. 回答落库 + 写语义缓存（best-effort）
        await ks.save_answer(conversation_id, user_input, full_response)
        if query_embedding:
            await ks.put_cache(
                user_input, query_embedding, full_response, docs, username
            )

        return {
            "rag_answer": full_response,
            "context_docs": docs,
            "context_text": context_text,
            **embedding_updates,
        }
    except Exception as e:
        logger.exception(f"[Agent-知识库] 节点执行失败: {e}")
        return {"rag_answer": None, "error": str(e)}
