"""
知识库问答节点

复用现有检索流程（混合检索 + 语义缓存 + 压缩历史），通过 KnowledgeService 薄适配层接入。
流程：预计算 embedding → 语义缓存检查 → 混合检索 → 组装 RAG Prompt → LLM 流式生成 → 落库。
产出写入 state["rag_answer"]（并行场景下供 merge 节点合并）。
"""
import asyncio
import logging
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig

from core.config import settings
from agent.knowledge_service import KnowledgeService
from agent.stream_emitter import extract_reasoning, extract_text
from agent.streaming import put_content, put_thinking, get_sse_queue
from agent.state import AgentState
from agent.observability import metrics, node_trace

logger = logging.getLogger(__name__)

# 检索返回文档数（与旧接口一致）
RETRIEVE_TOP_K = 4

# 模拟流式输出时每块字符数（语义缓存命中场景）
CACHE_CHUNK_SIZE = 20


def _intents_key(state: AgentState) -> str:
    """
    意图维度键（多意图时排序拼接，保证 put/get 两侧一致）。

    例如 ["task","knowledge_base"] → "knowledge_base+task"。
    """
    intents = state.get("intents") or []
    return "+".join(sorted({str(i) for i in intents}))


def _cache_allowed(state: AgentState) -> bool:
    """
    本轮是否允许查答案缓存（§18.2 意图准入）。

    `cache_policy` 由入口节点每轮覆盖写入；老检查点可能没有该字段，
    此时按本轮意图现算一次，行为与新版一致。
    """
    try:
        from agent.intent_gate import CACHE_DENY, cache_policy_for

        policy = state.get("cache_policy")
        if policy is None:
            policy, _ = cache_policy_for(state.get("intents"))
        return policy != CACHE_DENY
    except Exception:
        # best-effort：策略读取异常时按「允许」处理，退回改动前行为
        return True


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

        # 1. query 双向量（稠密 + 稀疏）
        #    §18.5：优先复用入口节点预计算、经 `configurable` 传下来的向量，
        #    避免同一请求重复调用 embedding API。缺失时（审批恢复路径重建了
        #    config、或未开启预取）本节点自行计算。
        #    注意：**不再把向量回写 state** —— 1024 维向量进检查点会被逐轮
        #    序列化落库（撑大存储 + 拖慢 checkpoint），且上一轮残留的向量还会
        #    污染下一轮的意图判定。
        cfg = (config or {}).get("configurable") or {}
        query_embedding = cfg.get("query_embedding")
        query_sparse = cfg.get("query_sparse")
        if query_embedding is None or query_sparse is None:
            query_embedding, query_sparse = await ks.compute_embedding_with_sparse(
                user_input
            )

        # 2. 语义缓存检查（§18.2：先过意图准入，再查缓存）
        #    答案缓存命中会跳过检索与生成（不再落审计、不再溯源），
        #    因此必须先由本轮意图决定「能否使用」，而不是无条件查询。
        cached = None
        if _cache_allowed(state):
            cached = await ks.get_cached(
                user_input,
                username,
                query_embedding,
                intent=_intents_key(state),
                model=model,
                ttl_override=state.get("cache_ttl_seconds"),
            )
        else:
            metrics.incr("cache.skip.intent_deny")
            logger.info(
                "[Agent-知识库] 本轮意图禁用答案缓存（§18.2），跳过缓存查询"
            )
        if cached:
            logger.info(
                f"[Agent-知识库] 语义缓存命中: score={cached.score:.4f}"
            )
            answer = cached.answer
            # 模拟流式输出缓存回答
            for i in range(0, len(answer), CACHE_CHUNK_SIZE):
                await put_content(queue, answer[i : i + CACHE_CHUNK_SIZE], branch="knowledge")
            await ks.save_answer(conversation_id, user_input, answer)
            return {
                "rag_answer": answer,
                "cached_answer": answer,
                "context_docs": [],
                "context_text": "",
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
                # §21.2：思考过程走独立事件（前端分区域渲染），不混进正文
                reasoning = extract_reasoning(chunk)
                if reasoning:
                    await put_thinking(queue, reasoning)
                    metrics.incr("agent.reasoning.deltas")
                content = extract_text(chunk)
                if content:
                    buffer.append(content)
                    await put_content(queue, content, branch="knowledge")

        from agent.observability import call_with_timeout, get_trace_id

        # 阶段 5 加固：这里的 except 必须**同时覆盖超时与连接异常**。
        # 此前只捕获 asyncio.TimeoutError，导致 LLM 连接失败（APIConnectionError /
        # httpx2.ConnectError，如网络中断、代理 TLS 握手失败、5xx 重试耗尽）会直接
        # 落到最外层 except，把 buffer 里已流式推送给用户的片段**全部丢弃**并返回
        # rag_answer=None——注释宣称的「保留已生成的部分回答」在连接失败路径上是失效的。
        # 现在两条路径统一按「保留已生成内容」处理。
        #
        # 关键约定：**有部分产出时不得写入 error 字段**。finalize_node 见到 error 会
        # 覆盖 final_response（→「处理失败：…」），把用户已完整收到的回答变成错误文案；
        # 且 merge_node 会因 `not rag` 触发知识分支重执行、再打一次已失败的 LLM。
        # 因此 error 仅在「零产出」时写入，语义严格保持为「无可用结果」。
        stream_error: Optional[Exception] = None
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
        except Exception as e:
            stream_error = e
            logger.warning(
                f"[Agent-知识库] LLM 流式生成中断（保留已生成的部分回答）: "
                f"{type(e).__name__}: {e} trace_id={get_trace_id()}"
            )
            metrics.incr("agent.knowledge.stream_error")
        full_response = "".join(buffer)

        # 6. 回答落库 + 写语义缓存（best-effort）
        #    仅在确有产出时落库/写缓存：避免把空串写进会话历史与语义缓存。
        if full_response:
            await ks.save_answer(conversation_id, user_input, full_response)
            # §18.2：写入同样受意图准入约束——被 DENY 的意图不产出缓存条目，
            # 避免往集合里塞入永远命不中的脏数据。
            if query_embedding and _cache_allowed(state):
                await ks.put_cache(
                    user_input,
                    query_embedding,
                    full_response,
                    docs,
                    username,
                    intent=_intents_key(state),
                    model=model,
                    ttl_class=state.get("cache_ttl_seconds"),
                )
        elif stream_error is not None:
            # 零产出（如建连即失败）：无可用结果，如实上报，由 merge 转可读提示
            return {
                "rag_answer": None,
                "context_docs": docs,
                "context_text": context_text,
                "error": f"知识库生成失败：{type(stream_error).__name__}",
            }

        return {
            "rag_answer": full_response,
            "context_docs": docs,
            "context_text": context_text,
        }
    except Exception as e:
        logger.exception(f"[Agent-知识库] 节点执行失败: {e}")
        return {"rag_answer": None, "error": str(e)}
