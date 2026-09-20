"""
普通聊天节点

不使用 RAG 检索，复用 CHAT_SYSTEM_PROMPT + 最近历史进行流式生成。
产出写入 state["rag_answer"]（chat 为单分支兜底，经 merge 透传为最终回答）。

§16 优化：新增两条**零 LLM 直达**路径——
1. **斜杠命令回执**：`state["slash_handoff"]` 存在时直接流式输出本地文案，
   不构造 Prompt、不调模型（指令类输入的首字延迟降到毫秒级）；
2. **上一轮意图继承**：加载历史时顺带取回 `last_intents`（从最近一条 assistant
   消息的 metadata 解析），写回 state 供意图节点判断短追问。
"""
import json
import logging

from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from core.config import settings
from prompt.prompt_storage import CHAT_SYSTEM_PROMPT
from rag.memory_mysql import MySQLChatMessageHistory
from agent.streaming import put_content, get_sse_queue
from agent.state import AgentState
from agent.observability import node_trace

logger = logging.getLogger(__name__)

# 普通聊天使用的历史条数（与旧接口一致）
HISTORY_RECENT_NUM = 10

# 会话内意图继承的回溯条数（找最近一条带 intents 元数据的 assistant 消息）
_LAST_INTENT_LOOKBACK = 6

# 斜杠回执的模拟流式分片大小（保持前端渲染体验一致）
_HANDOFF_CHUNK = 24


@node_trace("chat")
async def chat_node(
    state: AgentState, config: RunnableConfig | None = None
) -> dict:
    """
    普通聊天节点。

    :param state: 图状态
    :param config: 运行配置（configurable 携带 db 会话与 sse_queue）
    :return: 状态增量（rag_answer）
    """
    queue = get_sse_queue(config)
    db = (config or {}).get("configurable", {}).get("db")

    conversation_id = state.get("conversation_id", "")
    model = state.get("model", "qwen-turbo")
    user_input = state["messages"][-1].content

    # ---- 斜杠命令回执：本地文案，零 LLM 直达 ----
    handoff = state.get("slash_handoff")
    if handoff:
        for i in range(0, len(handoff), _HANDOFF_CHUNK):
            await put_content(queue, handoff[i : i + _HANDOFF_CHUNK])
        logger.info(
            f"[Agent-聊天] 斜杠回执本地输出: /{state.get('slash_command')} "
            f"len={len(handoff)}"
        )
        return {"rag_answer": handoff}

    try:
        # 1. 加载最近历史 + 解析上一轮意图（供意图层短追问继承）
        history_messages = []
        last_intents = state.get("last_intents")
        try:
            mysql_history = MySQLChatMessageHistory(
                session=db, conversation_id=conversation_id
            )
            history_messages = await mysql_history.aget_messages()
            if not last_intents:
                last_intents = _extract_last_intents(history_messages)
        except Exception as e:
            logger.warning(f"[Agent-聊天] 加载历史失败（忽略）: {e}")

        # 2. 构建 Prompt（系统提示 + 最近历史 + 当前输入）
        prompt_messages = [("system", CHAT_SYSTEM_PROMPT)]
        for msg in history_messages[-HISTORY_RECENT_NUM:]:
            if hasattr(msg, "type") and msg.type == "human":
                prompt_messages.append(("human", msg.content))
            elif hasattr(msg, "type") and msg.type == "ai":
                prompt_messages.append(("ai", msg.content))
        prompt_messages.append(("human", "{input}"))
        prompt = ChatPromptTemplate.from_messages(prompt_messages)

        # 3. 创建 LLM 并流式生成
        llm = ChatOpenAI(
            model=model,
            openai_api_key=settings.api_key,
            openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0.7,
            streaming=True,
            # 阶段 4 加固：流式会话若无超时，模型侧挂死会让 SSE 永久沉默
            timeout=settings.agent_llm_timeout_seconds,
        )
        chain = prompt | llm

        full_response = ""
        async for chunk in chain.astream({"input": user_input}):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            if content:
                full_response += content
                await put_content(queue, content)

        # 4. 消息落库（assistant 消息附带本轮意图元数据，供下一轮继承）
        try:
            mysql_history = MySQLChatMessageHistory(
                session=db, conversation_id=conversation_id
            )
            from langchain_core.messages import HumanMessage, AIMessage
            await mysql_history.add_message(HumanMessage(content=user_input))
            await mysql_history.add_message(
                AIMessage(
                    content=full_response,
                    additional_kwargs=_intent_metadata(last_intents),
                )
            )
            await db.commit()
        except Exception as e:
            logger.warning(f"[Agent-聊天] 消息落库失败（非致命）: {e}")

        out = {"rag_answer": full_response}
        if last_intents:
            out["last_intents"] = last_intents
        return out
    except Exception as e:
        logger.exception(f"[Agent-聊天] 节点执行失败: {e}")
        return {"rag_answer": None, "error": str(e)}


# =============================================================================
# 会话内意图继承（§16 规则层「短追问继承」的数据来源）
# =============================================================================

def _intent_metadata(intents) -> dict:
    """构造写入 assistant 消息的意图元数据（空值不落库）。"""
    if not intents:
        return {}
    return {"lingxi_intents": list(intents)}


def _extract_last_intents(history_messages) -> list:
    """
    从历史消息中回溯最近一轮的意图标签。

    只解析 `additional_kwargs["lingxi_intents"]`（由本节点写入），
    解析失败/缺失一律返回空列表——继承是**加速手段**，取不到就正常走 LLM。
    """
    try:
        for msg in reversed(list(history_messages)[-_LAST_INTENT_LOOKBACK:]):
            if getattr(msg, "type", "") != "ai":
                continue
            kwargs = getattr(msg, "additional_kwargs", None) or {}
            raw = kwargs.get("lingxi_intents")
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except Exception:
                    continue
            if isinstance(raw, list) and raw:
                return [str(i) for i in raw if i]
        return []
    except Exception:
        return []
