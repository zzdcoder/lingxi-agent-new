"""
普通聊天节点

不使用 RAG 检索，复用 CHAT_SYSTEM_PROMPT + 最近历史进行流式生成。
产出写入 state["rag_answer"]（chat 为单分支兜底，经 merge 透传为最终回答）。
"""
import logging

from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from core.config import settings
from prompt.prompt_storage import CHAT_SYSTEM_PROMPT
from rag.memory_mysql import MySQLChatMessageHistory
from agent.streaming import put_content, get_sse_queue
from agent.state import AgentState

logger = logging.getLogger(__name__)

# 普通聊天使用的历史条数（与旧接口一致）
HISTORY_RECENT_NUM = 10


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

    try:
        # 1. 加载最近历史
        history_messages = []
        try:
            mysql_history = MySQLChatMessageHistory(
                session=db, conversation_id=conversation_id
            )
            history_messages = await mysql_history.aget_messages()
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
        )
        chain = prompt | llm

        full_response = ""
        async for chunk in chain.astream({"input": user_input}):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            if content:
                full_response += content
                await put_content(queue, content)

        # 4. 消息落库
        try:
            mysql_history = MySQLChatMessageHistory(
                session=db, conversation_id=conversation_id
            )
            from langchain_core.messages import HumanMessage, AIMessage
            await mysql_history.add_message(HumanMessage(content=user_input))
            await mysql_history.add_message(AIMessage(content=full_response))
            await db.commit()
        except Exception as e:
            logger.warning(f"[Agent-聊天] 消息落库失败（非致命）: {e}")

        return {"rag_answer": full_response}
    except Exception as e:
        logger.exception(f"[Agent-聊天] 节点执行失败: {e}")
        return {"rag_answer": None, "error": str(e)}
