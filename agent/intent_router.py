"""
意图识别器

对用户输入进行多标签意图分类（knowledge_base / task / chat），
使用 ChatOpenAI 的官方结构化输出 API，并内置降级与归一化策略：
- LLM 调用失败 / 输出非法 → 降级为 ["chat"]，保证主流程可用；
- 置信度低于阈值 → 降级为 ["chat"]；
- 多意图归一化：去重、丢弃 chat（仅作兜底）、异常组合按优先级收敛。
"""
import logging
from typing import Literal

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from core.config import settings
from prompt.prompt_storage import INTENT_CLASSIFICATION_PROMPT

logger = logging.getLogger(__name__)

# 意图集合（结构化输出约束）
INTENT_TYPES = ("knowledge_base", "task", "chat")

# 置信度兜底阈值：低于该值视为低置信度，路由到 chat
CONFIDENCE_THRESHOLD = 0.5


class IntentResult(BaseModel):
    """意图识别结构化输出（多标签：同一输入可命中多个意图）"""
    intents: list[Literal["knowledge_base", "task", "chat"]] = Field(
        description="意图列表"
    )
    reason: str = Field(description="分类依据")
    confidence: float = Field(description="整体置信度 0~1")


class IntentRouter:
    """意图识别器：多标签 LLM 结构化输出 + 降级策略"""

    def __init__(self, model: str = "qwen-turbo"):
        self._llm = ChatOpenAI(
            model=model,
            openai_api_key=settings.api_key,
            openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0,
        ).with_structured_output(IntentResult)

    async def classify(self, user_input: str) -> IntentResult:
        """
        对用户输入进行多标签意图分类。

        失败或异常时降级为 ["chat"]，保证主流程可用。

        :param user_input: 用户输入
        :return: IntentResult（intents 已归一化）
        """
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", INTENT_CLASSIFICATION_PROMPT),
                ("human", "{input}"),
            ])
            result = await (prompt | self._llm).ainvoke({"input": user_input})
            if result is None:
                logger.warning("意图识别返回空结果，降级为普通聊天")
                return IntentResult(
                    intents=["chat"], reason="意图识别返回空结果，降级为普通聊天",
                    confidence=0,
                )
            if result.confidence < CONFIDENCE_THRESHOLD:
                logger.info(
                    f"意图识别置信度 {result.confidence:.2f} 低于阈值"
                    f"{CONFIDENCE_THRESHOLD}，降级为普通聊天"
                )
                return IntentResult(
                    intents=["chat"], reason="置信度低于阈值",
                    confidence=result.confidence,
                )
            return self._normalize(result)
        except Exception as e:
            logger.warning(f"意图识别异常（降级为普通聊天）: {e}")
            return IntentResult(
                intents=["chat"], reason="意图识别异常，降级为普通聊天",
                confidence=0,
            )

    @staticmethod
    def _normalize(result: IntentResult) -> IntentResult:
        """
        意图归一化：去重、丢弃 chat、异常多意图按优先级收敛。

        - ["task","knowledge_base"] → 保留两个，触发并行分支
        - 含 chat 的其它组合 → 丢弃 chat
        - 空列表 / 三个全命中 → 按 task > knowledge_base > chat 收敛
        """
        intents = list(dict.fromkeys(result.intents))          # 去重保序
        if "chat" in intents and len(intents) > 1:             # chat 不参与并行
            intents.remove("chat")
        if not intents or len(intents) > 2:
            # 异常组合收敛：task 优先于 knowledge_base，chat 仅兜底
            intents = [i for i in ("task", "knowledge_base") if i in intents] or ["chat"]
        return IntentResult(
            intents=intents, reason=result.reason, confidence=result.confidence
        )
