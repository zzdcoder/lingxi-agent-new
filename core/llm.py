"""
统一 LLM 构造入口（支持思考过程提取）

**为什么需要这个模块**

`langchain_openai.BaseChatOpenAI` 的类注释明确声明（`chat_models/base.py`）：

    Non-standard response fields added by third-party providers (e.g.,
    `reasoning_content`) are **not** extracted. Use a provider-specific subclass.

它的 `_convert_delta_to_message_chunk` 只读取 `content` / `function_call` /
`tool_calls` 三个字段，Qwen3 / DeepSeek / GLM 等模型放在 `delta.reasoning_content` 里的
**思考过程被静默丢弃**——不报错、不告警，只是拿不到。

**为什么还能救**

openai SDK 的 `BaseModel` 配置是 `ConfigDict(extra="allow")`
（`openai/_models.py`），第三方字段**并未被丢弃**，仍留在
`chunk["choices"][0]["delta"]["reasoning_content"]` 里。
所以只需在「原始 chunk → LangChain 消息」的转换环节把它捞出来，
塞进 `additional_kwargs`，下游即可通过 `chunk.additional_kwargs["reasoning_content"]` 读取。

**请求侧**

DashScope 的 Qwen3 系列默认**不返回**思考过程，需显式下发
`extra_body={"enable_thinking": True}`。注意 `qwen-turbo` 等非思考模型传该参数
可能报错，故按模型名前缀白名单判定（`agent_reasoning_models`）。

**失败降级**

本模块的所有增强都是**尽力而为**：
- 模型不支持思考 → 不传 `extra_body`，行为与改造前完全一致；
- 提取过程抛异常 → 吞掉，正文照常返回（思考过程丢了不影响主链路）。
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_openai import ChatOpenAI

from core.config import settings

logger = logging.getLogger(__name__)

__all__ = ["ThinkingChatOpenAI", "get_chat_model", "supports_thinking"]


class ThinkingChatOpenAI(ChatOpenAI):
    """
    透传第三方 provider 思考过程的 ChatOpenAI。

    只重写 `_convert_chunk_to_generation_chunk`——这是「原始 SSE chunk dict」
    转为「LangChain 消息块」的唯一收口点，在此把 `reasoning_content` /
    `reasoning` 从 delta 捞出并写入 `additional_kwargs`。

    下游读取方式：

        async for chunk in chain.astream(...):
            r = (chunk.additional_kwargs or {}).get("reasoning_content")
    """

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: Optional[dict],
    ):
        generation = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        if generation is None:
            return generation
        try:
            choices = chunk.get("choices") or []
            delta = (choices[0] or {}).get("delta") if choices else None
            if isinstance(delta, dict):
                # reasoning_content 是主流字段（Qwen / DeepSeek / GLM / MiniMax）；
                # reasoning 是部分网关（OpenRouter 等）的别名，作为兜底。
                for key in ("reasoning_content", "reasoning"):
                    value = delta.get(key)
                    if value:
                        generation.message.additional_kwargs[key] = value
                        break
        except Exception:
            # 提取失败绝不影响正文 —— 思考过程是增强项，不是必需品
            logger.debug("[LLM] 思考过程提取失败（忽略）", exc_info=True)
        return generation


def supports_thinking(model: str) -> bool:
    """
    判断模型是否支持思考过程（按配置的前缀白名单匹配）。

    白名单而非黑名单：非思考模型传 `enable_thinking` 可能直接报错，
    宁可不传（行为与改造前一致），也不要为了少数模型破坏多数请求。

    :param model: 模型名（如 qwen-plus / qwen-turbo）
    :return: 是否支持思考过程
    """
    if not model:
        return False
    prefixes = [
        p.strip().lower()
        for p in (settings.agent_reasoning_models or "").split(",")
        if p.strip()
    ]
    name = model.lower()
    return any(name.startswith(prefix) for prefix in prefixes)


def get_chat_model(
    model: str,
    *,
    temperature: float = 0.7,
    streaming: bool = True,
    timeout: Optional[float] = None,
    enable_thinking: Optional[bool] = None,
    **kwargs: Any,
) -> ChatOpenAI:
    """
    构造对话模型实例（全项目统一入口，支持思考过程）。

    :param model: 模型名
    :param temperature: 采样温度
    :param streaming: 是否流式
    :param timeout: 单次推理超时（秒），缺省取 `agent_llm_timeout_seconds`
    :param enable_thinking: 是否开启思考过程；None 时按配置与模型白名单自动判定
    :param kwargs: 其余透传给 ChatOpenAI 的参数
    :return: ChatOpenAI 实例（已具备思考过程提取能力）
    """
    resolved_timeout = timeout if timeout is not None else settings.agent_llm_timeout_seconds

    if enable_thinking is None:
        enable_thinking = settings.agent_reasoning_enabled and supports_thinking(model)

    params: dict[str, Any] = {
        "model": model,
        "openai_api_key": settings.api_key,
        "openai_api_base": settings.llm_base_url,
        "temperature": temperature,
        "streaming": streaming,
        "timeout": resolved_timeout,
    }

    # 思考过程开关：仅对白名单模型下发。不支持该参数的模型会忽略它，
    # 但为降低风险，非白名单模型一律不传。
    if enable_thinking:
        extra_body = dict(kwargs.pop("extra_body", None) or {})
        extra_body.setdefault("enable_thinking", True)
        params["extra_body"] = extra_body

    params.update(kwargs)
    return ThinkingChatOpenAI(**params)
