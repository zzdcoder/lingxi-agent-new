"""
任务分支流式统一派发器（§21.3）

**它解决什么问题**

`task_node` 从 `ainvoke`（跑完一次性返回）改为 `astream`（逐帧产出）后，
流式内容不再只来自一处，而是混杂着：思考过程、正文 delta、工具调用进度。
若各处自行 `put_content`，会立刻遇到三个问题：

1. **审批重入导致的重复推送**。`task_node` 因 HITL 审批中断后**会被重入**
   （`_find_running_task` 复用同一条 running 记录，见 task_node.py），
   重入后模型会**重新生成一遍**此前已推送过的内容（尤其是审批前那段「我准备
   删除这个用户」的说明），用户会看到同样的话出现两次。
2. **旧流的残留 delta 串入新流**。超时重试 / 重入时，上一轮流可能仍有尾帧在
   事件循环里排队，它们会和新流的内容**交错**输出，得到一段语义错乱的文本。
3. **思考块跨帧被切断**。详见 `agent/think_scrubber.py` 的模块文档。

**设计（三件套，逐条对应上述问题）**

| 机制 | 解决的问题 | 实现 |
| --- | --- | --- |
| 单写者栅栏 | 问题 2 | `claim()` 递增 token；`_fenced()` 判定非当前 token 的帧一律丢弃 |
| 已推送内容去重 | 问题 1 | `_delivered` 记录已推送片段的归一化指纹 |
| 有状态思考过滤 | 问题 3 | 内部持有 `StreamingThinkScrubber` |

**与 Hermes 的对应关系**

- 单写者栅栏：Hermes `agent/stream_delivery.py::_claim_stream_writer` /
  `_stream_writer_superseded`（其 issue #65991）；
- 去重：Hermes `_delivered_interim_texts` / `_interim_text_was_delivered`；
- 思考过滤：Hermes `agent/think_scrubber.py`。

**失败降级**

本模块是**增强层**：任何内部异常都不应向调用方抛出，导致任务执行失败。
派发失败最多是「用户少看到一段流式内容」，绝不能是「任务没跑」。
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from agent.streaming import put_content, put_thinking
from agent.think_scrubber import StreamingThinkScrubber

logger = logging.getLogger(__name__)

__all__ = ["StreamEmitter", "normalize_fragment", "extract_reasoning", "extract_text"]

# 用于去重比对的空白归一化（连续空白折叠为单个空格）
_WS_RE = re.compile(r"\s+")


def extract_reasoning(chunk) -> str:
    """
    从流式 chunk 中取出**思考过程**文本（§21.2）。

    四级兜底（对齐 Hermes `agent_runtime_helpers.extract_reasoning`，只保留本项目
    用得到的层级）：

    1. `additional_kwargs["reasoning_content"]` —— `ThinkingChatOpenAI` 注入的主路径
       （Qwen / DeepSeek / GLM / MiniMax 都走这个字段）；
    2. `additional_kwargs["reasoning"]` —— 部分网关（OpenRouter 等）的别名；
    3. content 里的 typed block `{"type": "thinking", "thinking": "..."}`
       —— Anthropic 风格；
    4. 都没有 → 空串。

    任何异常都吞掉：思考过程是**增强项**，取不到绝不影响正文。

    :param chunk: LangChain 流式 chunk（AIMessageChunk）
    :return: 思考过程增量文本（无则空串）
    """
    try:
        kwargs = getattr(chunk, "additional_kwargs", None) or {}
        for key in ("reasoning_content", "reasoning"):
            value = kwargs.get(key)
            if isinstance(value, str) and value:
                return value
        content = getattr(chunk, "content", None)
        if isinstance(content, list):
            parts = [
                blk.get("thinking")
                for blk in content
                if isinstance(blk, dict) and blk.get("type") == "thinking"
            ]
            joined = "".join(p for p in parts if isinstance(p, str) and p)
            if joined:
                return joined
    except Exception:
        logger.debug("[Agent-Stream] 思考过程提取失败（忽略）", exc_info=True)
    return ""


def extract_text(chunk) -> str:
    """
    从流式 chunk 中取出**正文**增量（非字符串 content 需甄别）。

    - `content` 是 str：直接返回（绝大多数情况）；
    - `content` 是 list（typed blocks）：只拼 `type=="text"` 的块，
      **跳过** `type=="thinking"` 的块 —— 否则思考过程会被当作正文混排进气泡。

    :param chunk: LangChain 流式 chunk
    :return: 正文增量文本（无则空串）
    """
    content = getattr(chunk, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        try:
            return "".join(
                blk.get("text", "")
                for blk in content
                if isinstance(blk, dict) and blk.get("type") == "text"
            )
        except Exception:
            return ""
    return ""


def normalize_fragment(text: str) -> str:
    """把片段归一化为去重指纹（折叠空白、去首尾）。"""
    return _WS_RE.sub(" ", text).strip() if isinstance(text, str) else ""


class StreamEmitter:
    """
    任务分支流式统一派发器。

    典型用法（每进入一次 task_node，含审批恢复重入都要走一遍）::

        emitter = StreamEmitter(queue, task_execution_id=task_execution_id)
        token = emitter.claim()                      # 抢占写者身份，旧流全部失效
        async for mode, payload in agent.astream(...):
            await emitter.emit_thinking(r, token)
            await emitter.emit_content(text, token)
        await emitter.flush(token)                   # 吐出被扣留的正常尾巴
    """

    def __init__(self, queue, *, task_execution_id: str = "") -> None:
        """
        :param queue: SSE 事件队列（可为 None，此时所有方法安全空转）
        :param task_execution_id: 任务执行记录 ID（日志定位用）
        """
        self.queue = queue
        self.task_execution_id = task_execution_id

        self._token: int = 0          # 全局最新写者 token
        self._active: int = 0         # 当前有效写者 token
        self._scrubber = StreamingThinkScrubber()
        # 已推送正文的归一化指纹集合。**跨 claim 保留**：重入后需要接着上次的
        # 进度继续推，已推过的内容不能再推一遍。
        self._delivered: set[str] = set()
        # 已推送正文字符数（单调累加，跨 claim 保留）。供调用方做「兜底补推」时
        # 从该位置裁切，避免把流式已推送的部分整段重推一遍。
        self._pushed_chars: int = 0

    # ------------------------------------------------------------------
    # 写者令牌
    # ------------------------------------------------------------------

    def claim(self) -> int:
        """
        抢占写者身份，返回本次流的新 token。

        **每次进入 task_node（首轮 / 超时重试 / 审批恢复重入）都必须调用**。
        调用后，持有旧 token 的循环产出的所有帧都会被丢弃。

        :return: 本次流的写者 token
        """
        self._token += 1
        self._active = self._token
        self._scrubber.reset()
        return self._active

    def _fenced(self, token: int) -> bool:
        """判断 token 是否已被更新的写者取代（取代后其帧必须丢弃）。"""
        return token != self._active

    # ------------------------------------------------------------------
    # 派发
    # ------------------------------------------------------------------

    async def emit_content(self, text: str, token: int, *, dedup: bool = True) -> bool:
        """
        派发正文片段（经思考块过滤 + 去重）。

        :param text: 模型产出的正文 delta
        :param token: 本轮流写者 token
        :param dedup: 是否做重复推送去重（默认开；流末整段补推时建议关）
        :return: 是否实际推送成功
        """
        if self._fenced(token) or not text:
            return False
        try:
            # 过滤思考块：模型把思考直接写在正文里（<think>...</think>）时，
            # 这段必须剔除，否则用户会看到模型的内心独白。
            visible = self._scrubber.feed(text)
        except Exception:
            logger.debug("[Agent-Stream] 思考块过滤失败，按原文推送", exc_info=True)
            visible = text
        if not visible:
            return False

        fingerprint = normalize_fragment(visible)
        if dedup and fingerprint and fingerprint in self._delivered:
            from agent.observability import metrics

            metrics.incr("agent.task.stream_dedup")
            return False

        ok = await put_content(self.queue, visible)
        if ok and fingerprint:
            self._delivered.add(fingerprint)
            self._pushed_chars += len(visible)
        return ok

    async def emit_thinking(self, text: str, token: int) -> bool:
        """
        派发思考过程片段。

        注意：思考过程**不参与去重**。它是模型的推理流，重入后重新生成时内容
        会有细微差异（温度采样），做指纹去重意义不大且容易误杀。

        :param text: 思考过程 delta
        :param token: 本轮流写者 token
        :return: 是否实际推送成功
        """
        if self._fenced(token) or not text:
            return False
        try:
            return await put_thinking(self.queue, text)
        except Exception:
            logger.debug("[Agent-Stream] 思考过程派发失败（忽略）", exc_info=True)
            return False

    async def flush(self, token: int) -> bool:
        """
        流结束收尾：吐出被扣留的**正常**尾巴。

        未闭合思考块内的内容会被 `StreamingThinkScrubber.flush()` 丢弃
        （泄漏半截思考比答案不完整更糟）。

        :param token: 本轮流写者 token
        :return: 是否推送了尾巴
        """
        if self._fenced(token):
            return False
        try:
            tail = self._scrubber.flush()
        except Exception:
            logger.debug("[Agent-Stream] 思考块 flush 失败（忽略）", exc_info=True)
            tail = ""
        if not tail:
            return False
        fingerprint = normalize_fragment(tail)
        if fingerprint and fingerprint in self._delivered:
            return False
        ok = await put_content(self.queue, tail)
        if ok and fingerprint:
            self._delivered.add(fingerprint)
            self._pushed_chars += len(tail)
        return ok

    # ------------------------------------------------------------------
    # 可观测性
    # ------------------------------------------------------------------

    @property
    def delivered_count(self) -> int:
        """已推送片段数（审计/日志用）。"""
        return len(self._delivered)

    @property
    def delivered_chars(self) -> int:
        """
        已推送正文字符数（**近似**，用于兜底补推时裁切）。

        为什么不精确：思考块过滤会改变可见长度、去重会跳过重复片段，
        两者都让「推送字符数」与「模型原始输出长度」不再一一对应。
        因此调用方必须**容错**：按该位置裁切后若得到的尾巴异常
        （为空、或明显不是答案开头），退化为整段推送。

        该属性只用于「少推一点」与「多推一点」之间的权衡，不需要精确。
        """
        return self._pushed_chars
