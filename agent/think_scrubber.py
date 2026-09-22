"""
流式思考块过滤器（有状态）

**为什么必须是有状态的**：正则替换 `_strip_think_blocks` 对**完整字符串**是正确的，
但**逐 delta 运行时**有致命缺陷——很多模型（实测 MiniMax 系、部分 Qwen 微调档）会把
`<think>` 开标签与内容**分成两个 delta** 发送：

    delta1 = "<think>"
    delta2 = "让我查一下用户的订单状态..."

逐帧跑 `re.sub(r"<think>.*?</think>", "", delta1)` 时，delta1 里没有闭标签，
正则匹配不上 → `<think>` **原样返回**（或按"未闭合即删除"策略被删掉）；
无论哪种，下游状态机都**看不到开标签**，于是 delta2 的思考内容被当作**正文**推给用户，
用户界面上直接出现模型的内心独白。

**解法**：把「可能被切断的标签前缀」**扣留**在缓冲区里，等下一个 delta 拼起来再判定。
这正是本类的全部职责。

移植自 Hermes-Agent `agent/think_scrubber.py`（MIT 许可），并做如下适配：
- 保留原实现的全部边界规则（块边界判定、孤立闭标签、flush 重置标志）；
- 命名与注释按本项目风格调整，接口收敛为 `feed()` / `flush()` / `reset()`。

**使用约定**：
- 每个「流」开始前调用 `reset()`（多轮/重试/审批重入都必须 reset）；
- 逐 delta 调 `feed()`，把返回值推给前端（返回空串表示该帧无需推送）；
- 流结束调 `flush()`，把扣留的正常尾巴吐出（未闭合思考块内的内容**丢弃**）。
"""
from __future__ import annotations

import re
from typing import Tuple

__all__ = ["StreamingThinkScrubber", "THINK_TAG_NAMES", "THINK_OPEN_TAGS", "THINK_CLOSE_TAGS"]

# 模型思考标签的统一清单。所有需要「隐藏思考过程」的地方都绑定这一份清单，
# 在此新增一个标签名即可全覆盖。消费方按大小写不敏感匹配，故字面量一律小写。
THINK_TAG_NAMES: Tuple[str, ...] = (
    "think",
    "thinking",
    "reasoning",
    "thought",
    "REASONING_SCRATCHPAD",
)
THINK_OPEN_TAGS: Tuple[str, ...] = tuple(f"<{name.lower()}>" for name in THINK_TAG_NAMES)
THINK_CLOSE_TAGS: Tuple[str, ...] = tuple(f"</{name.lower()}>" for name in THINK_TAG_NAMES)


class StreamingThinkScrubber:
    """
    流式思考块过滤器（有状态）。

    内部状态：
    - `_in_block`：当前处于未闭合的思考块内（块内文本一律丢弃）；
    - `_buf`：扣留的「可能是标签前缀」的尾巴，等下一帧拼接；
    - `_last_emitted_ended_newline`：上一次输出是否以换行结尾（或尚未输出任何内容）。
      用于判定「缓冲区位置 0 的开标签」是否处在**块边界**。
    """

    # 用字面量做字符串操作而非每帧正则 —— feed 在流式热路径上
    _OPEN_TAGS: Tuple[str, ...] = THINK_OPEN_TAGS
    _CLOSE_TAGS: Tuple[str, ...] = THINK_CLOSE_TAGS
    _ALL_TAGS: Tuple[str, ...] = _OPEN_TAGS + _CLOSE_TAGS
    _MAX_TAG_LEN: int = max(len(tag) for tag in _ALL_TAGS)
    # 孤立闭标签（无对应开标签）连同其后空白一并清除
    _ORPHAN_CLOSE_RE = re.compile(
        "(?:" + "|".join(re.escape(t) for t in _CLOSE_TAGS) + r")[ \t\n\r]*",
        re.IGNORECASE,
    )

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """重置全部状态。**每一轮流开始时必须调用。**"""
        self._in_block: bool = False
        self._buf: str = ""
        self._last_emitted_ended_newline: bool = True

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------

    def feed(self, text: str) -> str:
        """
        喂入一个 delta，返回其中**可见**的部分。

        :param text: 本帧文本
        :return: 过滤后的可见文本（"" 表示整帧都是思考内容或被扣留）
        """
        if not text:
            return ""
        buf = self._buf + text
        self._buf = ""
        out: list[str] = []

        while buf:
            if self._in_block:
                # 块内：丢弃一切，直到遇见闭标签
                close_idx, close_len = self._find_first_tag(buf, self._CLOSE_TAGS)
                if close_idx == -1:
                    # 闭标签还没来：扣留「可能是半个闭标签」的尾巴，其余丢弃
                    self._hold_partial(buf, self._CLOSE_TAGS)
                    break
                buf = buf[close_idx + close_len:]
                self._in_block = False
                continue

            # 优先级 1：任意位置的**闭合** `<tag>X</tag>` 对（即使是行内的，也几乎
            #           必然是泄漏的思考内容）；
            # 优先级 2：处于**块边界**的**未闭合**开标签。
            #           加块边界约束是为了避免「正文里提到 <think>」被误杀。
            # 两者取位置更靠前者。
            pair = self._find_earliest_closed_pair(buf)
            open_idx, open_len = self._find_open_at_boundary(buf, out)
            if pair is not None and (open_idx == -1 or pair[0] <= open_idx):
                self._emit(out, buf[: pair[0]])
                buf = buf[pair[1]:]
                continue
            if open_idx != -1:
                self._emit(out, buf[:open_idx])
                self._in_block = True
                buf = buf[open_idx + open_len:]
                continue

            # 本帧没有任何可判定的标签：扣留尾部「可能是标签前缀」的片段，
            # 其余原样输出（保证正常文本零延迟）
            self._emit(out, self._hold_partial(buf, self._ALL_TAGS))
            break

        return "".join(out)

    def flush(self) -> str:
        """
        流结束时的收尾。

        规则：**未闭合思考块内的内容一律丢弃**——泄漏半截思考比答案被截断更糟
        （用户会看到模型的内心独白，而答案是确定不完整的）。

        必然重置 `_last_emitted_ended_newline`：轮内重试时会在不调 `reset()` 的
        情况下 flush 后再开新流，若该标志残留为 False，新流开头的 `<think>`
        会被误判成「行中间」而不生效（Hermes #17924 踩过的坑）。
        """
        tail = "" if self._in_block else self._buf
        self._buf = ""
        self._in_block = False
        self._last_emitted_ended_newline = True
        return self._strip_orphan_close_tags(tail) if tail else ""

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _emit(self, out: list[str], text: str) -> None:
        """把可见文本追加到 out（剥离孤立闭标签），并维护换行标志。"""
        text = self._strip_orphan_close_tags(text)
        if text:
            out.append(text)
            self._last_emitted_ended_newline = text.endswith("\n")

    def _hold_partial(self, buf: str, tags: Tuple[str, ...]) -> str:
        """把 buf 尾部「可能是标签前缀」的片段移入 `_buf`，返回剩余部分。"""
        held = self._max_partial_suffix(buf, tags)
        self._buf = buf[-held:] if held else ""
        return buf[:-held] if held else buf

    @staticmethod
    def _find_first_tag(buf: str, tags: Tuple[str, ...]) -> Tuple[int, int]:
        """返回 tags 中最早出现的 (索引, 标签长度)；无命中返回 (-1, 0)。"""
        buf_lower = buf.lower()
        hits = [(idx, len(tag)) for tag in tags if (idx := buf_lower.find(tag)) != -1]
        return min(hits) if hits else (-1, 0)

    def _find_earliest_closed_pair(self, buf: str):
        """返回最早的 `<tag>...</tag>` 对的 (起始索引, 结束索引)；无则 None。"""
        buf_lower = buf.lower()
        pairs = []
        for open_tag, close_tag in zip(self._OPEN_TAGS, self._CLOSE_TAGS):
            open_idx = buf_lower.find(open_tag)
            close_idx = (
                buf_lower.find(close_tag, open_idx + len(open_tag)) if open_idx != -1 else -1
            )
            if close_idx != -1:
                pairs.append((open_idx, close_idx + len(close_tag)))
        return min(pairs) if pairs else None

    def _find_open_at_boundary(self, buf: str, already_emitted: list[str]) -> Tuple[int, int]:
        """返回最早的「处于块边界」的开标签 (索引, 长度)；无则 (-1, 0)。"""
        buf_lower = buf.lower()
        hits = []
        for tag in self._OPEN_TAGS:
            idx = buf_lower.find(tag)
            while idx != -1 and not self._is_block_boundary(buf, idx, already_emitted):
                idx = buf_lower.find(tag, idx + 1)
            if idx != -1:
                hits.append((idx, len(tag)))
        return min(hits) if hits else (-1, 0)

    def _is_block_boundary(self, buf: str, idx: int, already_emitted: list[str]) -> bool:
        """
        判断 *idx* 是否处于块边界。

        成立条件（任一）：
        - 位置 0，且上一次输出以换行结尾（或尚无输出）；
        - 当前行 idx 之前的部分全是空白（此时若 buf 内没有换行，
          则要求上一次输出也以换行结尾）。
        """
        prior_newline = (
            already_emitted[-1].endswith("\n") if already_emitted else self._last_emitted_ended_newline
        )
        if idx == 0:
            return prior_newline
        preceding = buf[:idx]
        last_nl = preceding.rfind("\n")
        return (prior_newline if last_nl == -1 else True) and preceding[last_nl + 1:].strip() == ""

    @classmethod
    def _max_partial_suffix(cls, buf: str, tags: Tuple[str, ...]) -> int:
        """返回 buf 尾部「是某个标签的严格前缀」的最长长度（完整标签由别处处理）。"""
        buf_lower = buf.lower()
        for i in range(min(len(buf_lower), cls._MAX_TAG_LEN - 1), 0, -1):
            suffix = buf_lower[-i:]
            if any(len(tag) > i and tag.startswith(suffix) for tag in tags):
                return i
        return 0

    @classmethod
    def _strip_orphan_close_tags(cls, text: str) -> str:
        """清除无对应开标签的闭标签（必为噪音）及其后空白。"""
        return cls._ORPHAN_CLOSE_RE.sub("", text) if "</" in text else text
