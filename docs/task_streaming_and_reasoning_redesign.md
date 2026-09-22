# 任务分支流式输出 & LLM 思考过程 改造方案

> 参考来源：`Hermes-Agent`（`D:\hermes-agent-main`）`agent/stream_delivery.py`、`agent/chat_completion_helpers.py`、`agent/think_scrubber.py`、`agent/agent_runtime_helpers.py`
> 目标项目：`lingxi-agent`（FastAPI + LangGraph 1.2.10 + LangChain 1.3.10 + DashScope Qwen）
> 文档性质：**改造方案**（未落地，待评审后实施）

---

## 0. 结论先行

| 问题 | 现状 | 结论 |
| --- | --- | --- |
| 任务分支流式输出 | `task_node` 用 `agent.ainvoke()` 跑完，末尾一次性 `put_content(task_answer)` | **完全非流式**，用户要干等 30–120s。改造为 `astream` 是本次主线 |
| LLM 思考过程 | 全项目**零支持**，且技术栈存在**两处断点** | 需要同时打通「请求侧开启」+「响应侧提取」 |
| chat / knowledge 分支 | 已是真流式（`chain.astream` 逐 token） | 只需**追加** reasoning 通道，主链路不动 |

**Hermes 最值得搬的三块**（按价值排序）：
1. **`StreamingThinkScrubber`** —— 有状态思考块过滤器，解决 `<think>` 标签被 chunk 切断的经典 bug（第 3.3 节有完整代码）；
2. **统一派发 + 单写者栅栏** —— 所有 delta 走一个出口，重试/重入时旧流的残留 delta 被丢弃；
3. **分类事件通道** —— 正文 / 思考 / 工具进度 走不同回调，互不污染。

---

## 1. Hermes 的实现（学到了什么）

### 1.1 流式输出：三层结构

Hermes 的流式不是「节点各自推 SSE」，而是**统一派发 + 消费者层节流**：

```
LLM 流 (chat_completion_helpers.py:2889  for chunk in stream)
   │
   ├─ delta.reasoning_content / .reasoning   → _fire_reasoning_delta()  → reasoning_callback
   ├─ delta.content                          → _fire_stream_delta()     → stream_delta_callback
   └─ delta.tool_calls                       → tool_calls.feed()        → tool_gen_callback
                                                                            (stream_delivery.py:285/318/336)
```

三个关键设计：

**(a) 统一派发出口** `agent/stream_delivery.py:285`
```python
def _fire_stream_delta(self, text: str) -> None:
    if self._stream_writer_superseded():      # ① 单写者栅栏
        self._note_dropped_stream_writer("_fire_stream_delta")
        return
    text = think_scrubber.feed(text)          # ② 有状态思考块过滤
    text = scrubber.feed(text)                # ③ 记忆上下文围栏
    delivered = self._deliver_to_stream_callbacks(text)
    if delivered:
        self._record_streamed_assistant_text(text)   # ④ 累积，供去重
```

**(b) 单写者栅栏** `agent/stream_delivery.py:222`（对应 issue #65991）
```python
def _claim_stream_writer(self) -> int:
    with self._stream_writer_lock:
        self._stream_writer_token = token = self._stream_writer_token + 1
    self._stream_writer_tls.token = token
    return token

def _stream_writer_superseded(self) -> bool:
    token = getattr(getattr(self, "_stream_writer_tls", None), "token", None)
    return token is not None and token != getattr(self, "_stream_writer_token", token)
```
> **为什么需要**：重试/中断后，旧流的对象还在另一个线程上活着，它的 tail delta 会和新流的 token **交错输出**。每次新的流开始前 `claim` 一次，旧 token 的 delta 全部丢弃。
> **映射到本项目**：`task_node` 因 HITL 审批会**重入**（`_find_running_task` 复用记录），这是同一个问题的同构场景——必须加栅栏。

**(c) 节流放在消费层，不在产生层** `gateway/stream_consumer.py:706`
```python
DEFAULT_STREAMING_EDIT_INTERVAL: float = 0.8   # gateway/config.py:440
DEFAULT_STREAMING_BUFFER_THRESHOLD: int = 24

elapsed = time.monotonic() - self._last_edit_time
should_edit = bool((elapsed >= self._current_edit_interval and self._accumulated)
                   or len(self._accumulated) >= self.cfg.buffer_threshold)
```
> Agent 层每 token 直发；「攒够 0.8s 或 24 码点再刷一次」由展示层决定。产生层保持简单。

**(d) 子任务流式走独立通道**（`tools/delegate_tool_child_run.py:658`）
```python
def relay_text(self, delta: str) -> None:
    """子智能体的 token 走 progress 事件通道 (subagent.text → message.delta)，
    不复用父级的 stream_delta_callback，避免父子正文串在一起。"""
```
> Hermes 的 delegate 子体**不**直接复用父级正文回调。同理，本项目 task 子 Agent 的中间过程也应与最终答案区分。

### 1.2 思考过程：字段提取是多级降级的

`agent/agent_runtime_helpers.py:1219` 的统一提取器：

```python
def extract_reasoning(agent, assistant_message) -> Optional[str]:
    """reasoning / reasoning_content / reasoning_details (OpenRouter 统一),
    else inline thinking blocks in the content; None when absent."""
    _add(getattr(assistant_message, "reasoning", None))            # ① Anthropic / 通用
    _add(getattr(assistant_message, "reasoning_content", None))    # ② DeepSeek / Qwen / MiniMax
    for detail in getattr(assistant_message, "reasoning_details", None) or []:
        if isinstance(detail, dict):                               # ③ OpenRouter 结构化
            _add(detail.get('summary') or detail.get('thinking')
                 or detail.get('content') or detail.get('text'))
    if not parts and isinstance(content, list):
        for block in content:                                      # ④ 类型化内容块
            if isinstance(block, dict) and block.get("type") == "thinking":
                _add((block.get("thinking") or block.get("text") or "").strip())
    if not parts and isinstance(content, str) and content:
        for pattern in _INLINE_REASONING_PATTERNS:                 # ⑤ <think> 内联标签
            for block in pattern.findall(content):
                _add(block.strip())
    return "\n\n".join(parts) if parts else None
```

流式侧的取法（`agent/chat_completion_helpers.py:2924`）双字段兼容：
```python
reasoning_text = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
```

**落库策略**（`agent/session_persistence.py:42`）—— 保留但隔离：
```python
_ROW_REASONING_KEYS = ("reasoning", "reasoning_content", "reasoning_details",
                       "codex_reasoning_items", "codex_message_items")
```
> 关键取舍：**落盘保留，但不回灌进下一轮的 messages**（某些模型如 DeepSeek V4 不回传会 400）。本项目的 Qwen 无此约束，建议**不进 messages**（见 §5.3）。

### 1.3 最有价值的一块：`StreamingThinkScrubber`

`agent/think_scrubber.py` —— 为什么必须**有状态**：

> 正则 `_strip_think_blocks` 对**完整字符串**是正确的，但**逐 delta 运行时**会把单独到达的 `<think>` 开标签擦掉，导致下游状态机看不到开标签，思考内容就当作正文泄漏出去了（Hermes issue #17924，MiniMax-M2.7 把 `<think>` 和内容分成两个 delta 发送）。

核心状态机（`feed` / `flush` / `_hold_partial`）：

```python
THINK_TAG_NAMES = ("think", "thinking", "reasoning", "thought", "REASONING_SCRATCHPAD")

def feed(self, text: str) -> str:
    """喂一个 delta，返回可见部分（"" 表示该帧全部是思考或被扣留）"""
    if not text: return ""
    buf = self._buf + text
    self._buf = ""
    out: list[str] = []
    while buf:
        if self._in_block:                                   # 块内：丢弃，等闭标签
            close_idx, close_len = self._find_first_tag(buf, self._CLOSE_TAGS)
            if close_idx == -1:
                self._hold_partial(buf, self._CLOSE_TAGS)    # 只扣留可能的半个闭标签
                break
            buf = buf[close_idx + close_len:]
            self._in_block = False
            continue
        pair = self._find_earliest_closed_pair(buf)          # 优先级1：闭合的 <tag>X</tag>
        open_idx, open_len = self._find_open_at_boundary(buf, out)  # 优先级2：块边界的开标签
        if pair is not None and (open_idx == -1 or pair[0] <= open_idx):
            self._emit(out, buf[:pair[0]]); buf = buf[pair[1]:]; continue
        if open_idx != -1:
            self._emit(out, buf[:open_idx]); self._in_block = True
            buf = buf[open_idx + open_len:]; continue
        self._emit(out, self._hold_partial(buf, self._ALL_TAGS))   # 扣留尾部半个标签
        break
    return "".join(out)

def flush(self) -> str:
    """流尾：未闭合块内的内容丢弃（泄漏半截思考 比 答案截断更糟），
    否则原样吐出扣留的尾巴。"""
    tail = "" if self._in_block else self._buf
    self._buf = ""; self._in_block = False
    self._last_emitted_ended_newline = True
    return self._strip_orphan_close_tags(tail) if tail else ""
```

**两个易被忽略的细节**：
- `_is_block_boundary`：开标签只在**块边界**（流开始 / 换行后 / 当前行仅空白）才生效，否则正文里"提到" `<think>` 会被误杀；
- `flush()` **必须重置** `_last_emitted_ended_newline = True`，否则轮内重试后新流的开头 `<think>` 会被判定为行中间而不生效。

---

## 2. 当前项目现状盘点

### 2.1 现有流式链路

```
POST /api/agent/chat  (api/routes/agent.py:149)
   ├─ create_sse_queue()              有界队列 maxsize=1000   (streaming.py:57)
   ├─ run_graph() 后台任务            graph.ainvoke(inputs, config)
   └─ event_generator()              消费 queue → yield SSE 帧
        └─ StreamingResponse(media_type="text/event-stream")
```

节点通过 `get_sse_queue(config)` 拿队列，`put_content` 逐帧推（**已具备有界队列 + 超时丢帧 + `K_SSE_DROPPED` 指标**，背压这块做得不错，无需重做）。

### 2.2 各分支流式能力差距

| 分支 | 文件 | 流式 | 机制 |
| --- | --- | --- | --- |
| `chat` | `agent/nodes/chat_node.py:110` | ✅ | `chain.astream` 逐 token `put_content` |
| `knowledge` | `agent/nodes/knowledge_node.py:105` | ✅ | `chain.astream`（缓存命中时按 20 字切片模拟） |
| `merge` | `graph_builder.py:320` | ❌ | `ainvoke` 一次性 |
| **`task_agent`** | **`agent/nodes/task_node.py:127`** | **❌** | **`agent.ainvoke()` 跑完 → 末尾 `put_content(task_answer)` 一次推全文（:143）** |

任务节点的问题代码：
```python
result = await agent.ainvoke({"messages": [HumanMessage(content=user_input)]}, config=invoke_config)
task_answer = _extract_answer(result)
# 4. 推送最终回答到 SSE（任务结果非流式，一次性推送）   ← 注释已自认
if queue is not None and task_answer:
    await put_content(queue, task_answer)
```
**后果**：任务分支（多轮工具调用，耗时最长）恰恰是**唯一没有流式**的分支。用户看到的是长时间空白，然后整段砸下来。

### 2.3 思考过程：技术栈存在两处断点（已实测确认）

**断点 1 — 请求侧没开启。** DashScope Qwen3 系列需 `enable_thinking=True`，当前 11 处 `ChatOpenAI(...)` 均未传。

**断点 2 — 响应侧静默丢弃。** 这是关键发现：

`langchain_openai/chat_models/base.py:712-719` 类注释明确声明：
> "Non-standard response fields added by third-party providers (e.g., `reasoning_content`) are **not** extracted. Use a provider-specific subclass for full provider support."

而 `_convert_delta_to_message_chunk`（`base.py:484`）只读取 `content` / `function_call` / `tool_calls`，`reasoning_content` 被丢掉。

**但好消息**：OpenAI SDK 的 `BaseModel` 是 `extra="allow"`（`openai/_models.py:128`），实测 `reasoning_content` **仍在** `chunk["choices"][0]["delta"]` 里：

```
实测: ChoiceDelta.model_validate({'content':'答案是42','reasoning_content':'让我算一下'}).model_dump()
  → {'content':..., 'role':..., 'tool_calls':..., 'reasoning_content': '我在思考'}   ✅ 未被丢弃
```

**所以只需重写一个方法就能打通**。已跑通的原型验证：

```python
class ThinkingChatOpenAI(ChatOpenAI):
    def _convert_chunk_to_generation_chunk(self, chunk, default_chunk_class, base_generation_info):
        g = super()._convert_chunk_to_generation_chunk(chunk, default_chunk_class, base_generation_info)
        if g is None: return g
        delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
        if r := delta.get("reasoning_content"):
            g.message.additional_kwargs["reasoning_content"] = r
        return g

# 实测输出: content='答案是42'  reasoning='让我算一下'   ✅
```

---

## 3. 改造方案总体设计

### 3.1 分层架构

```
┌──────────────────────────────────────────────────────────────┐
│ L4 前端：思考折叠面板 / 工具进度条（按 event 字段渲染）            │
└───────────────────────────┬──────────────────────────────────┘
┌───────────────────────────┴──────────────────────────────────┐
│ L3 SSE 协议层  agent/streaming.py（扩展事件帧，向后兼容）        │
│   content / reasoning / tool / phase / status / ping / done    │
└───────────────────────────┬──────────────────────────────────┘
┌───────────────────────────┴──────────────────────────────────┐
│ L2 统一派发器  agent/stream_emitter.py（新增）                  │
│   StreamEmitter: 单写者栅栏 + ThinkScrubber + 去重 + 背压        │
└───────────────────────────┬──────────────────────────────────┘
┌───────────────────────────┴──────────────────────────────────┐
│ L1 LLM 层  core/llm.py（新增）                                 │
│   ThinkingChatOpenAI: enable_thinking + reasoning_content 提取 │
│   get_chat_model() 工厂（收敛 11 处 ChatOpenAI 实例化）           │
└───────────────────────────┬──────────────────────────────────┘
┌───────────────────────────┴──────────────────────────────────┐
│ L0 节点层  chat / knowledge / task_agent                       │
│   task: ainvoke → astream(["messages","updates"])              │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 SSE 协议扩展（向后兼容）

现有 `event` 取值全部保留，只**新增**：

```jsonc
// 思考过程 delta（新增）
data: {"event": "reasoning", "content": "让我先查一下用户表结构..."}

// 阶段切换：前端据此折叠/展开思考面板（新增）
data: {"event": "phase", "phase": "thinking"}    // thinking | answering

// 工具进度（新增）
data: {"event": "tool", "tool": "query_data", "status": "start", "args_summary": "{\"table\":\"user\"}"}
data: {"event": "tool", "tool": "query_data", "status": "end", "ok": true, "rows": 3}
```

> 前端未适配时：`reasoning` / `phase` / `tool` 是**未知 event**，前端忽略即可，`content` 行为完全不变 → **零破坏升级**。

---

## 4. 详细改造步骤

### 阶段 A：打通思考过程（低风险，可独立上线）

#### A1. 新增 `core/llm.py` —— 推理能力 LLM 工厂

```python
"""统一 LLM 构造入口（支持思考过程提取）。"""
from langchain_openai import ChatOpenAI
from core.config import settings

# 支持 thinking 的模型前缀白名单（qwen-turbo 不支持，勿开）
THINKING_MODEL_PREFIXES = ("qwen3", "qwen-plus", "qwen-max", "deepseek-r1", "glm-4.5")

class ThinkingChatOpenAI(ChatOpenAI):
    """ChatOpenAI 子类：透传 DashScope/Qwen 的 reasoning_content。

    langchain_openai.BaseChatOpenAI 明确不提取第三方非标准字段
    (base.py:712)，而 openai SDK 的 BaseModel 是 extra="allow"
    (_models.py:128)，故 reasoning_content 实际仍存在于 delta 中，
    仅需在转换时取出。
    """
    def _convert_chunk_to_generation_chunk(self, chunk, default_chunk_class, base_generation_info):
        gen = super()._convert_chunk_to_generation_chunk(chunk, default_chunk_class, base_generation_info)
        if gen is None:
            return gen
        try:
            delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
            for key in ("reasoning_content", "reasoning"):
                if delta.get(key):
                    gen.message.additional_kwargs[key] = delta[key]
                    break
        except Exception:
            pass        # 提取失败绝不影响正文
        return gen

def get_chat_model(model: str, *, temperature: float = 0.7, streaming: bool = True, **kw):
    """全项目统一 LLM 构造入口。"""
    enable = settings.agent_reasoning_enabled and _supports_thinking(model)
    return ThinkingChatOpenAI(
        model=model,
        openai_api_key=settings.api_key,
        openai_api_base=settings.llm_base_url,
        temperature=temperature,
        streaming=streaming,
        timeout=kw.pop("timeout", settings.agent_llm_timeout_seconds),
        extra_body={"enable_thinking": True} if enable else None,
        **kw,
    )
```
> **注意**：`extra_body=None` 时不要传空 dict，否则部分网关会报参数错误。

#### A2. `core/config.py` 新增配置

```python
# ---- 思考过程（reasoning）----
agent_reasoning_enabled: bool = True      # 总开关
agent_reasoning_models: str = "qwen3,qwen-plus,qwen-max"   # 开启白名单（前缀匹配）
agent_reasoning_persist: bool = False     # 思考内容是否落审计表（默认否）
agent_stream_throttle_ms: int = 0         # 正文节流（0=不节流，每 token 直发）
llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"   # 收敛硬编码
```

#### A3. `agent/streaming.py` 扩展事件帧

```python
async def put_reasoning(queue, content: str) -> None:
    """推送思考过程 delta"""
    if queue is not None and content:
        await _safe_put(queue, build_sse_frame("reasoning", content=content), "reasoning")

async def put_tool(queue, tool: str, status: str, **extra) -> None:
    """推送工具进度"""
    if queue is not None:
        await _safe_put(queue, build_sse_frame("tool", tool=tool, status=status, **extra), f"tool:{status}")

async def put_phase(queue, phase: str) -> None:
    """推送阶段切换"""
    if queue is not None:
        await _safe_put(queue, build_sse_frame("phase", phase=phase), f"phase:{phase}")
```

#### A4. chat / knowledge 节点接 reasoning

两处改动对称，以 `chat_node.py:110` 为例：

```python
async for chunk in chain.astream({"input": user_input}):
    # 思考过程（reasoning）
    r = (getattr(chunk, "additional_kwargs", None) or {}).get("reasoning_content")
    if r and settings.agent_reasoning_enabled:
        await put_reasoning(queue, r)
        reasoning_buf.append(r)
    # 正文
    content = chunk.content if hasattr(chunk, "content") else str(chunk)
    if content:
        full_response += content
        await put_content(queue, content)
```

**验收**：`qwen-plus` 提问，SSE 应能看到 `reasoning` 帧先到、`content` 帧后到。

---

### 阶段 B：任务分支真流式（核心改造）

#### B1. 新增 `agent/stream_emitter.py` —— 统一派发器

移植 Hermes 的三件套（scrubber + 单写者栅栏 + 去重），适配 asyncio：

```python
class StreamEmitter:
    """任务分支流式统一派发器。

    - 单写者栅栏：审批重入时旧流的残留 delta 被丢弃（Hermes #65991 同构问题）
    - ThinkScrubber：有状态思考块过滤（Hermes think_scrubber.py 移植）
    - 去重：重入后已推送的正文不重复下发（Hermes _delivered_interim_texts 思路）
    """
    def __init__(self, queue, *, turn_key: str = ""):
        self.queue = queue
        self._token = 0
        self._active = 0
        self._scrubber = StreamingThinkScrubber()
        self._delivered: set[str] = set()      # 已推送正文（归一化后）
        self._turn_key = turn_key

    def claim(self) -> int:
        """新的一轮流开始前调用：claim 新 token，旧 token 的 delta 全部丢弃。"""
        self._token += 1
        self._active = self._token
        self._scrubber.reset()
        return self._active

    def _fenced(self, token: int) -> bool:
        return token != self._active

    async def emit_content(self, text: str, token: int, *, dedup: bool = True) -> None:
        if self._fenced(token) or not text:
            return
        visible = self._scrubber.feed(text)
        if not visible:
            return
        if dedup and self._seen(visible):
            return
        await put_content(self.queue, visible)
        self._delivered.add(_norm(visible))

    async def emit_reasoning(self, text: str, token: int) -> None:
        if self._fenced(token) or not text:
            return
        await put_reasoning(self.queue, text)

    async def flush(self, token: int) -> None:
        """流尾：吐出扣留的正常尾巴（未闭合思考块内的一律丢弃）。"""
        if self._fenced(token):
            return
        tail = self._scrubber.flush()
        if tail:
            await put_content(self.queue, tail)
```

`StreamingThinkScrubber` 直接移植 `D:\hermes-agent-main\agent\think_scrubber.py`（176 行，MIT 许可），保留 `THINK_TAG_NAMES` 与全部边界逻辑。

#### B2. `task_node.py` 改造：`ainvoke` → `astream`

```python
emitter = StreamEmitter(queue, turn_key=task_execution_id)
token = emitter.claim()          # ← 每一次进入（含审批恢复重入）都 claim

async def _consume() -> None:
    async for mode, payload in agent.astream(
        {"messages": [HumanMessage(content=user_input)]},
        config=invoke_config,
        stream_mode=["messages", "updates"],
    ):
        if mode == "messages":
            chunk, meta = payload
            node = (meta or {}).get("langgraph_node")
            if node != "model":          # ← 关键：只取 model 节点，屏蔽 tools 节点回吐
                continue
            r = (getattr(chunk, "additional_kwargs", None) or {}).get("reasoning_content")
            if r:
                await emitter.emit_reasoning(r, token)
            if getattr(chunk, "content", None):
                await emitter.emit_content(chunk.content, token)
        elif mode == "updates":
            # 工具开始/结束 → 进度事件
            for node_name, update in (payload or {}).items():
                if node_name == "tools":
                    await _emit_tool_progress(update, emitter, token)
    await emitter.flush(token)

if approval_mode:
    await _consume()
else:
    await asyncio.wait_for(_consume(), timeout=settings.agent_task_timeout_seconds)
```

**三个必须注意的点**：

1. **只取 `langgraph_node == "model"`** —— `stream_mode="messages"` 会把 **ToolMessage（工具返回原文）** 也吐出来，不过滤会直接把数据库原始结果泄漏给用户。（`create_agent` 的节点名已确认：`factory.py:1476 add_node("model")` / `:1480 add_node("tools")`）
2. **`GraphInterrupt` 必须在 `astream` 中继续向上传播** —— 审批中断语义不能变，`except GraphInterrupt: raise` 保持原样。
3. **`ainvoke` 兜底降级** —— `astream` 抛非中断异常时，回落到原 `ainvoke` 路径，保证"宁可慢，不可断"。

#### B3. 审批重入的去重与栅栏

`task_node` 被 HITL 重入时（`:95 _find_running_task`）必须：

```python
# 重入：claim 新 token → 旧流栅栏失效；scrubber 重置；_delivered 保留用于去重
token = emitter.claim()
```

> `_delivered` **不能**随 claim 清空——重入后要接着上次的进度推，已推过的正文不能再推一遍。建议把 `_delivered` 摘要挂到 `state` / `task_execution` 记录上，跨进程恢复也能去重（一期可只在内存中，重入同进程即可覆盖 95% 场景）。

#### B4. 工具进度事件

`stream_mode="updates"` 的 `tools` 节点 payload 里解析 ToolMessage：

```python
async def _emit_tool_progress(update, emitter, token):
    for msg in (update.get("messages") or []):
        if getattr(msg, "type", "") != "tool":
            continue
        await put_tool(emitter.queue, name=getattr(msg, "name", ""), status="end",
                       ok="失败" not in str(msg.content)[:50])
```
> 工具**开始**事件一期可从 `stream_mode="messages"` 的 `tool_call_chunks` 里取 `name` 非空的第一帧触发（与 Hermes `tool_gen_callback` 同思路）。

#### B5. 观测指标（`agent/observability.py`）

```python
K_REASONING_DELTA = "agent.reasoning.deltas"
K_TASK_STREAM_CHUNK = "agent.task.stream_chunks"
K_TASK_STREAM_FENCED = "agent.task.stream_fenced"     # 被栅栏丢弃的 delta 数
K_TASK_STREAM_DEDUP = "agent.task.stream_dedup"
```

---

### 阶段 C：完善（可选）

| 项 | 内容 | 优先级 |
| --- | --- | --- |
| C1 | 节流：`agent_stream_throttle_ms`，仿 gateway 的「0.8s 或 24 码点」批量 flush | 中 |
| C2 | 思考过程落 `task_execution` 新列 `reasoning`（默认关） | 低 |
| C3 | 工具**内部**细粒度进度（`stream_mode="custom"` + `get_stream_writer()`） | 低 |
| C4 | 冒烟脚本 `scripts/smoke_task_streaming.py`：mock LLM 依次吐 reasoning/content/tool_calls，断言事件序列 | **高**（随 B 一起做） |

---

## 5. 风险与取舍

### 5.1 技术风险

| 风险 | 影响 | 处置 |
| --- | --- | --- |
| `enable_thinking` 与 `stream_options.include_usage` 冲突 | DashScope 可能报错 | A1 灰度：先只加提取（不改请求），确认字段能拿到后再开启 |
| 非 Qwen 模型无 `reasoning_content` | 无 | 白名单前缀匹配，未命中不传 `extra_body` |
| `astream` 漏掉 `GraphInterrupt` | **审批流程失效** | B2 保持 `except GraphInterrupt: raise`；冒烟脚本必须覆盖 |
| ToolMessage 泄漏 | **数据库原文直出给用户** | B2 强制 `langgraph_node == "model"` 过滤 |
| 重入后正文重复 | 用户看到两份答案 | B1 `StreamEmitter._delivered` 去重 + B3 栅栏 |
| 思考块跨 chunk 切断 | 思考内容当正文泄漏 | B1 `StreamingThinkScrubber`（Hermes 已踩过坑 #17924） |

### 5.2 设计取舍

**(1) 节流放哪？** → 一期**不做**。Agent 层每 token 直发，与现有 `chat`/`knowledge` 行为一致；真有性能问题再在 emitter 层加（配置已预留 `agent_stream_throttle_ms`）。

**(2) 思考过程是否进 messages？** → **不进**。
- Qwen 不像 DeepSeek V4 那样要求回传 thinking 块，回灌只会白白吃掉上下文窗口；
- 需要留存则落 `task_execution` 新列（默认关闭 `agent_reasoning_persist=False`）。
> 若将来接入 DeepSeek V4 类模型，需按 Hermes `message_sanitization.py:449 apply_reasoning_content_policy` 的「单 owner 决策」模式单独处理回传策略。

**(3) 子 Agent 正文 vs 最终答案？** → 采纳 Hermes 的做法：**区分通道**。任务分支的中间 token 仍走 `content`（用户要看过程），但工具进度走独立 `tool` 事件；最终 `task_answer` 落库不变。

**(4) 是否引入 Hermes 的 mixin 组织方式？** → **不引入**。本项目是 LangGraph 节点式，不是 Hermes 的 facade+mixin 单体 Agent，强行照搬会增加认知负担。只搬**算法**（scrubber）与**模式**（统一派发 + 栅栏 + 去重），不搬结构。

---

## 6. 验收标准

| # | 场景 | 期望 |
| --- | --- | --- |
| 1 | `qwen-plus` 普通提问 | SSE 出现 `reasoning` 帧（思考）→ `content` 帧（正文），顺序正确 |
| 2 | `qwen-turbo` 提问（无 thinking） | 无 `reasoning` 帧，正文正常，无报错 |
| 3 | 任务分支多轮工具调用 | 工具开始/结束时推 `tool` 事件，正文逐 token 到达，**首字延迟显著下降** |
| 4 | 任务触发写操作审批 | `approval_required` 正常；审批通过后恢复，**正文不重复**、**旧流残留不串入** |
| 5 | 模型返回 `<think>...<think>` 内联标签 | 思考内容不出现在 `content` 帧 |
| 6 | `<think>` 被切成两个 chunk | 不泄漏（scrubber 生效） |
| 7 | 流式中断/超时 | 保留已推送内容，**不写 error**（沿用现有约定） |
| 8 | 前端未适配新事件 | 忽略 `reasoning`/`tool`/`phase`，行为与改造前一致 |

---

## 7. 涉及文件清单

| 文件 | 动作 | 说明 |
| --- | --- | --- |
| `core/llm.py` | **新增** | `ThinkingChatOpenAI` + `get_chat_model` 工厂 |
| `agent/stream_emitter.py` | **新增** | 统一派发器 + 单写者栅栏 + 去重 |
| `agent/think_scrubber.py` | **新增** | 移植 Hermes（MIT），保留版权头 |
| `agent/streaming.py` | 修改 | 新增 `put_reasoning` / `put_tool` / `put_phase` |
| `core/config.py` | 修改 | 新增 reasoning 相关 4 个配置项 + `llm_base_url` |
| `agent/nodes/task_node.py` | **重构** | `ainvoke` → `astream`，接入 emitter |
| `agent/nodes/chat_node.py` | 修改 | 追加 reasoning 下发 |
| `agent/nodes/knowledge_node.py` | 修改 | 追加 reasoning 下发 |
| `agent/graph_builder.py` | 修改 | merge 的 LLM 改用 `get_chat_model`（可选） |
| `agent/observability.py` | 修改 | 新增 4 个指标常量 |
| `models/task_model.py` | 修改（可选） | 新增 `reasoning` 列 |
| `scripts/smoke_task_streaming.py` | **新增** | 流式事件序列冒烟 |

---

## 附：Hermes 源码索引（供实施时对照）

| 主题 | 路径 | 行号 |
| --- | --- | --- |
| 流式主循环 | `agent/chat_completion_helpers.py` | 2889 |
| reasoning delta 提取 | `agent/chat_completion_helpers.py` | 2924 |
| 统一提取器 | `agent/agent_runtime_helpers.py` | 1219 |
| 统一派发 | `agent/stream_delivery.py` | 285 / 318 / 336 |
| 单写者栅栏 | `agent/stream_delivery.py` | 222–251 |
| 有状态思考过滤器 | `agent/think_scrubber.py` | 全文 176 行 |
| 流式/非流式决策 | `agent/turn_api_call.py` | 47–62 |
| 中断控制 | `agent/interrupt_control.py` | 109–207 |
| 消费层节流 | `gateway/stream_consumer.py` | 706–724 |
| 子任务流式 relay | `tools/delegate_tool_child_run.py` | 658 |
| 推理落库键 | `agent/session_persistence.py` | 42 |
