# 多意图串行化 + 上游上下文注入 改造方案

> 对应主设计文档 `lingxi_agent_langgraph_redesign.md` 的后续项（该文档已至 §26）。
> 本方案为独立文档，内部使用 §1~§11 编号。
> 结论来源：2026-09-22 / 09-23 / 09-29 的代码核对与线上 trace 分析。

---

## §1 背景与目标

### §1.1 现状

主图 `intent_router → 条件路由（knowledge / chat / task_agent）→ merge → finalize`。
当 `intents == {"task", "knowledge_base"}` 时，`route_by_intents` 返回**节点名列表**
`[knowledge, task_agent]`，LangGraph 在同一 superstep **并行**执行两个分支，
再由 `merge_node` 用一次 **LLM 调用**把两段结果合并润色。

任务分支内部（`agent/nodes/task_node.py`）是**嵌套子 Agent**，绑定 `search_knowledge`
工具，提示词第 3 条要求模型「若仍缺信息，**先调用 search_knowledge** 检索知识库」。

### §1.2 由此产生的浪费

一次 `{knowledge_base, task}` 请求的实际开销：

| 环节 | 调用点 | 说明 |
|---|---|---|
| query 向量（整句原文） | 入口 `_prefetch_query_vectors` | 全链路复用 |
| 检索 + 生成 | `knowledge_node` | 产出 `rag_answer`，流式推送 |
| query 向量（子问题） | `KnowledgeToolExecutor.search_knowledge` | **模型改写后的 query**，与上游不同文本 |
| 检索 | 同上 | 走 `KnowledgeService.retrieve`，受检索缓存覆盖 |
| LLM：决定调工具 | 内层 agent loop | 一次完整的模型往返 |
| LLM：回灌后再决策 | 内层 agent loop | 又一次完整往返 |
| LLM：合并润色 | `merge_node` | 输入含 rag 全文 + task 全文，**链路里 token 最大的一次** |

三层缓存（§18.5 向量预取 / §18.13 检索缓存 / 答案缓存）已经覆盖了**检索链路**，
但**碰不到 LLM**，也解决不了「跨分支上下文不传递」。

### §1.3 目标

1. **多意图串行化**：知识分支先跑，产出落 state，任务分支复用；
2. **merge 拼接**：默认不再调用 LLM 合并；
3. **上游 context 注入**：任务分支拿到已检索的知识依据，不再重复检索；
4. 全程保留独立开关，任一项可单独回滚。

---

## §2 已核对的关键代码事实

改造前必须知道的事实（全部经代码核对，非推断）：

| # | 事实 | 位置 |
|---|---|---|
| F1 | `merge` 的 LLM 路径**已存在降级拼接**：`f"{rag}\n\n{task}"` | `graph_builder.py:430` |
| F2 | `agent_merge_timeout_seconds=60` + `agent_merge_llm_retries=1` → **最坏 120s** | `core/config.py:145-146` |
| F3 | 对比：`agent_task_timeout_seconds=120`（干完整件活的预算） | `core/config.py:79` |
| F4 | `_already_pushed_branches` **只有定义、全仓无调用点**（死代码） | `graph_builder.py:508` |
| F5 | `pushed_contents` **只有** `task_node` 写入，`knowledge_node` / `chat_node` 不写 | `task_node.py:1067` |
| F6 | `create_agent` 的 `system_prompt` 转为 `SystemMessage` 后作为**闭包变量**，每次 model 节点执行时 `messages = [request.system_message, *messages]` **临时前置**，**不写入 state** | `langchain/agents/factory.py:951, 1390-1391, 1412` |
| F7 | 内层 Agent 有**独立 thread + checkpointer**（`_inner_thread_id`），审批恢复走 `configurable.approval_resume` 透传 | `task_node.py:764-773` |
| F8 | `RETRIEVE_TOP_K=4`（知识节点）== `DEFAULT_K=4`（工具）→ top_k 已对齐 | `knowledge_node.py:24` / `agent_tool.py:919` |
| F9 | 并行审批中断会**取消**仍在执行的知识分支 → 需 `_recover_knowledge_branch` 补跑 | `graph_builder.py:384-385, 441` |
| F10 | `format_context_public` **无总长度上限**；工具的 `_build_truncated_context` 截断到 800/4000 | `agent_tool.py:921-923` |

**F6 是本次改造能成立的前提**：正因为 `system_prompt` 不进 state，
动态注入上下文才**不会**污染内层 checkpoint、**不会**在审批恢复轮产生重复 SystemMessage。

---

## §3 改造一：多意图串行化

### §3.1 路由改造

`agent/graph_builder.py::route_by_intents`

```python
if intents == {"task", "knowledge_base"}:
    if not _is_node_registered(NODE_TASK_AGENT):
        return NODE_KNOWLEDGE
    if settings.agent_multi_intent_serial:
        # 串行：先进知识分支，由 route_after_knowledge 接力进任务分支
        logger.info(
            f"[Agent-路由] {sorted(intents)} -> 串行 {NODE_KNOWLEDGE} → {NODE_TASK_AGENT} "
            f"trace_id={get_trace_id()}"
        )
        return NODE_KNOWLEDGE
    # 回滚路径：并行 fan-out
    return [NODE_KNOWLEDGE, NODE_TASK_AGENT]
```

### §3.2 knowledge 出边改条件边（含失败不断路）

新增路由函数：

```python
def route_after_knowledge(state: AgentState) -> str:
    """
    知识分支执行后的去向（§27.3）。

    **判据只看意图与节点可用性，不看知识分支是否成功** ——
    知识分支失败（rag_answer=None）时仍必须进任务分支：任务分支自带
    search_knowledge，可以自愈；若因失败改道 merge，会直接丢掉任务意图。
    """
    intents = set(state.get("intents") or [])
    if (
        "task" in intents
        and _is_node_registered(NODE_TASK_AGENT)
        and settings.agent_multi_intent_serial
    ):
        return NODE_TASK_AGENT
    return NODE_MERGE
```

`build_graph` 中把知识节点的出边由固定边改为条件边：

```python
# 原实现（必须移除，不能与条件边并存）
# for name in (NODE_KNOWLEDGE, NODE_CHAT, NODE_TASK_AGENT):
#     if _is_node_registered(name):
#         g.add_edge(name, NODE_MERGE)

if _is_node_registered(NODE_KNOWLEDGE):
    kb_map = {NODE_MERGE: NODE_MERGE}
    if _is_node_registered(NODE_TASK_AGENT):
        kb_map[NODE_TASK_AGENT] = NODE_TASK_AGENT
    g.add_conditional_edges(NODE_KNOWLEDGE, route_after_knowledge, kb_map)

if _is_node_registered(NODE_CHAT):
    g.add_edge(NODE_CHAT, NODE_MERGE)
if _is_node_registered(NODE_TASK_AGENT):
    g.add_edge(NODE_TASK_AGENT, NODE_MERGE)
```

### §3.3 禁用知识分支补跑

`merge_node` 中原本无条件补跑：

```python
# 串行模式下知识分支**已经执行过**，失败就是失败，不得在流程末尾再跑一次
# 完整生成（那只是把重复调用从开头搬到了结尾）。
if (
    not rag
    and "knowledge_base" in intents
    and not settings.agent_multi_intent_serial
):
    rag = await _recover_knowledge_branch(state, config)
```

### §3.4 不改动的部分

- `_planned_branches`：串行不改变「本轮计划执行的分支」，仍返回 `["knowledge", "task_agent"]`；
- `merge` 的跨轮残留守卫（按 `executed_branches` 判定）：不受影响；
- 语义缓存的 `_intents_key`（intent 维度）：不受影响；
- 审批/追问恢复链路：`resume_graph` 重入主图 → 重入 `task_agent` 节点 → 读
  `configurable.approval_resume` 转内层，**一行不动**。

---

## §4 改造二：merge 拼接（省 LLM）

### §4.1 开关与实现

`agent/graph_builder.py::merge_node`

```python
if rag and task:
    if not getattr(settings, "agent_merge_llm_enabled", False):
        metrics.incr("agent.merge.mode.concat")
        logger.info(
            f"[Agent-汇总] 拼接模式（未调用 LLM） rag={len(rag)} task={len(task)} "
            f"trace_id={get_trace_id()}"
        )
        return {
            "final_response": _concat_answers(rag, task),
            # 两段都已由各自分支流式推送过 → 抑制 finalize 的补推，避免重复
            "final_push_suppressed": True,
        }
    # LLM 合并路径保持不变（含超时与重试）
```

拼接格式（替代裸 `\n\n`，保持可读性）：

```python
def _concat_answers(rag: str, task: str) -> str:
    """拼接两段分支产出（§27.4.1）。任一段为空时只输出另一段。"""
    if not rag:
        return task
    if not task:
        return rag
    return f"【知识依据】\n{rag}\n\n【执行结果】\n{task}"
```

### §4.2 新增 state 字段

`agent/state.py`：

```python
final_push_suppressed: bool   # 本轮是否抑制 finalize 补推（§27.4.1）
```

**必须每轮由 `merge_node` 覆盖写入**（merge 是必经节点），否则检查点会残留上一轮的值。
单分支 passthrough 路径也要显式写 `False`。

### §4.3 保留 LLM 路径的理由

`agent_merge_llm_enabled=True` 时行为与改动前**逐字节一致**，用于：
- 灰度对比输出质量；
- 需要「综合建议」段的场景；
- 出问题时的即时回滚。

---

## §5 改造三：上游 context 注入任务分支

### §5.1 注入机制（基于 F6 的验证结论）

`agent/nodes/task_node.py`：

```python
async def _build_task_agent(
    model: str,
    db,
    executor: DbToolExecutor,
    kb_executor: KnowledgeToolExecutor,
    approval_mode: bool,
    kb_context: str = "",          # 新增：已检索的知识依据（限长后的纯文本）
):
    ...
    system_prompt = TASK_AGENT_SYSTEM_PROMPT
    if kb_context:
        system_prompt = f"{TASK_AGENT_SYSTEM_PROMPT}\n\n{kb_context}"

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,     # 闭包变量，不进 state（F6）
        **kwargs,
    )
```

调用侧（`task_agent_node` 内）：

```python
kb_context = ""
if getattr(settings, "agent_task_kb_context_inject", True):
    kb_context = _build_kb_context_section(
        state.get("context_text") or "",
        int(getattr(settings, "agent_task_kb_context_max_chars", 3000) or 0),
    )
agent = await _build_task_agent(
    model, db, executor, kb_executor, approval_mode, kb_context=kb_context,
)
```

### §5.2 限长构造

`format_context_public` **没有长度上限**（F10），直接注入会与
`_load_task_history`（最多 10 条）叠加撑大 prompt，必须限长：

```python
def _build_kb_context_section(context_text: str, limit: int) -> str:
    """
    构造注入段（§27.5.2）。空输入或 limit<=0 时返回空串（等价于未开启）。
    """
    text = (context_text or "").strip()
    if not text or limit <= 0:
        return ""
    truncated = len(text) > limit
    if truncated:
        text = text[:limit]
    head = "【已检索的知识依据】（由上游知识分支提供，可直接使用）"
    tail = "\n（以上依据已截断，如不足请调用 search_knowledge 补充）" if truncated else ""
    return f"{head}\n{text}{tail}"
```

同时上报计数便于观测：`agent.task.kb_context.injected` / `.truncated`。

### §5.3 提示词改造（**必须同步做，否则注入等于白做**）

`prompt/prompt_storage.py`：

1. 修改 `TASK_AGENT_SYSTEM_PROMPT` 第 3 条：

```
原：3. 若仍缺信息，先调用 search_knowledge 检索知识库补全信息与依据；仍无有效信息时，
      调用 ask_user 工具向用户批量追问缺失内容…

改：3. 若下方已给出【已检索的知识依据】，**优先直接使用，不要重复检索**；
      仅当其中确实缺少本次执行所需的信息时，才调用 search_knowledge 补充；
      仍无有效信息时，调用 ask_user 工具向用户批量追问缺失内容…
```

2. 系统提示末尾补一条约束（防止模型"看不见"注入段）：

```
关于【已检索的知识依据】：
- 该段由系统在上游知识分支检索后注入，属于本次请求的已有信息；
- 若其中包含回答所需的政策、规范、条件，**禁止再次调用 search_knowledge 检索同一内容**；
- 报告中引用该依据时，如实标注来自知识依据段，不要臆造出处。
```

> 依据 §25 E11 的教训：**提示词不定义动作边界，模型就会执行默认动作**（反复检索）。
> 只注入不改提示词，模型仍会按原第 3 条去检索。

### §5.4 恢复轮一致性

审批恢复轮会**重新调用** `_build_task_agent`，此时 `kb_context` 从 `state` 重新生成：

- `context_text` / `context_docs` 写进 state → 进检查点 → 恢复轮可见（**串行改造的附带收益**：并行时知识分支会被中断取消，恢复轮根本拿不到，才需要 `_recover_knowledge_branch` 补跑）；
- 因此恢复轮的注入段与首轮一致，模型决策上下文不漂移。

---

## §6 改造四（前置依赖）：修 finalize 重复推送

**这一项必须在 §4 之前完成**，否则拼接会立刻暴露一个可见的体验退化。

### §6.1 问题

`finalize_node` 无条件补推 `final_response`（`agent_push_final_response=True`），
但「与已推内容重复则不推」的承诺**从未生效**，原因是**三重失效叠加**：

| # | 失效点 | 后果 |
| --- | --- | --- |
| 1 | **`pushed_contents` 未在 `AgentState` 中声明** | LangGraph 的 StateGraph 只认识 state schema 声明的 channel，**未声明字段的写入被静默丢弃且不报错**（已实测确认）→ `task_node` 写了也存不下来 |
| 2 | `_already_pushed_branches`（F4）**全仓无调用点** | 即使有数据也无判定入口（死代码） |
| 3 | `knowledge_node` / `chat_node` **不写**该字段（F5） | 即使前两项修好，去重集合也只认任务分支 |

现状（LLM 合并）下问题不明显：`final_response` 是 LLM **新产出**的文本；
**改成拼接后** `final_response` = 用户刚刚逐字看过的两段 → **必然重复推送**。

### §6.2 修法（实施时相较初稿有修正，以本节为准）

**修正点**：初稿打算让三个分支节点各自补写 `pushed_contents`；实施时改为
**由 `merge_node` 单点覆盖写入**，原因是更好的工程取舍：

- 多意图场景下 `knowledge` / `chat` / `task_agent` 可能在同一 superstep **并行**执行，
  同时写同一个 channel 会触发并发写冲突（需要 reducer），而 reducer 又会带来跨轮累积
  （检查点按 thread 累积）的新问题；
- `merge` 是 **barrier**（全部前驱完成后才执行），单点写入既无并发冲突，
  又天然按「本轮实际产出」判定，且**每轮覆盖写、不留跨轮残留**；
- 分支节点因此**完全不需要改动**，改动面更小。

具体：

1. `agent/state.py` 声明两个字段（**不声明则写入无效**）：

```python
pushed_contents: list          # merge 每轮覆盖写入：本轮各分支实际推给用户的内容
final_push_suppressed: bool    # merge 走拼接路径时写 True；单分支路径显式写 False
```

2. `merge_node` 在**所有**返回点写这两个字段（含 LLM 合并成功 / LLM 降级拼接 /
   拼接模式 / 单分支 passthrough 四条路径）；passthrough 的兜底文案
   （`"暂无可用结果"`）必须 `final_push_suppressed=False`（那是用户唯一能看到失败原因的机会）。

3. `finalize_node`：`final_push_suppressed=True` → 跳过补推；**兜底文案不受抑制标记影响**。

4. `_push_final_response(config, content, state=None)`：真正使用
   `_already_pushed_branches(state)` 做去重（激活死代码）。

---

## §7 收益

### §7.1 省 merge 的 LLM

| 维度 | 数值 |
|---|---|
| 均值延迟 | 约 2~4s |
| **P99 尾部风险** | **最坏 60s × 2 = 120s**（F2），且位于 DAG 必经汇聚点，挂死拖垮整条请求 |
| Token | 输入 = prompt + 问题 + rag 全文 + task 全文（链路最大）；输出 400~600 |
| 附带 | 去掉「超时 + 重试 + 降级」这一整段关键路径逻辑 |

对照 F3：一个"润色"步骤的最坏耗时等于"干完整件活"的预算。

### §7.2 任务分支少两轮

注入后模型不再发起检索，省掉：

```
1 次「决定调工具」的 LLM  +  1 次 embedding  +  1 次向量检索  +  1 次回灌后的 LLM
```

按 §25 实测（24 次 LLM / 136s，单次 LLM 平均约 5s）估算，**少两轮 ≈ 省 10s 量级**。
**注意：仅在多意图场景生效**；单意图 task 无上游可注入，行为不变。

### §7.3 正确性与一致性

- 审批恢复轮的知识上下文不再丢失（F9 的补跑逻辑退出主路径）；
- 消除双口径上下文（F10）：任务分支优先使用上游 `format_context_public` 的结果；
- 消除 C7「跨分支功能不对称」。

### §7.4 体验

- SSE 事件由「两分支交错」变为「先依据、后执行」，顺序可预期；
- 审批卡片出现在知识内容之后 → 用户先看到依据再决定是否批准。

---

## §8 风险与缓解

| # | 风险 | 等级 | 缓解 |
|---|---|---|---|
| R1 | 延迟由 `max(kb, task)` 变 `kb + task` | 中 | 省掉 merge LLM 后**与现状基本持平**；若不可接受，KB 走限长/精简模式，并上线前实测 |
| R2 | 拼接使输出质量下降（无「综合建议」段、无缺信息合并说明） | 中 | 保留 `agent_merge_llm_enabled` 开关；拼接加小标题分段 |
| R3 | 注入段撑大 prompt | 低-中 | `agent_task_kb_context_max_chars`（默认 3000）+ 截断计数指标 |
| R4 | **只注入不改提示词 → 模型照样检索** | **高（确定性）** | §5.3 必须同步落地；冒烟断言系统提示含新约束 |
| R5 | 知识分支失败后断路，任务意图被丢 | 低 | `route_after_knowledge` 判据不看成败（§3.2），冒烟覆盖 |
| R6 | 动态 `system_prompt` 污染内层检查点 | **已排除** | F6 源码验证：闭包变量、临时前置、不进 state |
| R7 | 旧检查点缺少新增 state 字段 | 低 | 全部用 `state.get(...)` 读取并给默认值 |
| R8 | `final_push_suppressed` 跨轮残留 | 中 | 由 `merge_node` **每轮覆盖写入**（含单分支路径显式写 `False`） |
| R9 | 若启用 KB 精简模式，答案缓存 intent key 不含"模式"→ 老缓存返回完整版 | 低 | 本期**不做**精简模式；如做需 cache key 并入 mode 或清缓存 |

---

## §9 可以省去或优化的代码

### §9.1 退出主路径（保留代码，默认不触发）

| 项 | 位置 | 说明 |
|---|---|---|
| `_recover_knowledge_branch` | `graph_builder.py:441` | 串行模式下不再需要（§3.3）；保留供 `agent_multi_intent_serial=False` 回滚 |
| merge 的 LLM 重试循环 | `graph_builder.py:398-429` | 开关关闭时不在主路径；保留供回滚 |
| merge 的并行中断容错注释块 | `graph_builder.py:344-346` | 串行后前置条件不再成立，注释需更新 |

### §9.2 激活 / 修正

| 项 | 位置 | 说明 |
|---|---|---|
| `_already_pushed_branches` | `graph_builder.py:508` | 由**死代码**变为生效（§6.2） |
| `pushed_contents` 写入方 | `knowledge_node` / `chat_node` | 由只有 task 写 → 三个分支都写 |

### §9.3 建议顺带优化（独立小改动）

| 项 | 说明 |
|---|---|
| 上下文口径统一 | `_build_truncated_context`（800/4000）与 `format_context_public` 二选一，避免同一问题走不同分支得到不同上下文 |
| `search_knowledge` 的 `k` 归一化 | `k` 当前由模型自由传（1~10），而 `top_k` 是检索缓存的**强过滤维度** → 参数漂移即 miss。建议钳制到固定档位（如仅允许 4/8） |
| 单分支 passthrough 的重复推送 | 现状单分支场景 finalize 也会补推（去重未生效），§6 一并修掉 |

---

## §10 验收与回归

### §10.1 新增冒烟脚本

`scripts/smoke_multi_intent_serial.py`（离线、可重复、import 生产实现）：

| # | 断言（不变量） |
|---|---|
| 1 | 串行开关开 + 多意图 → `route_by_intents` 返回**单个** `"knowledge"`（不是列表） |
| 2 | 串行开关关 + 多意图 → 仍返回**列表** `["knowledge", "task_agent"]`（回滚路径完好） |
| 3 | `route_after_knowledge`：`rag_answer=None`（知识分支失败）时**仍返回** `"task_agent"` |
| 4 | `route_after_knowledge`：非多意图 → 返回 `"merge"` |
| 5 | 拼接模式（`agent_merge_llm_enabled=False`）：注入计数用假 LLM，断言**零调用** |
| 6 | 拼接结果同时包含两段内容，且含 `【知识依据】` / `【执行结果】` 标题 |
| 7 | 注入生效：`kb_context` 非空时 `system_prompt` 含「已检索的知识依据」字样 |
| 8 | 注入限长：超长 `context_text` → 注入段长度 ≤ 配置上限 + 固定头尾文案长度 |
| 9 | 单意图 task（`context_text` 为空）→ 注入段为空串，`system_prompt` 与改动前一致 |
| 10 | 串行 + 知识分支失败 → `merge` **不调用** `_recover_knowledge_branch` |
| 11 | `final_push_suppressed=True` → finalize 不推送；`False` → 正常推送 |
| 12 | `TASK_AGENT_SYSTEM_PROMPT` 含「不要重复检索」约束（防 R4 回归） |

### §10.2 回归

- 全量既有冒烟脚本（30+）必须全绿，尤其：
  `smoke_approval_resume.py`（审批恢复链路）、`smoke_retrieval_cache.py`、
  `smoke_db_pool_pre_ping.py`；
- `compileall` 通过；
- **断言一律写不变量**，禁止写「某源码串出现 N 次」（C2/C8 已踩两次）。

### §10.3 线上观测指标

| 指标 | 预期 |
|---|---|
| `agent.merge.mode.concat` / `.llm` | 多意图请求应几乎全为 concat |
| `agent.task.kb_context.injected` / `.truncated` | 多意图时 injected 计数 ≈ 多意图请求数 |
| `cache.retrieve.hit` | 注入生效后，任务分支的检索调用应显著下降 |
| `agent.finalize.push_suppressed` | 与 concat 次数一致 |
| 端到端 P50 / P99 | 对照 R1，确认未劣化 |

---

## §11 实施顺序与回滚

每步独立可验证、可回滚：

| 步骤 | 内容 | 开关 | 回滚方式 |
|---|---|---|---|
| **Step 1** | 改造四：修 finalize 去重 + 补 `pushed_contents` | 无新开关（行为修正） | `agent_push_final_response=False` |
| **Step 2** | 改造三：context 注入 + 提示词改造 | `agent_task_kb_context_inject` | 关开关即回到改动前 |
| **Step 3** | 改造一：多意图串行化 | `agent_multi_intent_serial` | 关开关即恢复并行 fan-out |
| **Step 4** | 改造二：打开拼接（关 LLM 合并） | `agent_merge_llm_enabled=False` | 置 `True` 恢复 LLM 合并 |

**建议顺序理由**：
- Step 1 必须最先（否则 Step 4 会重复推送）；
- Step 2 不碰图结构，风险最低，收益可独立观测（看 `cache.retrieve.hit` 下降）；
- Step 3 是图结构变更，放 Step 2 之后可单独评估串行本身的影响；
- Step 4 最后，因为**注入做完后拼接才合理**——两段内容高度重叠时，LLM 合并做的其实是"费力去重"。

### §11.1 新增配置汇总

```python
# core/config.py
agent_multi_intent_serial: bool = True          # 多意图串行（False 恢复并行 fan-out）
agent_merge_llm_enabled: bool = False           # merge 是否调用 LLM 合并（False 走拼接）
agent_task_kb_context_inject: bool = True       # 上游知识依据是否注入任务分支
agent_task_kb_context_max_chars: int = 3000     # 注入段字符上限
```

---

## §12 明确不在本方案范围内

以下问题与本次改造**无关**，需要独立立项（避免误以为做了本方案就解决）：

- **工具能力缺口**：`query_data` 无 COUNT / SUM / GROUP BY / JOIN，上限 50 行
  —— 这是 trace=0e7fc2dd93554f24「136s 未解决」的**根因**（§25）；
- **提示词能力边界**（§25 E11）：模型不知道「做不到什么」；
- **错误回灌文案**（§25 E8）：对确定性拒绝也回「请修正参数后重试」＝喂养空转；
- **连接池 non-checked-in 告警**（§25 E10）：未根治，需把超时下沉到服务端。
