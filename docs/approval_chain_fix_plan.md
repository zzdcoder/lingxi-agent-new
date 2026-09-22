
---

## 七、实施记录（本轮已落地）

> 待确认问题答复：① 双意图并行 → 知识库 + 审批卡片**两段分栏展示**；
> ② 审批等待期**允许用户离开会话**；③ 逻辑删除合理；④ 修复。
> 据此实施，前后端均已改动并通过校验。

### 7.1 后端改动清单

| 编号 | 文件 | 改动 |
| --- | --- | --- |
| F3.0 | `agent/nodes/task_node.py` | 恢复轮读 `configurable.approval_resume` / `clarify_resume`，转 `Command(resume=...)` 交内层 Agent；首轮仍传带稳定 id 的 `HumanMessage` |
| F3.0 | `agent/approval/approval_service.py` | `resume_graph` 构造 `resume_value` 并注入 `configurable.approval_resume`（含 `approval_id`） |
| F3.0 | `agent/approval/clarify_service.py` | 同上，注入 `configurable.clarify_resume`；`_resume_config` 新增 `resume_value` 入参 |
| F3.1 | `approval_service.resume_graph` | 恢复后检测返回值 `__interrupt__`：记 error 日志 + `resume_interrupted` 指标 + 任务标 failed + 明确提示，**不再误报成功** |
| F3.1 | `clarify_service.resume_clarify` / `cancel_clarify` | 同上二次中断检测（取消路径仍标 `canceled`，用户取消意图明确） |
| F3.2 | `approval_service._notify_result` | 推送 `approval_completed` 状态事件 + 结论文案（**始终推送**，不再按内容比对决定）；`queue is None` 时记日志 + `push_no_queue` 指标 |
| F3.2 | `approval_service._build_result_summary` | 新建：按 decision 生成结论文案（approved/rejected/canceled 三态） |
| F3.2 | `api/routes/approval.py` | 决策响应补 `task_execution_id` / `conversation_id`（前端轮询句柄） |
| F3.3 | `approval_service._update_task_after_resume` | 补 `approved → completed` 分支（原实现任务永久停 `running`，且会被下一轮 `_find_running_task` 复用） |
| F3.3 | `api/routes/agent.py::get_task_status` | 补越权校验（任务 `user_id` 或会话归属），原先任意用户可读他人任务 |
| F3.4 | `core/config.py` | 新增 `agent_delete_mode`(默认 `logical`) / `agent_delete_flag_column` / `agent_delete_flag_true_value` |
| F3.4 | `agent/tools/agent_tool.py::delete_data` | 逻辑删除分支（`UPDATE ... SET flag=1`）；标记列不存在时**回落物理删除并记录 `fallback_reason`**；结果体回传 `mode` / `flag_column` |
| F1.1/F1.2 | `prompt/prompt_storage.py` | `TASK_AGENT_SYSTEM_PROMPT` 新增「关于写操作审批（重要）」段；第 2 条「超出授权范围」收窄为「白名单之外的资源」，明确写操作**不属于**超范围；第 5 条改为「确认写意图后立即调用」 |
| F1.3 | `agent/tools/agent_tool.py::_build_tools` | 三个写工具 description 补注「调用后自动进入审批等待，属正常流程，不要因需审批而拒绝调用」 |
| F2.1 | `agent/streaming.py::put_content` | 新增 `branch` 参数（空串时不写该字段，兼容旧前端） |
| F2.1 | `knowledge_node` / `chat_node` / `task_node` | 全部 `put_content` 调用点标注 `branch`（knowledge / chat / task） |
| F2.4 | `agent/graph_builder.py::finalize_node` | 补推 `final_response`（`branch="final"`）；新增 `_push_final_response`；开关 `agent_push_final_response` |
| F2.4 | `core/config.py` | 新增 `agent_push_final_response`（默认 True） |

**决策：F3.0 采用方案 B（显式透传），未采用原方案的方案 A（去掉内层 checkpointer）。**
理由：内层 thread 隔离是 §17.x 的刻意设计，共用检查点会导致主图 `messages` channel
被内层整条覆盖、跨轮串味固化（见 `task_node.py` 的长注释）。去掉 checkpointer
虽更"简单"，但会把已修复的串味缺陷重新引入。方案 B 保留隔离，仅显式传递决策。

### 7.2 回归验证结果

**`scripts/smoke_approval_resume.py`**（已升级为三段判据，4 条断言全 PASS）：

```
[对照组 GOOD] 直接 Command(resume)          → 中断=False, delete_data 执行=['user:id=3']
[缺陷活样本 BAD] 修复前方式（全新 input）    → 中断=True,  delete_data 执行=[]
[修复路径 FIX] configurable.approval_resume → 中断=False, delete_data 执行=['user:id=3']

PASS 对照组正确执行 delete_data
PASS 对照组恢复后未再次中断
PASS 缺陷活样本符合预期（全新 input → 再次中断、写操作未执行）
PASS 修复路径与对照组行为一致 —— 审批决策已正确透传（F3.0 已修复）
```

脚本第三段 `_resolve_invoke_input` **逐行复刻** `task_node` 的取值逻辑，
验证的是生产代码路径而非脚本自带期望值。

### 7.3 前端改动清单（`D:/lingxi-agent-portal`）

| 编号 | 文件 | 改动 |
| --- | --- | --- |
| F2.1 | `src/services/llm.ts` | `onContent(text, branch)` 透传 branch；`StreamCallbacks` 类型更新 |
| F2.2 | `src/hooks/useChat.ts` | 新增 `appendSegment` / `joinSegments`；`onContent` 按 branch 分段累积；`content` 保持为各段拼接（历史消息载体） |
| F2.2 | `src/components/MessageBubble.tsx` | 按 `segments` 分段渲染，多分支时显示「知识库结论 / 综合结论」小标题 |
| **F2.3** | `src/components/MessageBubble.tsx` | **缺陷 2 根因修复**：`shouldRenderBubble` 改为「卡片与正文并列」——原先 `{!hasApproval && ...}` 使挂上审批卡后正文整段不渲染，用户看到回答"突然消失" |
| F2.3 | `src/hooks/useChat.ts` | `approval_required` / `clarification_required` 不再清空 `thinkingContent`，改为仅结束思考/流式态 |
| F3.2 | `src/services/llm.ts` | 新增 `getTaskStatus` / `BackendTask` / `TASK_TERMINAL_STATUSES`；决策响应类型补 `task_execution_id` |
| F3.2 | `src/hooks/useChat.ts` | 新增任务轮询：`startPollingTask`（2s 间隔 / 上限 90 次）、`clearPoll` / `clearAllPolls`、`handleApprovalDecided`；结果作为 `task` 分支回填（含 SSE/轮询去重） |
| F3.2 | `src/components/ApprovalCard.tsx` | `onDecided` 回传 `task_execution_id` 供上层轮询 |
| F3.2 | `MessageBubble` / `MessageList` / `App.tsx` | 透传 `onApprovalDecided`；新增「审批已受理，正在执行…」提示 |
| — | `scripts/verify-segments.mjs` | 新建：分段归并语义校验（6 条断言全 PASS） |

**TypeScript 编译零错误、`vite build` 成功**；ESLint 仅剩 2 条**改动前既有**的
报错（`App.tsx:46` setState-in-effect、`llm.ts:39` explicit-any，均在我未触碰的行）。

### 7.4 未做与遗留

1. **历史脏数据清理**（原待确认问题 4）：`task_execution.result` 曾因
   `finalize_node` 与 `_persist_final_snapshot` 的时序被误填。本轮**未做数据修复
   SQL** —— 涉及线上数据变更，需单独评估影响面与备份方案后执行。
2. **`insert_data` 默认不审批**（`agent_insert_requires_approval=False`）保持原状，
   未随本轮调整。
3. **回归项未全部实测**：验收用例 9（追问链路回归）、10（多写操作批次审批）、
   11（审批通过但执行失败）**需在联调环境实跑验证** —— F3.0 触及 HITL 核心机制，
   本地脚本只能覆盖单写操作 approve 路径。
 写 state，
`finalize_node` 只落库 —— **没有任何 `put_content(final_response)`**。
当前是"靠各分支自己推 content、merge 结果不推"。这导致
`task` + `knowledge_base` 并行时，用户看到的是两段拼接的原始内容，
而不是 merge 润色后的 `final_response`（而落库的却是 `final_response`）——
**看到的和存下来的不一致**。建议在 `finalize_node` 补一次推送：

```python
# finalize_node 内，落库后
queue = get_sse_queue(config)
if settings.agent_push_final_response and final:
    already = state.get("rag_answer") and state.get("task_answer")
    if already or state.get("error"):   # 多分支或异常时才推，避免单分支重复
        await put_content(queue, final, branch="final")
```

配套新增开关 `agent_push_final_response: bool = True`。

**影响范围**：`agent/streaming.py`、三个 node、`approval_service.py`、
`graph_builder.py`、**前端 SSE 消费逻辑**（需同步改，属跨端变更）。
**回归风险**：中。前端若未同步处理 `branch` 字段，行为与现在一致（向后兼容，
因为新字段是**追加**的）。这是刻意的兼容设计。

---

## 三、缺陷 3：审批通过后无执行、无推送

### 现象

审批卡片点了同意，`POST /api/approval/.../decision` 返回 200；
恢复日志显示 `[Approval] 审批恢复完成`；但 `user` 表数据没变，前端也没收到任何推送。

### 根因

**根因已确认为「审批恢复轮内层 Agent 认为任务已完成，`delete_data` 从未被调用」。
且「无推送」有一条独立的确切缺陷。**

#### 3a. 恢复轮内层 Agent 用「全新 input + 旧 thread」重跑，审批决策被彻底丢弃（**已实验证实**）

**这是"用户没删除"的确切根因。**

`agent/nodes/task_node.py:211` 恢复轮的执行方式：

```python
result = await agent.ainvoke(invoke_input, config=invoke_config)
# invoke_input = {"messages": [HumanMessage(content=user_input, id=f"task-input-{task_execution_id}")]}
```

`invoke_input` 是一个**普通 dict**，不是 `Command(resume=...)`。
审批决策只传给了**主图**（`approval_service.resume_graph` 的
`Command(resume=...)`），**没有转交给内层 Agent**。

**最小复现实验已证实**（`temp/verify_f30b.py`，结论稳定）：

```
1st invoke  → __interrupt__=True      （第一次正常中断）
2nd invoke  → __interrupt__=True      （用全新 dict 再 invoke）
             请求模型次数: 2
             delete_data 实际执行: []        ← 从未执行
消息序列:
  HumanMessage  '删除a'
  AIMessage     tool_calls=['delete_data']   ← 第 1 轮模型请求
  AIMessage     tool_calls=['delete_data']   ← 第 2 轮模型又请求一遍
```

行为链条（与生产日志逐条吻合）：

1. 第二次 `ainvoke` 带的是**新输入**，LangGraph 把它当作**一次全新的 run**，
   挂在同一个 thread 上 → 检查点里那条"待审批中断"被**直接丢弃**
   （既没 approve 也没 reject，静默作废）；
2. 内层 Agent 从头重跑 → **再次请求模型** → 模型看到用户问题，
   **又发起一次 `delete_data` 调用**；
3. HITL 中间件再次拦截 → **第二次 `GraphInterrupt`**；
4. 外层 `resume_graph` 只准备了**一次** `Command(resume=...)`，
   无法处理第二次中断 → 流程在此终结，`delete_data` 从未执行。

**生产日志完全对应**（trace `ab9589981356411f`，恢复轮仅 2.8s）：

```
11:39:10.558  开始执行（恢复轮）
11:39:13.400  HTTP 200 (dashscope)      ← 唯一一次 LLM 往返
11:39:13.429  节点 task_agent 异常: GraphInterrupt   ← 第二次中断
11:39:13.443  UPDATE task_execution SET result='## 当前系统用户清单...'
```

只有**一次** LLM 往返、**第二次** `GraphInterrupt`、`user` 表**仍有 4 人
`deleted=0`** —— 三个证据与实验结果一一对应。

**旁证**：模型在恢复轮给出的是"当前系统用户清单"（4 位用户）。
这不是删除报告，而是模型在**没有得到删除确认**的情况下，
基于上下文里已有的查询结果凑出的一份回答 —— 所以任务被误判为"完成"。

**修复 F3.0（最高优先级，本轮核心）**：把审批决策透传给内层 Agent。三种方案：

**方案 A（推荐，改动最小）**：内层 Agent **不挂 checkpointer**，
HITL 中断直接冒泡到主图，主图的 `Command(resume=...)` 直达内层中间件。

```python
# agent/nodes/task_node.py::_build_task_agent
if approval_mode:
    kwargs["middleware"] = [HumanInTheLoopMiddleware(...)]
    # 删除这行：kwargs["checkpointer"] = get_checkpointer()
```

代价：内层 Agent 不再有跨调用的线程记忆。当前 `invoke_input` 每轮只传
单条 `HumanMessage`，本就不依赖历史，因此**功能等价**。

**方案 B（语义最显式）**：保留 checkpointer，恢复轮显式传 `Command`。

```python
# approval_service.resume_graph：把决策塞进 configurable
configurable["approval_resume"] = _build_resume_value(decision, reason, action_count)

# task_node 恢复轮：
resume_value = (config or {}).get("configurable", {}).get("approval_resume")
invoke_input = Command(resume=resume_value) if resume_value else invoke_input
```

**方案 C（最省事但有隐患）**：不隔离内层 thread，让内外共用同一个 checkpointer
与 thread_id —— 主图的 `Command(resume=...)` 自然被内层 Agent 看到。
但当前代码**刻意**隔离了 thread（`_inner_thread_id` 的注释说明是为了避免
主图/内层检查点互相污染），改回共用会引入新的相互干扰。**不建议。**

> **实施建议**：先上方案 A（一次改动、可立即用同一份复现脚本验证）；
> 若后续内层 Agent 需要跨中断保留工具调用记忆，再切方案 B。

**验证方式**：把 `temp/verify_f30b.py` 的第二个 `ainvoke` 换成
`Command(resume={"decisions":[{"type":"approve"}]})`，应当观察到
`delete_data 实际执行: ['user:id=3']` 且 `2nd 中断: False`。

#### 3b. 审批结果推送存在双重缺陷：去重条件错误 + 缺少「审批完成」事件

`approval_service.resume_graph` 的最终推送逻辑：

```python
if final_response and final_response != task_answer:
    await put_content(queue, final_response)
elif not final_response and task_answer:
    await put_content(queue, task_answer)
```

**缺陷 1（去重条件错误）**：本案例中
`final_response == task_answer == "## 当前系统用户清单..."`，
两个分支**都不满足** → **一条都不推**。

这个去重条件的假设是"任务节点已推送过 `task_answer`，恢复时无需重复"。
但它在**审批恢复路径上不成立**：审批等待期间前端已经渲染了
"知识库内容 + 审批卡片"，随后需要一条**明确的审批结论**来告诉用户
"审批已通过 / 已拒绝，结果如下"。靠内容比对来判断"推不推"是脆弱的 ——
内容相同不代表用户已经看到过。

**缺陷 2（缺事件）**：整条链路只发 `approval_decided`（决策回显），
**没有任何"审批执行完成"的事件**。前端无法知道恢复执行何时结束，
只能被动等 `content` 或 `done`。若 `content` 因去重被跳过，
前端就彻底收不到信号 —— 用户感知即"点了同意，然后什么都没有"。

**修复 F3.2'**：改用**显式事件**驱动，不再靠内容比对：

```python
# resume_graph 内，无论内容是否重复，都推一条审批结论
await put_status(
    queue, "approval_completed",
    approval_id=approval_id, decision=decision,
    task_execution_id=approval.task_execution_id,
    summary=...,
)
# 结论文案始终推送（前端需据此渲染结果区）
await put_content(queue, summary, branch="task")
```

并在 `docs/` 的 SSE 协议表里登记 `approval_completed` 事件，
前端据此结束"审批执行中"状态。

**缺陷 3（可选加固）**：`_pending_queues` 是进程内字典，
多 worker 部署时 `/decision` 与首轮 SSE 可能落在**不同进程**，
`get_queue()` 返回 `None`，推送静默跳过（只有 `if queue is not None` 守卫，
无任何日志）。建议：`get_queue()` 返回 `None` 时**记一条 INFO 日志**，
并在协议层面明确"前端必须以轮询兜底"。

### 已确认的独立缺陷（不依赖复现）

**F3.1（必做）`task_node` 的 `except GraphInterrupt` 在恢复轮吞掉第二次中断，
让任务被误标为「已完成」。**

结合已证实的 F3.0：恢复轮内层 Agent 会**再次**中断。此时
`task_node` 的 `except GraphInterrupt` 分支执行：

```python
await _update_task_record(db, task_execution_id, status="running",
                          tool_calls=_merge_tool_calls(executor, kb_executor))
raise
```

`raise` 让中断继续上抛到主图 → `resume_graph` 的 `graph.ainvoke` **正常返回**
（LangGraph 用返回值而非异常表达中断），于是代码继续往下走：

```python
result_dict = result if isinstance(result, dict) else {}
final_response = result_dict.get("final_response")   # 第二次中断 → 无此字段 → None
task_answer = result_dict.get("task_answer")         # → None
await _update_task_after_resume(db, approval, decision, None)
```

**`_update_task_after_resume` 收到 `final_response=None`**，而它对
`decision="approved"` **不做任何状态赋值**（见 F3.3）→ 任务停在 `running`。

**更严重的是**：整个恢复流程把"第二次中断"当成了"执行完成"，
`resume_graph` 末尾照常打日志 `[Approval] 审批恢复完成`，
并 `metrics.incr(K_APPROVAL_PREFIX.format("approved"))` ——
**审批被记为成功，实际写操作从未执行**。这是静默的假成功，危害最大。

**修复 F3.1**：恢复轮必须能识别"我又中断了"，并走正确分支。

```python
# resume_graph 内，invoke 返回后
result_dict = result if isinstance(result, dict) else {}
if result_dict.get("__interrupt__"):
    # 恢复后再次中断 = 上一轮决策未生效（F3.0 的后果）
    logger.error(f"[Approval] 恢复后再次中断，决策未生效: approval_id={approval_id}")
    metrics.incr(K_APPROVAL_PREFIX.format("resume_interrupted"))
    await _mark_task_error(db, approval, "审批决策未生效：恢复后再次触发中断")
    if queue is not None:
        await put_content(queue, "审批已受理，但执行未生效，请重新发起该操作")
    return
```

同时 `metrics.incr(...format(decision))` 必须**只在真正成功时**计数，
不得无条件打成功点。

**F3.2（必做）SSE 队列消费生命周期与审批恢复解耦 —— 多 worker 下必然失效。**

`resume_graph` 的推送全部走 `queue = get_queue(approval_id)`，
而 `_pending_queues` 是**进程内字典**（`agent/approval/approval_service.py:58`）。
两个失效场景：

1. **多 worker / 多实例部署**：`POST /api/approval/{id}/decision` 与首轮
   `POST /api/agent/chat` 落在**不同进程** → `get_queue()` 返回 `None` →
   推送被 `if queue is not None` 静默跳过（**无任何日志**）。
2. **进程重启**：审批等待期间服务重启 → 注册表清空 → 同上。

本案例是单进程，故走的是 3b 的"去重条件"缺陷；
但**该缺陷在多 worker 部署下会必然触发**，属必须一并修的结构性问题。

**修复 F3.2**：
1. `/decision` 响应体返回 `approval_id` + `task_execution_id`，
   前端**以轮询为主通道**：决策提交后轮询
   `GET /api/approval/{id}`（终态）+ `GET /api/agent/tasks/{id}`（结果），
   SSE 只作为"快路径"。
2. `get_queue()` 返回 `None` 时**记 INFO 日志**并计数
   （`agent.approval.push_no_queue`），消除当前静默盲区。
3. `GET /api/agent/tasks/{id}` 的响应补 `final_response`
   （即 `task_execution.result`），让轮询能拿到完整结果。

**F3.4（建议）`delete_data` 使用物理删除，与业务语义可能不符。**

`agent/tools/agent_tool.py::delete_data` 执行的是
`DELETE FROM \`user\` WHERE ...`（物理删除）。而 `user` 表有
`deleted INT` 字段（"逻辑删除：0-正常 1-删除"），
且**本案例中模型自己主动建议了逻辑删除**：

> 如果按你提供的 `user` 表结构处理，建议对 `testuser99` 做逻辑删除，
> 而不是物理删除。

模型基于表结构做出的这个判断是**合理的**。建议把删除策略改为可配置：

```python
agent_delete_mode: str = "physical"   # physical / logical
agent_delete_flag_column: str = "deleted"
```

`logical` 模式下 `delete_data` 实际执行 `UPDATE user SET deleted=1 WHERE ...`，
并在工具描述里向模型说明。**此项需产品确认**（见第六节问题 3）。

> 注：本条与"删除未执行"无关 —— 删除未执行的根因是 F3.0。
> 本条是修复之后才会显现的**语义正确性**问题。

**影响范围**：
- F3.0/F3.1：`agent/nodes/task_node.py`、`agent/approval/approval_service.py`
- F3.2：`api/routes/approval.py`、`api/routes/agent.py`、前端轮询逻辑
- F3.3：`agent/approval/approval_service.py`
- F3.4：`agent/tools/agent_tool.py`、`core/config.py`
- F1.x：`prompt/prompt_storage.py`、`agent/tools/agent_tool.py`（描述）
- F2.x：`agent/streaming.py`、三个 node、`graph_builder.py`、**前端**

**回归风险**：
- F3.0 改动最小但**触及 HITL 核心机制**，必须完整回归：
  单写操作审批、多写操作批次审批、审批拒绝、审批取消、追问（ask_user）中断。
  **追问链路同样依赖内层 Agent 的 interrupt**（`ask_user` 工具），
  方案 A 去掉内层 checkpointer 后需重点验证追问是否仍能正常中断与恢复。
- F3.2 涉及前端轮询与 SSE 并存，需回归"审批等待期断线重连"
  "审批等待超时"两条既有路径。

---

## 四、修复优先级与实施顺序

| 序 | 编号 | 内容 | 依赖 | 风险 |
| --- | --- | --- | --- | --- |
| 1 | **F3.0** | **内层 Agent 透传审批决策（方案 A：去掉内层 checkpointer）** | 无 | **中（触及 HITL 核心）** |
| 2 | F3.1 | 恢复后再次中断的识别 + 不误报成功 | F3.0 | 低 |
| 3 | F3.3 | `_update_task_after_resume` 补 `completed` 分支 | 无 | 低 |
| 4 | F1.1/F1.2 | 系统提示词改写（审批=正常流程） | 无 | 低 |
| 5 | F3.2 | 恢复推送解耦 + 前端轮询兜底 + `approval_completed` 事件 | 无 | 中 |
| 6 | F2.1/F2.3 | SSE `content` 加 `branch` + 审批卡片追加语义 | 前端同步 | 中 |
| 7 | F2.4 | `finalize_node` 补推 `final_response` | 6 | 中 |
| 8 | F3.4 | `delete_data` 删除策略可配置（待产品确认） | 无 | 低 |
| 9 | F1.3/F2.2 | 工具描述补注 + 多分支分栏渲染 | 4/6 | 低 |

**建议第 1~4 项本轮先做**。这四项只动后端、能直接消除三个现象中的
「模型拒答」与「审批后无响应（数据没删）」两个，收益最高。
**F3.0 是本轮唯一的必做核心** —— 不修它，删除操作永远不可能成功。

**验证脚本已就绪**：`temp/verify_f30b.py`（当前输出
`delete_data 实际执行: []` 复现缺陷；修好后应输出
`['user:id=3']`）。建议把它转正为
`scripts/smoke_approval_resume.py` 纳入回归。

---

## 五、验收用例

| # | 用例 | 预期 |
| --- | --- | --- |
| 1 | 单意图 `删除用户testuser99` | 模型**不拒答**，直接发起 `delete_data`，弹审批卡片 |
| 2 | **审批通过（F3.0 核心验收）** | `user` 表 `testuser99` **确实被删除**，任务状态 `completed`，`tool_calls` 审计含 `delete_data` |
| 3 | 审批拒绝 | 写操作不执行，任务状态 `rejected`，前端展示拒绝说明 |
| 4 | 双意图 `删除X，然后查Y` | 知识库气泡与审批卡片**同时存在**，互不覆盖 |
| 5 | 用例 4 审批通过后刷新页面 | 知识库答案 + 审批卡片 + 任务结果三者齐全，顺序正确 |
| 6 | 审批等待期断开 SSE 再点同意 | 前端经轮询拿到最终结果（F3.2 核心验收） |
| 7 | 审批等待超时（不点） | 任务不卡 `running`，前端有明确提示 |
| 8 | 用户未授权写意图（"看看张三的信息"） | 模型**不调用**写工具（安全边界不回退） |
| 9 | **追问链路（`ask_user`）回归** | F3.0 方案 A 后，追问仍能正常中断/恢复（重点回归项） |
| 10 | 多写操作批次审批 | 一次中断含 2 个写操作，决策数量匹配，两者都执行 |
| 11 | 审批通过但执行失败 | 任务标 `failed`，**不**记成功指标，前端有明确说明 |

---

## 六、待确认问题（需产品/前端决策）

1. **双意图并行时，用户期望看到什么？** 是"知识库答案 + 审批卡片"两段，
   还是等审批走完再给一份 merge 后的合并回答？这决定 F2.2 的渲染方案。
2. **审批等待期是否允许用户离开会话？** 若允许，恢复结果必须落库 + 轮询；
   若不允许，可保留现 SSE 长连接方案。
3. **`delete_data` 用物理删除是否合理？** 日志里模型主动建议了逻辑删除
   （`deleted=1`），而工具实现是 `DELETE FROM`。`user` 表有 `deleted` 字段，
   业务语义上**逻辑删除更可能符合预期**（见 F3.4）。
4. **`task_execution.result` 当前被误填为"用户清单"** —— 修复后
   是否需要对历史这类脏数据做一次清理/标注？

---

## 八、第二轮生产日志复盘（trace=00b0b22f2b0446b7，2026-09-21 14:45~14:48）

> 本轮只做**诊断**，不改代码。§七 的修复（F1/F2/F3）已全部落地并生效，
> 但本节四个问题揭示了 **F3.0 只修好了一半**：决策确实送到了内层，
> 可内层在「批准执行之后」又被模型拖出一次新中断，把成果吃掉了。

### 8.0 本轮日志的事实基线

| 项 | 值 |
| --- | --- |
| trace_id | `00b0b22f2b0446b7`（恢复段 `fd7a8c062f1540f1`） |
| conversation_id | `951c376c-6ec2-4ec7-b401-8528f8b99785` |
| task_execution_id | `740f0afc-6795-4f1f-8157-47e4bdb8296d` |
| 模型 | `qwen3.8-max-0902` |
| 审批单 | `01529795-4a84-4e7f-a6ae-08e6034fe49a`，`approved`，`actions` 长度 **1** |
| 首次中断 id | `3ef485e2d85cbe2f0e1adba5688cf58b` |
| 恢复后中断 id | `1ba66082c993bd93fce3189c18dc732e`（**不同 → 是全新中断**） |
| 终态 | `task_execution.status='failed'`，`error='审批决策未生效：恢复执行后再次触发人工介入中断'` |
| `user` 表 | `testuser99` **仍在**（`id=3, deleted=0`）→ 写操作确实没执行 |

**已确认修复生效的部分（§七 成果保留）**：

```
[Agent-任务] 恢复轮透传人工介入决策: type=dict keys=['decisions']
[Approval] 恢复后再次中断，本次决策未生效: ... 中断数=1
UPDATE task_execution SET status='failed', error='审批决策未生效：...'
```

F3.0（决策透传）与 F3.1（不误报成功）都按设计工作。**没有静默假成功**，
这本身是本轮最有价值的护栏。

---

### 8.1 问题一：为何调用这么多次 LLM 请求

不是"一轮请求调了十几次模型"，而是**同一个内层 thread 上跑了三轮完整的
Agent 循环**，每轮各自 2~3 次模型调用，日志里连成一串。

| 轮次 | 触发者 | 内层 thread | 模型调用 | 产出 |
| --- | --- | --- | --- | --- |
| 轮 1 | 首轮 `/api/agent/chat` | `…::task_agent::<tid1>` | 2 次 | 第 1 次中断 `3ef485e2…` |
| 轮 2 | 用户**同一会话再发一次**（或前端重放） | 同上 | 2~3 次 | 第 2 次中断（**新 id**） |
| 轮 3 | `approval_service.resume_graph` | 同上 | 3 次 | 第 3 次中断 `1ba66082…` |

**放大机制（根因）**：`task_node` 的 `invoke_input` 首轮用
`HumanMessage(content=user_input, id=f"task-input-{task_execution_id}")`
—— **稳定 id 是刻意的**（避免 `add_messages` 重复 append）。但它带来的副作用是：

> 只要 `task_execution_id` 相同（`_find_running_task` 复用同一条 running 记录
> → 同一个 `_inner_thread_id`），后续任何一次"new input 进入"都会
> **upsert 覆盖那条历史 HumanMessage**，于是内层 Agent 的
> `last_ai_msg.tool_calls` 仍然带着上一轮未消费的写操作请求 →
> `HumanInTheLoopMiddleware.after_model` 再次 `interrupt()`。

实测（`scripts/smoke_resume_multiround.py`）已固定该结论：
连续两次 new-input 进入 → **两个 id 不同**的中断
（`a51c02ac…` / `c2220322…`），且 `delete_data` 一次都没执行。

---

### 8.2 问题二：报错信息 / 异常分析

本轮**没有真正的异常（exception）**，只有业务失败状态 + 一行 ERROR 日志。

1. **`[Approval] 恢复后再次中断，本次决策未生效`**
   —— 这是 §七 F3.1 **主动加的护栏**，命中即 `metrics.incr(resume_interrupted)`
   且任务标 `failed`。它是**结论**不是**原因**：说明恢复轮 `graph.ainvoke`
   正常返回、但返回值里带 `__interrupt__`。

2. **中断 id 不同才是关键证据**：`3ef485e2…` ≠ `1ba66082…`。
   若 F3.0 是把 `Command(resume=…)` 沿同一条中断链投递，被消费的应当是
   **同一个 id**。id 变了 ⇒ 中间又发生了一次「丢弃旧中断 + 生成新中断」。

3. **没有 `ValueError: Number of human decisions (N) does not match…`**
   —— 本轮 `actions` 长度为 1、`decisions` 也是 1，**数量是对齐的**。
   该分支只在「一次 AI 消息里同时有写操作 + 只读操作」时才触发
   （`interrupt_indices` 只统计**写了 `interrupt_on` 的工具**，
   而 `actions` 也只统计写工具，两者口径一致；但若未来把只读工具也纳入
   `interrupt_on` 就会对不上）。**本轮不是这个原因，不要误判。**

4. **`filters` 被传成字符串**：日志与 DB 里都是
   `"filters": "{\"username\": \"testuser99\"}"`（字符串），
   而工具签名是 `filters: Optional[Dict[str, Any]]`。本轮因为审批就没走通，
   所以这个格式问题被掩盖了 —— **修复审批后它会立刻暴露**（`_build_where`
   拿到 str 会拼出错误的 WHERE）。列为待修项 F3.6。

---

### 8.3 问题三：为何审批通过后还是执行不了具体操作（**核心**）

**结论：`delete_data` 其实执行了，只是执行完之后模型又发起一次同样的调用，
把这次中断留在了返回值里，`approval_service` 据此判失败 ——
而事务/审计也没能反映"已执行"。**

实测复刻（与生产时序一致）：

```
R1 首轮请求              [LLM#1] msgs=2  -> 中断 cb1aaee8de18df356cecbcccba646f01
R2 同一 thread 再来一轮   [LLM#2] msgs=3  -> 中断 01b9101ecee75cb60d11c516647f8b33
R3 审批恢复 Command(resume=approve)
                          [LLM#3] msgs=5 tool_msgs=1
                          -> 中断 790dbb279e2f4c29258c4c022b5766d1   ← 第三次中断
                          -> 本次执行: ['delete:user']              ← 它真的执行了
```

展开 R3 的链路：

1. LangGraph 把旧中断 `01b9101e…` 消费掉，`_process_decision` 返回
   `(tool_call, None)` → **`delete_data` 工具被真正调用**（`EXEC` 里有记录）。
2. 工具结果作为 `ToolMessage` 回灌（`msgs=5 tool_msgs=1` 即证）。
3. **模型看到 ToolMessage 后，又发了一次 `delete_data`**
   （`qwen3.8-max-0902` 在此处不遵守"看到工具结果就给结论"的常规行为）；
   新的 tool_call 未在本次 `decisions` 覆盖范围内 →
   `after_model` 生成**第三个** `Interrupt`。
4. `resume_graph` 拿到 `result["__interrupt__"] = [1ba66082…]`，
   F3.1 判定"决策未生效" → 标 `failed` → **不推任何成功文案给前端**。

所以用户的体感是"审批通过了却什么都没发生"，而真实情况是
**写操作已经落在数据库里，只是没提交/没上报成功**（取决于事务边界）。

**为什么 `testuser99` 还在？** 因为本轮 `agent_delete_mode="logical"` 而
`agent_delete_flag_column="is_deleted"` **在 `user` 表不存在**
（真实列名是 `deleted`，已 SHOW COLUMNS 确认）→ `_column_exists` 返回 False
→ 回落物理删除。这条路径本来就会**先把 `fallback_reason` 记进结果**；
但由于 F3.1 提前中断，`_update_task_after_resume` 根本没跑到，
`affected rows` / `fallback_reason` 全部丢失。**列名配置错误（F3.4b）与
本轮中断（F3.5）叠加，才造成"删了等于没删"的表象。**

#### 修复方向（待确认后实施）

- **F3.5（核心）**：恢复轮禁止"再开新循环"。落地方式二选一：
  - **B1（推荐，改动小）**：`task_node` 恢复轮把内层 `invoke_input` 改为
    `Command(resume=…)` **并且**在调用前用 `aget_state` 确认内层
    `next` 命中 `HumanInTheLoopMiddleware.after_model`；
    若内层残留**多个**悬挂中断，先按中断 id 逐个投递（langgraph 1.2.10
    已支持 `{interrupt_id: value}` 映射，见 `pregel/_loop.py`
    `CONFIG_KEY_RESUME_MAP`）再进入正常循环。
  - **B2（更彻底）**：给内层 Agent 加 `recursion_limit` 收紧 +
    在工具执行后注入一条 `AIMessage`（或加一个 "no more tool calls"
    stop 条件），从机制上杜绝"批准后原地重发"。
- **F3.6（新）**：`filters` 参数做 JSON 字符串 → dict 的兼容解析
  （模型偶尔返回字符串）；`user_intent_quote` 同步校验。
- **F3.4b（新）**：`agent_delete_flag_column` 改为按表探测可用列
  （`deleted` / `is_deleted` / `delete_flag` 候选集），而不是写死一个默认值。
- **F3.7（新）**：写操作执行与"结果上报"解耦 —— 工具一旦执行成功，
  立刻独立落一条审计（含 `affected_rows`），不要因为后续模型行为
  把已发生的事实丢掉。

---

### 8.4 问题四：为什么 LLM 还是会回答"没删除权限"

**结论：不是提示词没生效，是模型（`qwen3.8-max-0902`）没有遵守提示词。**

先说清事实边界 —— **§七 F1.1/F1.2 的提示词改动是在位的**：

```
长度: 1205
含「务必正确理解」: True
含「不是权限不足」: True
含「不要预告」: True
```

而落库的 `ai` 消息（`sequence_number=2`，`14:45:38`）内容是：

> **删除用户 `testuser99`** 我这里无法直接操作数据库或执行删除。若你有数据库权限，
> 按知识库中的 `user` 表结构，建议优先使用**逻辑删除**而不是物理删除…

这段回答有**两个特征**：

1. 它把写操作**推荐为一条 SQL 让用户自己去执行**，而不是调工具 ——
   正是提示词第 3 条明令禁止的"预告我做不到"。
2. 它**同时**回答了第二个意图（包邮），说明模型在**一次生成里合并了
   knowledge + chat 两个分支**，写意图被降级成了"建议"。

**为什么模型不遵守？** 三个叠加因素：

- **模型档位**：`qwen3.8-max-0902` 与首轮 trace 里的模型不同
  （说明会话的 `model` 字段被改过，或者是新接入的档位）。
  提示词是按旧档位调过的，换档后指令跟随强度下降是常见现象。
- **知识库污染**：`search_knowledge` 召回了 5 篇文档，其中很可能包含
  `user` 表结构说明（回答里引用了"知识库中的 `user` 表结构"）。
  模型把「知识库给了 SQL 建议」误当成「我的职责是转达建议」。
- **双意图稀释**：一次请求里塞了写操作 + 知识问答，模型的安全偏好
  会优先满足后者，把前者让渡给用户。

**修复方向**：靠提示词继续加重语气收益递减。建议改为**机制约束**：
`TASK_AGENT_SYSTEM_PROMPT` 之外，在 `create_agent` 的 `middleware` 里加一个
"写意图检测 → 强制调用"的前置钩子（或至少在 `finalize_node` 做
"用户有明确写意图但本轮无写工具调用"的告警指标 + 兜底追问）。

---

## 九、第二轮实施记录（F3.4b / F3.5 / F3.6 / F3.7 已落地）

> 承接 §八 的四问诊断，本轮把结论落成代码。**只动后端**（前端既有
> `approval_completed` 处理已足够，`tsc -b` 复核通过、无需改动）。

### 9.1 改动清单

#### F3.5b — 恢复轮「冗余中断」就地消化（核心）

**文件**：`agent/nodes/task_node.py`

生产时序是：批准 → 写工具**真的执行** → 模型**原地重发** → 新中断 →
外层误判失败。修法是**在中断冒泡出内层之前就地纠偏**：

```python
# task_node：内层 ainvoke 返回后
if resume_payload is not None:
    result = await _absorb_reissued_interrupt(
        agent, invoke_config, result, resume_payload,
        _merge_tool_calls(executor, kb_executor),
    )
```

`_absorb_reissued_interrupt` 的**吞并条件（必须同时满足）**：

1. 恢复载荷是**已批准**的（`_is_approval_granted`：decisions 非空且全为 `approve`）；
2. 本轮**确有写操作执行成功**（`_successful_writes` 命中
   `tool` ∈ 写工具 ∧ `result_summary.ok=True` ∧ 含 `affected_rows`/`inserted_id`）。

任一不满足则**原样冒泡**，保留 F3.1 护栏效力：

| 情形 | 是否消化 | 理由 |
| --- | --- | --- |
| 批准 + 有写操作成功 | ✅ 消化 | 目标已达成，中断是模型的重复请求 |
| 决策是 reject / canceled | ❌ 保留 | 新中断是真信号，决策确实没生效 |
| 批准但本轮无写操作成功 | ❌ 保留 | 决策确实没生效，不能包装成成功 |

**反面教训（已写进代码注释）**：**不要**用
`agent.aupdate_state(cfg, None, as_node="HumanInTheLoopMiddleware.after_model")`
去"清"内层中断。实测它虽然能清掉 `interrupts`，但会把 `next` 置为 `('tools',)`
—— 相当于让**已批准的写操作再执行一遍**（重复删除级事故）。
冗余中断留在内层无害：它属于旧的内层 thread（带旧 `task_execution_id`），
下一轮任务会新建 thread，不会误命中。

#### F3.5a — 恢复轮按中断 id 定向投递

**文件**：`agent/nodes/task_node.py`（新增 `_inner_pending_interrupts` / `_build_resume_command`）

稳定 id 的副作用会让同一内层 thread 累积**多个 id 不同**的悬挂中断（§八）。
此时裸 `Command(resume=<决策>)` 会被 LangGraph 当作"下一个中断的答案"误消费。
故恢复前先 `aget_state` 对齐：

- 0 个悬挂 → 裸 resume 值（记录 `resume.no_hanging_interrupt`）
- 1 个悬挂 → 裸 resume 值（语义最清晰）
- ≥2 个悬挂 → `Command(resume={target_id: payload})` **只投第一个**
  （`resume.multi_interrupt`）；其余保留，不该被本次决策消费

#### F3.7 — 写操作执行与结果上报解耦

**文件**：`agent/nodes/task_node.py` + `agent/approval/approval_service.py`

1. `_update_task_record` 的 `status` 改为可选（`None` = 只更新审计、不动状态机）；
2. 新增 `_persist_write_audit`：写工具执行成功的当下就**独立落库**
   `tool_calls`，不受后续模型行为影响；
3. `GraphInterrupt` 分支在抛出前调用它 —— 这是"操作其实已经做过"的唯一凭据；
4. `approval_service` 遇到 `__interrupt__` 时**先查审计**再定性：

```python
executed = await _executed_writes_from_task(db, approval)
if executed:
    # 第二道兜底：即便 F3.5b 消化失败冒泡上来，也按「已执行」上报
    metrics.incr(K_APPROVAL_PREFIX.format("resume_reissued_but_executed"))
    ...  # 不标 failed，如实告知已执行了哪些写操作
    return
# 否则才是真的没生效 → 走 F3.1 的 failed 路径
```

#### F3.6 — `filters` / `values` 字符串容错

**文件**：`agent/tools/agent_tool.py`（新增 `_coerce_json_object`）

生产日志里 `filters` 是**字符串** `'{"username": "testuser99"}'`。
原实现 `isinstance(filters, dict)` 直接判否 → 抛错 → 多一次 LLM 往返自我纠错。
现在容忍：dict 原样 / JSON 字符串（含双重编码，最多解两层）/
空串与 `"{}"` / `None` → 空 dict。**无法解析才报错**（数组、非 JSON 串、数字均拒绝）。
`_validate_values` 同样走这条路径。

#### F3.4b — 删除标记列按表探测

**文件**：`agent/tools/agent_tool.py` + `core/config.py`

`agent_delete_flag_column` 默认值由 `"is_deleted"` 改为 **`""`（= 自动探测）**。
新增类常量与解析器：

```python
DELETE_FLAG_COLUMN_CANDIDATES = (
    "deleted", "is_deleted", "delete_flag", "del_flag", "is_del", "deleted_flag",
)
```

解析顺序：**显式配置优先（且须该列真实存在）→ 候选集探测 → 全部未命中才回落物理删除**
（记 `fallback_reason` + `delete.fallback_physical` 指标）。
删除了已无引用的 `_column_exists`。

### 9.2 实测验证

**F3.4b（真实库 `SHOW COLUMNS` 驱动）**

```
user                     -> 'deleted'      ← 修复前会误判为不存在
conversation             -> 'deleted'
conversation_message     -> ''             ← 无标记列，正确回落
```

**F3.6（纯函数）**

| 输入 | 结果 |
| --- | --- |
| `{"username": "testuser99"}` | ✅ 原样 |
| `'{"username": "testuser99"}'` | ✅ 解析 |
| `'"{\\"username\\": \\"testuser99\\"}"'` | ✅ 双重编码解析 |
| `"{}"` / `None` / `""` | ✅ 空 dict |
| `"[1,2,3]"` / `"not json"` / `123` | ✅ 正确拒绝 |

### 9.3 回归基线（`scripts/`，共 30 条断言全 PASS）

| 脚本 | 断言 | 覆盖 |
| --- | --- | --- |
| `smoke_approval_resume.py` | 4 | F3.0 决策透传（GOOD/BAD/FIX 三段） |
| `smoke_resume_multiround.py` | 5 | 事实 A/B/C（中断残留 / 原地重发 / 决策口径） |
| `smoke_resume_reissue.py` | 7 | **F3.5b 消化判据 + 边界安全**（新） |
| `smoke_final_integration.py` | 14 | **F3.4b 真实库探测 + F3.5b 判据 + F3.6 容错**（新） |

`smoke_resume_reissue.py` 的端到端部分**直接 `import` `task_node` 的真实实现**
（`_absorb_reissued_interrupt` / `_successful_writes` / `_is_approval_granted`），
而不是在测试里重写一遍逻辑 —— 避免"测试通过但生产没修"的假绿。

`smoke_final_integration.py` 走**生产同一条 `AsyncSessionLocal` 通道**连真库，
实测 `_resolve_delete_flag_column`：

| 表 | 探测结果 |
| --- | --- |
| `user` | `deleted` ✅（不是旧默认值 `is_deleted`） |
| `conversation` | `deleted` |
| `conversation_message` | `''`（无标记列，会回落物理删除） |

> 注意审计记录的字段口径是 `tool` / `params_summary` / `result_summary`
> （**不是** `name` / `args`）—— 写判据脚本时踩过一次，已修正。

### 9.4 校验状态

- `python -m compileall agent/ core/` ✅
- `tsc -b`（前端）✅ 无需改动
- **四个**冒烟脚本 exit code 均为 0 ✅

### 9.5 仍未做（诚实列出）

1. **F1「模型拒答」的机制约束**（§八问题四）—— 本轮未动。
   提示词已在位但模型不遵守，需另上"写意图检测 → 强制调用"或
   `finalize` 告警指标，属**独立改动**。
2. **§八 的四个验收用例需真实联调环境验证**：
   批准后写操作落库、`qwen3.8-max-0902` 原地重发被正确消化、
   逻辑删除真的写 `deleted=1`、审批结论文案送达前端。
   本地脚本只能覆盖到 Graph 层，覆盖不了真实模型行为。
3. **历史脏数据**：`task_execution.result` 曾被误填"用户清单"，未清理。
4. **`testuser99` 仍是 `deleted=0`** —— 需要重新跑一次真实审批流程验证。
