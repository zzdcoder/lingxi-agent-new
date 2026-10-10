# 项目长期记忆（灵犀 Agent）

## 架构约定

- **主图**：`intent_router → 条件路由（knowledge/chat/task_agent）→ merge → finalize`。
  merge 是 barrier。**§27 起多意图 `{task,kb}` 为串行接力**（knowledge 跑完再进 task_agent），
  不再并行 fan-out；回滚开关 `agent_multi_intent_serial=False`（见 F）。
- **StateGraph 只认识 schema 声明的 channel**：**未声明字段的写入被静默丢弃且不报错**。
  凡「返回了新 state 字段却没生效」，第一嫌疑是 `AgentState` 未声明（§27 实证）。
- **意图分层**：L0 规则门控 / L1 进程内缓存 / L2 向量就近 / L3 LLM 结构化输出。
  LLM 是兜底非必经，任何失败**降级 chat**，不得击穿主链路。
- **任务 Agent 是嵌套子 Agent**（`agent/nodes/task_node.py`，约 1760 行，改前必重读），
  内层有**独立 thread + checkpointer**（`_inner_thread_id`）。
- **状态跨轮隔离**：`executed_branches` 由入口节点每轮覆盖写入；merge 按它判断「本轮跑过
  哪些分支」，不按「字段是否非空」（检查点会累积残留）。
- **韧性原则**：优化项失败必须降级，不得击穿主链路。

## 项目约定

- 文档用 `§` 编号章节，代码注释引用编号（§20/§21/§22…）。
- 每个修复都要：根因 + 影响范围 + 回归风险 + 验收用例。
- 冒烟脚本 `scripts/smoke_*.py`（断言判定、可重复、离线），诊断 `scripts/diag_*.py`。
- **审计字段口径是 `tool`/`params_summary`/`result_summary`**，不是 `name`/`args`。
- 观测指标走 `agent/observability.py` 的 `metrics`，键 `agent.*`/`task.*`/`cache.*`。
- 断言写**不变量**，绝不写「某源码串出现 N 次」（已踩两次：C2/C8）。
- 回归脚本必须 import 生产实现，不能重写逻辑（否则假绿）。
- 配置默认值凡涉及「列名/表名」必须对着真实 schema 核（C4）。
- Windows + 中文长文本**禁止 heredoc 追加**，用 Write 工具或 Python
  `write_text(..., encoding='utf-8')`（C6，踩过三次）。

---

## A. HITL 中断与恢复（§20/§22/§24）

- **A1 恢复必须 `Command(resume)`，不跨嵌套层透传**：外层 resume 不会自动给内层。
  恢复轮把决策经 `configurable.approval_resume`/`clarify_resume` 下发给 `task_node`，
  由它转 `Command(resume=...)`。**不要**摘掉内层 checkpointer（那是 §19 的修复）。
- **A2 任何 resume 后必须判 `result.get("__interrupt__")`**，命中即失败+告警。
  否则「静默假成功」（LangGraph 用返回值表达中断，ainvoke 正常返回）。
- **A3 批准后模型原地重发** → 新 tool_call 不在 decisions 内 → 第三个中断 →
  F3.1 判失败。**写其实落库了**。区分「工具没跑」vs「跑了被误判」。
- **A4 稳定 id 副作用**：同一内层 thread 可残留多个悬挂中断。恢复前 `aget_state`
  看 `next`/`tasks[].interrupts`，按 interrupt id 逐个投递。
- **A5 decisions 数量对齐「写工具数」**，不是全部 tool_calls（HITL 只统计
  `interrupt_on` 配置过的工具），否则 ValueError。
- **A6 冗余中断就地消化，不要清状态**：禁止 `aupdate_state(None, as_node=...after_model)`
  （会把 `next` 置 `('tools',)` → 已批准写操作再执行一遍，事故级）。正确：`task_node`
  内层返回后，若「载荷已批准」∧「本轮确有写操作成功」→ 抹掉 `result["__interrupt__"]`。
  吞并条件必须严格（reject/canceled、或批准但无写成功 → 必须保留给 F3.1）。
  **§24 修正**：该法只管同一轮内，跨轮有害（见 A10）。
- **A7 写操作执行与结果上报解耦**：写成功当下就独立落审计（`_persist_write_audit`）。
- **A8 判据区分调用 vs 成功**：`_successful_writes` = 写工具 ∧ `ok=True` ∧
  含 `affected_rows`/`inserted_id`。
- **A9 两类 `interrupt()` 共存，载荷必须与中断类型配对**：`ask_user` 的
  `interrupt({"type":"clarification"})` 要 `{answers}`/`{canceled}`；HITL 的
  `interrupt(hitl_request)` 要 `{decisions}`。`human_in_the_loop.py:435` 是
  `interrupt(...)["decisions"]` **下标无兜底** → 投错必 KeyError。
  投递**前一刻**重新读（中断类型会随轮次变化）。判据不足归「未知」并放行。
  附带：同一条 AIMessage 同时有 ask_user 与写操作时，HITL 的 after_model **先**抛，
  ask_user 根本不执行 —— 「边问边删」时审批抢占追问。
- **A10 入口必须按「内层检查点是否仍有悬挂中断」判活**（§24 根因）：
  僵尸 `running` 记录 → `_find_running_task` 按会话复用 `task_execution_id`
  → 复用内层 thread → 上一轮悬挂中断被带出 → **重复审批卡**。
  修法：`_find_running_task(..., reject_stale=True)`（恢复轮必须复用，
  `reject_stale = not is_resume_round`）+ `_close_stale_running_tasks`。
  判据失败一律**保守放行**（拒收代价 >> 放行代价）。
  ⚠️ `aget_tuple` 返回 `CheckpointTuple`：**顶层没有** `next`/`tasks`/`interrupts`；
  `next` 在 `cpt.checkpoint["next"]`，待写在 `cpt.pending_writes`。
  停在 interrupt 时 `pending_writes=True, next=None`；跑到 END 两者皆空。
  指标：`agent.task.stale_running_rejected` / `stale_running_closed`。
- **A11 ⚠️ 内层 Agent 的 config 必须过滤 LangGraph 内部键**（2026-09-23 根因）：
  `_build_subgraph_config` 的 `dict(config)` **照抄**了主图节点的
  `configurable`，其中含 LangGraph 注入的 `__pregel_scratchpad` / `__pregel_send` /
  `__pregel_call` / `__pregel_task_id` / `__pregel_read` / `__pregel_replay_state` /
  `__pregel_runtime` / `__pregel_checkpointer` / `checkpoint_id` / `checkpoint_map` /
  `checkpoint_ns`。后果：
  ① 内层 `interrupt()` 用**外层 scratchpad** ⇒ `GraphInterrupt` **直接抛出**而非
  转成 `__interrupt__`；② 恢复轮 resume 投递错位 ⇒ 从 **model 节点**恢复
  （**重跑 LLM**、模型**重新规划**出参数不同的新 tool_call）而不是从
  `after_model` 节点恢复执行已批准的旧 tool_call ⇒ **写操作从未执行**；
  ③ `_absorb_reissued_interrupt` 只处理返回值形态 ⇒ 在异常路径下**永不执行**。
  表现：审批通过后数据没删、任务判 failed（**判定本身正确**，无假成功）。
  修法：剔除 `__pregel*` 前缀 + `checkpoint_*`，只保留业务键
  （db / sse_queue / trace_id）。铁证脚本
  `scripts/diag_nested_interrupt_form.py`（同样嵌套调用，过滤前后写操作
  「未执行 → 执行」）。
  附带结构事实：`HumanInTheLoopMiddleware.after_model` 是**独立节点**
  （`langchain/agents/factory.py:1543`），正常恢复**不重跑 model**、
  直接执行旧 tool_call —— 所以「恢复后参数变了」= config 污染的信号。

## B. 流式与思考过程（§21/§23）

- **B1 思考两个断点**：① 请求侧 `extra_body={"enable_thinking": True}`
  （DashScope Qwen3 默认不返回）；② `langchain_openai` 静默丢 `reasoning_content`
  → 覆盖 `_convert_chunk_to_generation_chunk` 捞回（`core/llm.py`）。
  非思考模型传该参数会报错 → **前缀白名单**。
  `<think>` 过滤必须**有状态**（`StreamingThinkScrubber`）：开标签与内容会拆两个 delta。
- **B2 思考型模型 `max_tokens` 要算 reasoning 预算**，否则 `finish_reason=length`
  → 正文零 token → `LengthFinishReasonError`。分类/摘要直接关思考。
- **B3 `stream_mode="messages"` 会吐 ToolMessage**：必须按
  `meta["langgraph_node"] == "model"` 过滤（节点名 `"model"`/`"tools"`）。
- **B4 `GraphInterrupt` 绝不走降级重跑**（会导致已批准写操作执行两遍，事故级）：
  `except (GraphInterrupt, asyncio.TimeoutError): raise`。
- **B5 回落只对「环境类故障」有效**：输入类确定性错误（载荷不匹配/参数非法）
  不得回落，直接转可读错误（`ResumeMismatchError`）。
- **B5b 校验点位置 = 校验逻辑**：图内部新产生的中断不经过入口校验。校验必须下沉到
  每次投递/异常出口；`astream` **外层**包 try 把 `KeyError('decisions')` 转译
  （异常在迭代中抛出，async for 内层接不住）。**只认 `'decisions'` 键名**
  （用 `exc.args[0]`，`str(KeyError)` 带引号）。用 `except BaseException` 防继承链变化。
  诊断「修复没生效」先看覆盖面，再怀疑部署。
- **B6 推送函数返回值不能吞**：`put_*` 必须 `-> bool` 并显式 return（上层当判据用）。
- **B7 SSE 帧必须带 `branch`**（knowledge/chat/task/final），空串不写字段兼容旧前端。
  协议：`content`/`status`/`ping`/`done` + `thinking`/`tool`。
- **B8 前端「内容消失」查渲染条件**：卡片与正文是**并列**不是互斥
  （`{!hasApproval && ...}` 短路是根因）。同模式：`isThinking` 在吐正文时置 false
  → 改 `hasThinking = Boolean(thinkingContent?.trim())`。

## C. 通用陷阱

- **C1 提示词治不住就上机制**：提示词加重语气收益递减 → 上「写意图检测→强制调用」
  或打告警指标（`task.clarify_answer_no_write`）。
- **C2/C8 断言写不变量**，改造后主动扫既有断言。
- **C3 回归 import 生产实现**。
- **C4 默认值核对真实 schema**：`agent_delete_flag_column` 默认 `is_deleted` 而真列是
  `deleted` → 逻辑删除静默回落物理删除。已改默认空串 + 候选集探测
  （`deleted`→`is_deleted`→`delete_flag`→`del_flag`→`is_del`→`deleted_flag`）。
  `conversation_message` 无标记列。
- **C5 先分清「哪条 UPDATE」**：`task_execution`/`tool_registry`/审计表 = 框架记账，
  业务表才算用户期待的写。
- **C7 跨分支功能不对称是隐性缺陷高发区**：`chat_node` 有
  `MySQLChatMessageHistory` 历史、`task_node` 曾没有（§23 修复 2）。修法
  `_load_task_history(db, conversation_id)`，三条边界：① 末条若是 human 必须裁
  （恢复轮重入时末条即本轮输入，且本轮用稳定 id、历史无 id → `add_messages`
  去重不生效）；② 历史消息不带 id；③ 条数上限 `agent_task_history_recent_num`=10。
  开关 `agent_task_history_inject`。失败一律返回空列表 + 告警（上下文是增强）。

## E. 任务迭代预算与循环防护（§25）

溯源：trace=0e7fc2dd93554f24「统计某用户有多少条消息」→ 24 次 LLM / 136s / 未解决。
四条根因：**工具能力不可达**（根因）> 无迭代预算 > 失败不可见+文案喂空转 > 提示词无能力边界。

- **E1 `recursion_limit` ≠ LLM 调用预算**。前者计**超步**（model=1、tools=1），
  且每次 invoke 用 checkpoint 的 step **重发一整份预算**（`pregel/_loop.py:1700`）
  → **不是**任务生命周期上限。`ModelCallLimitMiddleware` 只数模型节点，才是真闸门。
- **E2 预算中间件必须 `exit_behavior="error"`**。`"end"` 注入**英文** AIMessage，
  而 `_extract_answer` 取 `messages[-1].content` → 英文直出中文界面。
- **E3 thread_limit 主、run_limit 辅**：run 计数是 `UntrackedValue`，每次 invoke 重置
  （审批恢复即清零），**不能单独依赖**。
- **E4 中间件链不能挂在 `if approval_mode:` 里** —— 那等于「非审批模式完全没有预算」，
  而多数任务是非审批模式。顺序：ModelCallLimit → HITL → RepeatGuard
  （守卫在后：重复发起写操作时应先走审批，别被掐掉）。
- **E5 审批模式也要墙钟上限**。`interrupt()` 立刻抛出、协程随即返回，
  **人类等待发生在两次 invoke 之间，不占预算** —— 旧注释「审批等待久所以不超时」是错推理。
- **E6 `ToolCallLimitMiddleware` 只计次数不比参数**，治不住「同参空转」→ 自研
  `agent/task_guards.py::RepeatToolCallGuardMiddleware`（参数排序后做指纹，累计 ≥3 抛
  `RepeatedToolCallError`）。预算/守卫类异常都要进 `_run_task_agent` 的**不回落白名单**
  （计数在检查点，重跑必撞同一闸门）。
- **E7 被拒调用必须留痕**：`_record_call` 只在成功路径 → 白名单拒绝/参数非法完全隐形
  （24 次调用只有 4 条痕迹）。`@_audited` 装饰器统一包 6 个工具方法。
- **E8 错误回灌文案要区分可重试/禁止重试**。对所有错误都回「请修正参数后重试」
  等于**喂养空转** —— 确定性拒绝要明确说「禁止同参重试」并给替代动作。
- **E9 熔断器的统计口径与熔断口径必须分开**：任何失败都计 total/fail（否则成功率
  虚高 100%），但**只有系统级故障**计 consecutive_failures（否则误熔断）。
  另：成功必须清 `last_error`（否则看板长期残留旧错误）。
- **E10 连接池「non-checked-in connection」告警**：根因是 `asyncio.wait_for`
  **取消在途 DB 操作**（工具/语句/任务三层），取消时来不及 checkin。
  **未根治**（根治要把超时下沉到服务端 `MAX_EXECUTION_TIME`）。
- **E12 ⚠️ 千万别给 aiomysql 开 `pool_pre_ping`**（2026-09-23 我亲手引发的事故）：
  开启后登录正常但会话接口**全 400** ——
  `AsyncAdapt_aiomysql_connection.ping() missing 1 required positional argument: 'reconnect'`。
  根因：SQLAlchemy 用 **pymysql** 的 `Connection.ping` 签名探测
  `_send_false_to_ping`，pymysql≥2.2 改成 `ping(self, reconnect=False)` → 探测
  返回 False → `do_ping` 调 `ping()` 不传参，而 aiomysql 适配器的
  `ping(self, reconnect)` **无默认值** → TypeError。
  **只有复用连接才 pre_ping** → 表现为「第一条请求成功、后续全挂」。
  现状：默认关闭；`_resolve_pool_pre_ping()` 在「开了 pre_ping 但没开 shim」时
  强制关闭并打 ERROR。要真开需同时 `DB_POOL_PRE_PING_SHIM=true`。
  回归：`scripts/smoke_db_pool_pre_ping.py`。
  **教训**：连接池这类参数改变的是**每次取连接**的高频路径，
  不能照搬最佳实践，必须在目标依赖组合下实测。
- **E11 提示词要定义「能力边界」**：只说「缺信息就追问」不够，模型会把追问埋进正文。
  必须列出做不到的事（JOIN/LIKE/时间范围/TopN）+ 规定动作顺序。

## D. 前端（D:\lingxi-agent-portal）

- React 19 + TS + Vite + Tailwind；SSE 用 `fetch` + `ReadableStream`（**非 EventSource**）。
- 无测试框架，纯逻辑校验 `scripts/verify-*.mjs` 用 node 跑。
- 类型检查 `./node_modules/.bin/tsc -b`，构建 `./node_modules/.bin/vite build`。
- **既有 lint 报错（非本轮引入）**：`App.tsx:46` setState-in-effect、`llm.ts:39` explicit-any。
- 审批/追问卡片：实时路径挂同一 assistant 消息，刷新重建走 `reconcile*Messages`
  独立消息 —— 改渲染两边都要考虑。

---

## F. 多意图串行化 × merge 拼接 × 上游 context 注入（§27，2026-09-29）

溯源：`intents == {task, knowledge_base}` 并行 fan-out 的三重代价 ——
① 知识分支与任务 Agent 工具**重复检索**；② merge 调 LLM 合并两段（实测 ~120s）；
③ **并行审批中断会取消知识分支**，恢复后需 `_recover_knowledge_branch` 补偿重跑。

### 四个改造
- **① 多意图串行**：`route_by_intents` 返回**单个** `NODE_KNOWLEDGE`（回滚时返回列表）；
  `knowledge` 出边由 `add_edge` 改 `add_conditional_edges(NODE_KNOWLEDGE, route_after_knowledge, kb_map)`。
  **`route_after_knowledge` 判据只看意图与节点可用性，不看知识分支成败**
  —— 知识检索失败也**必须**进任务分支自愈，否则任务被静默吞掉。
- **② merge 拼接省 LLM**：`_concat_answers` 用 `【知识依据】/【执行结果】` 两段拼接，
  默认 `agent_merge_llm_enabled=False`。**拼接前必须先修 finalize 补推**，否则重复推送。
- **③ 上游 context 注入**：`task_node` 把 `state["context_text"]` 限长（默认 3000 字，
  超限截断 + 附「已截断」提示，允许模型补检索）拼进任务 Agent 的 system prompt。
  **依赖「`create_agent` 的 `system_prompt` 是闭包变量、不进 state」** ——
  每次 model 节点执行才临时前置，故不污染内层 checkpoint、恢复轮不会重复 SystemMessage。
  提示词同步加「优先复用【已检索的知识依据】…不要调 `search_knowledge` 重复检索」。
- **④ 修 finalize 重复推送**：`_push_final_response(config, content, state=None)` 真正使用
  `pushed_contents` 去重；`final_push_suppressed` 控制 finalize 是否补推。
  **注意：抑制标记不作用于兜底文案**（否则异常时用户看到空）。

### ⚠️ `pushed_contents` 三重失效（§20 F2.4 从未生效）
① 未在 `AgentState` 声明 → 静默丢弃；② `_already_pushed_branches` 无调用点；
③ knowledge/chat 不写该字段。
**修法：由 `merge_node` 单点覆盖写入**（merge 是 barrier：避免并行写 reducer 冲突、
天然按本轮产出判定、每轮覆盖不留残留、分支节点零改动）。

### 配置
`agent_multi_intent_serial`(True) / `agent_merge_llm_enabled`(False) /
`agent_task_kb_context_inject`(True) / `agent_task_kb_context_max_chars`(3000)。

### 验收
新增 `scripts/smoke_multi_intent_serial.py`（39 项，`_LLMProbe` 被实例化即 raise
→ 证明拼接模式**零 LLM 构造**）；`smoke_cross_turn_isolation.py` 第 3 节原断言
「并行双分支触发 LLM 合并」是**改造前行为**，已改为不变量断言 + 回滚路径覆盖。
全量 22 个脚本 exit=0。
⚠️ 跑脚本用 **`.venv/Scripts/python.exe`**（受管 Python 无 langchain_core）。
