# 项目长期记忆（灵犀 Agent）

## 架构约定

- **主图**：`intent_router → 条件路由（knowledge / chat / task_agent）→ merge → finalize`。
  多分支是**并行 fan-out**，merge 是 barrier（等所有前驱完成）。
- **意图分层判定**：L0 规则门控 / L1 进程内缓存 / L2 向量就近 / L3 LLM 结构化输出。
  LLM 是兜底而非必经路径，任何失败都必须**降级 chat**，不得击穿主链路。
- **任务 Agent 是嵌套子 Agent**（`agent/nodes/task_node.py`），
  内层有**独立 thread 与 checkpointer**（`_inner_thread_id`）。
- **状态跨轮隔离**：`executed_branches` 由入口节点每轮覆盖写入，merge 按它判断
  「本轮跑过哪些分支」，不按「字段是否非空」（检查点会累积历史残留）。
- **韧性原则**：优化项失败必须降级，不得击穿主链路（意图识别、SSE 投递、熔断计数均如此）。

## 项目约定

- 文档用 `§` 编号章节，代码注释引用编号（如 `§20`、`§21`、`§22`）。
- 每个修复都要求：根因 + 影响范围 + 回归风险 + 验收用例。
- 冒烟脚本统一 `scripts/smoke_*.py`，断言判定、可重复运行；诊断脚本 `scripts/diag_*.py`。
- **审计记录字段口径是 `tool` / `params_summary` / `result_summary`**，
  不是 `name` / `args`（踩过：按 `name` 取 → 命中 0 条，误判「判据失效」）。
- 观测指标走 `agent/observability.py` 的 `metrics`，键名 `agent.*` / `task.*` / `cache.*`。
  任务侧：`task.resume.multi_interrupt`、`task.resume.reissue_absorbed`、
  `task.resume.no_hanging_interrupt`、`task.resume.payload_mismatch`、
  `task.resume.secondary_interrupt`、`task.clarify_answer_no_write`、
  `task.history_injected`、`task.history_load_failed`、`task.write_audit.persisted`、
  `task.stale_running_rejected`、`task.stale_running_closed`。

---

## A. HITL 中断与恢复（§20 / §22）

### A1. 恢复必须用 `Command(resume)`，且不跨嵌套层级自动透传

外层主图的 `Command(resume=...)` **不会**自动传给内层子 Agent。内层若挂了自己的
checkpointer 又收到「全新 input」，待审批中断被**静默丢弃**、内层从头重跑、模型
再次发起同一写操作、**再次中断**，写操作从未执行而外层误报「恢复完成」。

- 规则：恢复轮把决策经 `configurable.approval_resume` / `clarify_resume` 下发给
  `task_node`，由它转 `Command(resume=...)` 交内层 Agent。
- **不要**改成「内层去掉 checkpointer」—— 内层 thread 隔离是 §19 的修复手段。
- 复现：`scripts/smoke_approval_resume.py`。

### A2. 恢复后必须判 `result.get("__interrupt__")`，否则「静默假成功」

LangGraph 用**返回值**表达中断，`ainvoke` / `astream` 都正常返回。若决策没送到真正
中断的那层，图会**再次中断**，而外层顺着往下走、照常打「恢复完成」日志并计数成功。

- 规则：任何 resume 调用后都要判 `result.get("__interrupt__")`，命中即标记失败 + 告警。

### A3. ⚠️ F3.1 护栏的「反向误判」：批准后模型原地重发

`Command(resume=...)` 修好之后仍报「审批通过却没执行」的真正原因。时序：
① 决策消费旧中断 → **写工具确实被调用**；② ToolMessage 回灌后模型**又发一次同一写操作**；
③ 新 tool_call 不在本次 decisions 覆盖内 → `after_model` 抛**第三个中断**；
④ F3.1 检测命中 → `status=failed`、不推成功文案。

⇒ **写操作落库了，用户看到全是失败。** 别把「审批后没执行」简写成「写操作从未执行」，
要先区分「工具没跑」还是「跑了但被误判」。（复现：`smoke_resume_multiround.py`）

### A4. 稳定 id 的副作用：同一内层 thread 残留多个悬挂中断

`task_node` 用 `id=f"task-input-{task_execution_id}"` 构造首轮 HumanMessage（稳定 id
防重复 append）。但 `_find_running_task` 复用同一条 running 记录 → 同一
`_inner_thread_id` → 后续任何「new input」都 **upsert 覆盖**那条消息，而内层
`last_ai_msg.tool_calls` 仍带上一轮未消费的写操作 → `after_model` **再次 interrupt**。

- 规则：恢复前先用 `aget_state` 看内层 `next` / `tasks[].interrupts`；有多个悬挂中断时
  按中断 id 逐个投递（langgraph 1.2.10 支持 `Command(resume={interrupt_id: value})`，
  见 `pregel/_loop.py` 的 `CONFIG_KEY_RESUME_MAP`）。

### A5. decisions 数量对齐的是「写工具数」，不是「全部 tool_calls」

`HumanInTheLoopMiddleware.after_model` 的 `interrupt_indices` **只统计配置了
`interrupt_on` 的工具**。一写一读混在同一条 AI 消息里时，把 `action_count` 取成全部
tool_calls 数会直接抛 `ValueError: Number of human decisions (N) does not match ...`。

### A6. 冗余中断不要清理，要在内层就地消化

批准后模型原地重发写操作会拖出一个**冗余中断**。

- **不要**用 `aupdate_state(cfg, None, as_node="...after_model")` 去清它 —— 实测能把
  `next` 置为 `('tools',)`，等于让**已批准的写操作再执行一遍**（事故级）。
- 正确做法：`task_node` 内层返回后，若「载荷已批准」**且**「本轮确有写操作成功」
  同时成立，就地抹掉 `result["__interrupt__"]`（`_absorb_reissued_interrupt`）。
- **吞并条件必须严格**：reject/canceled 的新中断是真信号；批准但无写操作成功说明决策
  确实没生效 —— 两种都**必须保留**给 F3.1 护栏。
- ⚠️ **§24 修正：原先「冗余中断留在内层无害」的假设是错的。**
  该假设依赖「下轮会新建 thread」，但 `_find_running_task` 是**按会话**查最近一条
  `running` 记录 —— 上轮若未把状态同步成 `completed`（僵尸记录），下轮就**复用**
  同一个 `task_execution_id` → 同一个内层 thread → 上一轮的悬挂中断被带出来
  → **用户看到重复审批卡片**（明明已批准执行过）。
  所以「返回值层面消化」只解决**同一轮内**的问题，**跨轮有害**。

### A7. 写操作执行必须与「结果上报」解耦

模型在工具执行后的行为不可控（原地重发、超时、乱答）。**写工具执行成功的当下就独立
落审计**（`_persist_write_audit`），否则后续被误判失败时 `affected_rows` / `mode` /
`fallback_reason` 全丢。`approval_service` 判定失败前**必须先查这条审计**。

### A8. 判据函数要严格区分「调用」与「成功」

`_successful_writes` 三重条件：工具名 ∈ 写工具 ∧ `result_summary.ok=True` ∧
含 `affected_rows`/`inserted_id`。只看工具名会把失败调用算进去，只看 `ok` 会把只读
工具算进去。

### A9. 两类 `interrupt()` 共存时，恢复载荷必须与中断类型配对（§22 根因）

内层任务 Agent 上同时存在：① `ask_user` 工具内部 `interrupt({"type":"clarification"})`
（期望 `{answers}` / `{canceled}`）；② `HumanInTheLoopMiddleware.after_model` 的
`interrupt(hitl_request)`（期望 `{decisions}`）。

`langchain/agents/middleware/human_in_the_loop.py:435` 是**下标取值无兜底**：
`decisions = interrupt(hitl_request)["decisions"]` → 投 `{answers}` 即 `KeyError`。

- 规则：投递 resume 前用 `aget_state` 读悬挂中断的 **value 形态**并与载荷配对
  （`answers`/`canceled`→clarify，`decisions`→HITL），不匹配就**不下发**并给可读提示
  （`ResumeMismatchError`，见 `_match_resume_target`）。投递必炸，不是概率问题。
- **悬挂中断的类型会随轮次变化**：clarify 被消费后模型发出写操作 → 变成 HITL。
  所以载荷不能按「本轮开头是什么中断」假定，必须在**投递前一刻**重新读。
- **判据不足 ≠ 真实冲突**：中断 `value` 取不到时应归为「未知」并**放行**，不要拒绝。
- 附带发现：同一条 AIMessage 里同时有 `ask_user` 与写操作时，HITL 的 `after_model`
  **先**抛中断（在 model 之后、tools 之前），`ask_user` 的 `interrupt()` 根本不执行 ——
  即「边问边删」时审批抢占追问。

### A10. 恢复/复用的入口必须按「内层检查点是否仍有悬挂中断」判活（§24 根因）

**症状**：第一轮「删除 dada 信息」审批通过并执行成功；第二轮「同时删除会话信息」
**弹出两张审批卡**（一张是上轮的删除 dada）。用户视角是「历史任务复活了」。

**根因链**（比 A4/A6 更深一层，务必记住）：
1. 上一轮结束**未把 `TaskExecution.status` 同步为 `completed`** → 留下僵尸
   `running` 记录；
2. `_find_running_task` **按会话**取最近一条 `running` 记录 → **复用**
   `task_execution_id`；
3. 复用即复用 `_inner_thread_id` → **同一个内层 thread**；
4. 内层 checkpointer 里上一轮的悬挂中断**还在** → 恢复时被读出来 → 重复弹卡。

⇒ A6 的「返回值层面消化冗余中断」只覆盖**同一轮内**；**跨轮必须靠入口拒收**。

- 规则：`_find_running_task(..., reject_stale=True)` —— 复用前先判该记录的内层
  检查点**是否已无悬挂中断**（`_is_stale_running_record`），陈旧则拒绝复用并新建
  `task_execution_id`；顺带 `_close_stale_running_tasks` 把僵尸记录收尾为 `completed`。
- **恢复轮必须复用**（`reject_stale = not is_resume_round`）：恢复轮要的就是那条
  带悬挂中断的 thread，拒收会把真实待审批弄丢。
- **判据失败一律「保守放行」**：拒收的代价是真实审批失效（用户要重做写操作），
  放行的代价只是偶发串味 —— 两者不对称，**读不到检查点时放行**。
- ⚠️ **实现陷阱：`aget_tuple` 返回 `CheckpointTuple`，字段位置与 `StateSnapshot`
  **不同**。`CheckpointTuple` **顶层没有** `next` / `tasks` / `interrupts`：
  - `next` 在 `cpt.checkpoint["next"]`；待写在 `cpt.pending_writes`
  - 实测：停在 interrupt 时 `pending_writes=True, next=None`；跑到 END 时两者皆空
  按 `StateSnapshot` 形态写（`state.tasks[].interrupts`）会直接 `AttributeError`。
- 指标：`agent.task.stale_running_rejected` / `agent.task.stale_running_closed`。

---

## B. 流式输出与思考过程（§21）

### B1. 思考过程的两个断点

1. 请求侧必须下发 `extra_body={"enable_thinking": True}`（DashScope Qwen3 默认不返回）；
2. `langchain_openai` 的 `_convert_delta_to_message_chunk` 只读 content/function_call/
   tool_calls，**静默丢弃 `reasoning_content`**。但 openai SDK `BaseModel` 是
   `extra="allow"`（`openai/_models.py:128`），字段仍在
   `chunk["choices"][0]["delta"]` 里 → 覆盖 `_convert_chunk_to_generation_chunk`
   即可捞回（见 `core/llm.py`）。
- 非思考模型传 `enable_thinking` 可能报错 → 走**前缀白名单**。
- 逐 delta 过滤 `<think>` 必须**有状态**（`StreamingThinkScrubber`）：模型会把 `<think>`
  与内容拆两个 delta 发，逐帧 `re.sub` 匹配不上 → 开标签原样返回 → **思考内容被当正文推**。

### B2. qwen 思考型模型的 max_tokens 必须算上 reasoning 预算

思考型模型**先消耗 reasoning_tokens 再产正文**。给小 `max_tokens` 会被思考吃满 →
`finish_reason=length` → 正文零 token → 抛 `LengthFinishReasonError`。分类/摘要类
任务直接 `extra_body={"enable_thinking": False}`。

### B3. `stream_mode="messages"` 会吐 ToolMessage —— 必须按节点名过滤

`astream(..., stream_mode=["messages","updates"])` 的 `messages` 产出 `(chunk, meta)`，
**所有**节点消息都抖出来，包括 `tools` 节点的 `ToolMessage`（content 是数据库原始结果）。

- 规则：`meta["langgraph_node"] == NODE_MODEL`（`"model"`）才推，配类名排除兜底。
- `create_agent` 节点名字面量：`"model"` / `"tools"`（`factory.py:1476/1480`）。
- `updates` 模式用于拿工具 end 事件与 `__interrupt__`。

### B4. 流式化改造中 `GraphInterrupt` 绝不能走「降级重跑」分支

一旦降级重跑到 `ainvoke`，模型会**再发一次同一写操作** → 已批准的删除/更新**执行两遍**
（事故级）。规则：`except (GraphInterrupt, asyncio.TimeoutError): raise`。
（非 GraphInterrupt 的异常走回落是安全的 —— 回落用同一份 `invoke_input`。）

### B5. 「回落重跑」只对环境类故障有效，对确定性错误必须排除

「除了 Exception 就回落 ainvoke」的前提是「非流式能跑通」。但**恢复载荷与中断类型不匹配**
是确定性错误 —— 回落用**同一个** `Command(resume=...)` 重投只会**再炸一次**。

- 规则：回落白名单按「故障是否与环境相关」划分。环境类（版本不支持 stream_mode、
  图结构变化）→ 可回落；输入类（载荷格式/类型不匹配、参数非法）→ **不得回落**，
  直接转可读错误（见 §22 修复2，`except (..., ResumeMismatchError): raise`）。

### B5b. ⚠️ 校验点的**位置**与校验逻辑同等重要（§23 修复 1）

§22 修复 1 在 `_build_resume_command` 里做载荷/中断配对校验，但它**只在节点进入时
执行一次**，且在 `_run_task_agent` **之前**。于是：

1. 进入时悬挂的是 clarify，与载荷 `{answers}` 配对 → **通过（这是对的）**；
2. 模型消费该中断后**自己发出写操作** → HITL 中间件在 `astream` **内部**抛**新中断**
   （期望 `{decisions}`），而 `resume_payload` 仍是常量 `{answers}` →
   `human_in_the_loop.py:435` 下标取值 → `KeyError('decisions')`；
3. 第二次投递**完全在图内部产生**，不经过校验函数 —— 校验彻底失位。

- 规则：**凡是「图内部可能新产生同类事件」的场景，校验必须下沉到每次投递/异常出口**，
  不能只在入口做一次。
- 修法：`astream` **外层**包 try，把 `KeyError('decisions')` 转译成
  `ResumeMismatchError`（走不回落路径）。转译点**必须在 `astream` 之外** ——
  异常是在**迭代过程中**抛出的，`async for` 内层接不住。
- **只认 `'decisions'` 这一个键名**（`exc.args[0]`，注意 `str(KeyError)` 带引号不可比）。
  泛化捕获 KeyError 会把检查点结构变化、字段重命名这类**真实缺陷**静默转成
  「载荷不匹配」的可读提示 —— 让确定性 bug 变成看不懂的降级，是更坏的结果。
- 用 `except BaseException` 而非 `Exception`：`GraphInterrupt` 当前继承 `Exception`，
  但用 BaseException 防未来继承链变化时漏接中断（转译函数对非 KeyError 返回 None，不误吞）。
- 诊断「修复为何没生效」时，**先看修复的覆盖面，再怀疑部署** ——
  本轮的 KeyError 与修复 1 的生效日志出现在**同一次节点执行内**。

### B6. 推送函数的返回值是「上层判据的输入」，不能吞

`put_content` / `put_thinking` 曾写成 `-> None`（内部布尔被丢弃）→ `StreamEmitter` 里
`if ok and fingerprint` 恒假 → 去重集合永远为空；`delivered_chars` 恒 0。

- 规则：`put_*` 必须 `-> bool` 并显式 return。凡「返回值被上层当判据用」的函数，
  冒烟里都要有一条直查返回值的断言（这类缺陷只有断言能发现）。

### B7. SSE 帧缺少分支标识，多分支内容会挤同一个气泡

`knowledge_node` 逐 token 推、`task_node` 整段推，共用同一队列且帧内无 branch。

- 规则：`put_content` 必须携带 `branch`（`knowledge` / `chat` / `task` / `final`），
  空串时不写该字段以兼容旧前端。前端按 branch 累积为 `segments` 分段渲染；
  `message.content` 仍是各段拼接（它是发给后端的历史消息载体，必须与界面一致，
  否则多轮对话丢上下文）。SSE 协议：`content` / `status` / `ping` / `done` +
  `thinking` / `tool`。

### B8. 前端「内容消失」要从渲染条件本身查，别停在数据层

现象：模型先给出的回答从界面消失，刷新才恢复；数据层完全正常。

- 根因：`MessageBubble` 渲染短路 `{!hasApproval && (...)}` —— 卡片存在即让整个正气泡
  条件为假。**卡片与正文是并列关系，不是互斥关系。** 诊断必须读到 JSX 渲染条件。
- 同一模式复发：思考面板 `{isThinking && <ThinkingIndicator/>}`，而 `isThinking` 在
  「开始吐正文」时被置 false → 改成 `hasThinking = Boolean(thinkingContent?.trim())`。

---

## C. 通用工程陷阱

### C1. 提示词治不住模型时，该上机制约束

提示词已明确「审批是正常流程、调用前不要预告我做不到」，但模型仍答「我这里无法直接
操作数据库」。叠加因素：模型换档、`search_knowledge` 召回的表结构文档让模型把自己定位成
「转达 SQL 建议」、双意图稀释。⇒ **提示词加重语气收益递减**，应上「写意图检测 → 强制
调用」或打「有写意图但无写调用」告警指标（`task.clarify_answer_no_write`）。

### C2. 断言的「实现形态耦合」要在改造时同步

曾有断言 `src.count("config=invoke_config") == 2` —— 断言**调用形态**而非不变量。

- 规则：断言写**不变量**，不要写「某字符串出现 N 次」。改造后主动扫既有断言。

### C3. 回归脚本必须 import 生产实现，不能重写逻辑

在测试里重写一遍等价逻辑 → 测试通过但生产没修（**假绿**）。脚本要验证生产代码路径。

### C4. 配置默认值必须与真实 schema 核对

`agent_delete_flag_column` 曾默认 `"is_deleted"`，而 `user` / `conversation` 真实列名是
**`deleted`** → 逻辑删除静默回落物理删除。已改默认空串 + 候选集探测（`deleted` →
`is_deleted` → `delete_flag` → `del_flag` → `is_del` → `deleted_flag`），探测不到显式告警。
`conversation_message` 无标记列，探测返回 `''`。
**凡「某个列名/表名」的默认值，都必须对着真实 schema 核一遍。**

### C5. 诊断方法论：先分清「哪条 UPDATE」

日志里的 `UPDATE xxx` 未必是业务写操作。诊断「执行了更新但状态没变」时**先确认被
UPDATE 的表名**：`task_execution` / `tool_registry` / 审计表 → 框架记账 SQL；业务表
（如 `user`）→ 才是用户期待的写。框架记账成功 ≠ 业务写成功。

### C6. Windows 下 heredoc 写中文会「插错位置 + 损坏字节 + 静默失败」

`cat >> file << 'EOF'` 在 Git Bash + 中文长文本场景反复出问题（§22 踩了三次）：
① 内容被插到文件**开头**而非末尾；② 产生孤立字节（如 `\x86` 应为 `\xe5\x86\x85`＝「内」）
→ 文件无法 UTF-8 解码；③ **写入静默失败**（文件大小与写入前完全一致）。

- **规则：Windows + 中文长文本一律用 Write 工具或 Python
  `path.write_text(..., encoding='utf-8')`，禁止 heredoc 追加。**
- 事后校验三步：`raw.decode('utf-8')` 不抛异常；`text.count('\ufffd') == 0`；
  `grep -n "^## "` 确认章节顺序。
- 修复手法：按字节定位后精确替换；`raw.find(bad)` 可能命中**正常序列的尾部**，必须打印
  上下文人工确认。

### C7. 跨轮上下文：内层任务 Agent 只收到「当前轮」输入（§23 修复 2 已修）

**§23 修复 2 之前的缺陷**：`task_agent_node` 构造 `invoke_input` 时只放一条
`HumanMessage(content=user_input)`，**不带历史对话**。内层 thread 隔离
（`{conv}::task_agent::{tid}`）把「靠检查点兜底拿历史」这条后路也断了 ——
内层图物理上拿不到任何上下文。用户说「给我把**这个用户**的会话信息失效了」，
模型无从知道「这个用户」是谁 → 重复追问用户上一轮已给的实体。

- **根因是跨分支功能不对称**：`chat_node` 有 `MySQLChatMessageHistory.aget_messages()`
  + `[-HISTORY_RECENT_NUM:]` 拼进 prompt，**task_node 没有**。
  消息其实已落库（`_save_conversation_messages`），只是没人读回来。
- 修法：`_load_task_history(db, conversation_id)`，三条边界必须处理——
  ① **末条若是 human 必须裁**（恢复轮重入时末条就是本轮输入；不裁会让同一问题
  出现两次，且本轮用**稳定 id**、历史那条**无 id**，`add_messages` 的按 id
  去重完全不生效）；② **历史消息不带 id**（表无 message_id 列，补 id 会与检查点冲突）；
  ③ 条数上限（`agent_task_history_recent_num`，默认 10）。
- **降级策略**：db 为 None / conversation_id 空 / 构造失败 / 查询失败 → 一律返回空列表
  + 计数告警。上下文是**增强**，拿不到就退回改动前行为，绝不能因加载失败让任务跑不起来。
- 开关 `agent_task_history_inject`（默认 True）可逐字节回滚。
- **诊断教训**：提示词层面的「不要重复问」治不住**输入缺失** —— 先看模型拿到什么，
  再看提示词怎么写。跨分支功能不对称（chat 有、task 没有）是隐性缺陷高发区。

### C8. 断言要写不变量，别写源码字面串（已踩两次）

`diag_task_context_and_keyerror.py` 的 B4 曾断言
`"except (GraphInterrupt, asyncio.TimeoutError, ResumeMismatchError):" in src` ——
修复 1 落地后白名单形态变化，断言必然失配。**这是第二次踩同一个坑**（C2 已记过一次）。

- 规则：改造完成后**必须主动扫一遍全部脚本**，凡「某字符串 in 源码」形式的断言都要
  审一遍：它断言的是**不变量**还是**实现形态**？后者一律改写。
- 本次的正例写法：断言 `"KeyError" not in 白名单`（不变量）而不是白名单字面串。

---

## D. 前端工程（D:\lingxi-agent-portal）

- React 19 + TypeScript + Vite + TailwindCSS；SSE 用 `fetch` + `ReadableStream` 读取
  （**不是 EventSource**，因为要 POST + 自定义 header）。
- 无测试框架。纯逻辑校验脚本放 `scripts/verify-*.mjs`，用 `node` 直接跑。
- 类型检查 `./node_modules/.bin/tsc -b`，构建 `./node_modules/.bin/vite build`。
- **已知既有 lint 报错（非本轮引入，勿误判）**：`App.tsx:46` setState-in-effect、
  `llm.ts:39` explicit-any。
- 审批/追问卡片是**独立消息**还是**挂在 assistant 消息上**有区别：实时路径挂同一消息、
  刷新重建走 `reconcile*Messages` 独立消息。改渲染逻辑时两边都要考虑。
