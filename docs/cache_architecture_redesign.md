# 缓存架构改进方案（语义缓存 × 意图缓存）

> 适用范围：`rag/semantic_cache.py`、`agent/intent_gate.py`、`agent/intent_router.py`、
> `agent/nodes/knowledge_node.py`、`agent/graph_builder.py`
> 目标：在不牺牲正确性与可审计性的前提下，让两级缓存各司其职、叠加生效，而不是互相掩盖。

> **落地状态（2026-09-20）**：
> - **P0 四项已全部实现**，见下方 §4 表格与 `docs/lingxi_agent_langgraph_redesign.md` §18
>   （含配置项、观测指标、改动清单、验收结果：`scripts/smoke_cache_policy.py` 65 项）。
> - **P1-2「检索缓存层」已实现**（2026-09-21），见 §3.5 与
>   `docs/lingxi_agent_langgraph_redesign.md` §18.13，验收：`scripts/smoke_retrieval_cache.py`
>   103 项 + 全量回归 **325 项**全通过。检索缓存自此成为**默认加速层**，答案缓存降为高档位。
> - 剩余 P1（`kb_version` 真实指纹 / 按 `entity_tags` 定向失效）与 P2（Redis 化 / 影子模式）待排期。
>
> 实现细节（函数名、字段名、指标 key）以 §18 为准。

---

## 0. 结论先行

你观察到的现象（重复提问时语义缓存直接返回答案、意图缓存"白做"）是真的，但它只是**表层症状**。
真正的根因有三条，按严重度排序：

| # | 根因 | 严重度 | 说明 |
|---|------|--------|------|
| R1 | **答案级缓存跨过了控制平面** | 高 | 语义缓存命中 = 跳过意图执行、分支副作用、工具调用、审计落痕。它不是"加速"，是一次**全链路短路** |
| R2 | **缓存键维度严重不足** | 高 | 当前 key 只有 `query 向量 + login_username`。缺 intent / 知识库版本 / 模型版本 / 提示词版本 / 实体一致性 → 正确性不可控 |
| R3 | **L2 向量原型层是死代码** | 中 | `state["query_embedding"]` 由 `knowledge_node` 写入，而 `intent_router` 在它之前执行，所以 `classify(query_embedding=...)` 恒为 `None`，L2 永不触发 |

一句话定位：**意图缓存是控制平面（决定"做什么"），语义缓存是数据平面（复用"做出来的东西"）。
现在的问题不是意图缓存没价值，而是数据平面的开关装错了位置——它应该受控制平面管辖，而不是绕过它。**

---

## 1. 现状盘点（基于当前代码）

```
api/routes/agent.py
  └─ graph.ainvoke()
       └─ intent_router_node                      ← 控制平面
            ├─ L0 intent_gate.gate()              规则（斜杠/寒暄/强信号/追问继承）
            ├─ L1 _DECISION_CACHE                 进程内 dict，TTL 1800~3600s
            ├─ L2 match_by_embedding()            ⚠️ query_embedding 恒为 None，从不生效
            └─ L3 LLM 结构化输出（+ 低置信升级重判）
       └─ route_by_intents() → knowledge / chat / task_agent（可并行 fan-out）
            └─ knowledge_node                     ← 数据平面
                 ├─ 1. 计算 query 双向量（稠密 + 稀疏）
                 ├─ 2. ks.get_cached()   ⚠️ 语义缓存命中 → 直接返回，全链路终止
                 ├─ 3. 混合检索
                 ├─ 4. LLM 流式生成
                 └─ 5. ks.put_cache() 写入语义缓存
```

语义缓存 payload 当前字段：`query_text / answer / login_username / created_at / hit_count / context_preview`。
过滤条件：`login_username == 当前用户 OR == __anonymous__`，相似度阈值 `0.92`，TTL `86400s`。

### 1.1 具体风险清单

1. **副作用与缓存共存**：并行意图 `["task","knowledge_base"]` 时，task 分支真的删/改了数据，
   kb 分支却可能命中缓存返回**写操作之前**的答案，最后 merge 节点把"新执行结果"和"旧知识答案"拼在一起。
   ——这是当前架构下最危险的一条。
2. **语义漂移**：`0.92` 余弦阈值对短中文文本过于宽松。"删除张三的记录" 与 "删除李四的记录"
   向量相似度通常在 0.95+，命中即返回**错误答案且无任何提示**。
3. **知识库变更失效粒度太粗**：`invalidate_all()` 删集合重建，一次文档上传清空全部用户缓存；
   若不调用则 24h 内一直返回旧答案。
4. **答案不可溯源**：命中路径 `context_docs=[]`、`context_text=""` 直接返回，审计与引用链断裂，
   企业场景（合规/追责）不可接受。
5. **进程内决策缓存多 worker 不一致**：`_DECISION_CACHE` 是进程级 dict，多副本部署时命中率随机漂移，
   且重启即失，无法做容量/命中率运营。
6. **命中不可观测**：只有一行 `logger.info`，没有 hit/miss/skip 分类计数，也没有"错误命中"反馈回路。
7. **`_evict_if_needed` 每次 put 全量 scroll**：O(n) 开销，条目数上万时写入路径被拖慢。
8. **过期条目只删命中的那一条**：`get()` 里发现过期只删自己，其余长期堆积（依赖不常跑的定时清理）。

---

## 2. 目标架构

### 2.1 核心原则（三条，务必写进团队规范）

> **P1 副作用不可缓存。** 任何带有写操作/工具调用/实时数据读取的意图，答案缓存一律 `DENY`。
> **P2 缓存命中必须等价于"重跑一次的结果"。** 凡是不能证明等价的维度，必须进 key 或进失效策略。
> **P3 控制平面永远先跑。** 意图判定不做答案级缓存短路，只做决策缓存；答案缓存受意图的准入结果管辖。

### 2.2 目标流程

```
入口（api/routes/agent.py）
  └─ ① 预计算 query embedding（一次 API，全链路复用：意图 L2 / 检索 / 语义缓存）
  └─ ② 意图判定（L0 规则 → L1 决策缓存(Redis) → L2 向量原型 → L3 LLM）
  └─ ③ 缓存准入决策  cache_policy = f(intent, 并行?, 时间敏感度)
  └─ ④ policy=ALLOW 时按「升维后的键」查缓存（检索缓存 → 答案缓存）
  └─ ⑤ miss 则正常执行 → 结果写缓存（带完整维度标签）
  └─ ⑥ 写操作成功 → 按 entity_tags 定向失效
```

两级缓存从"互相掩盖"变成"串行两级加速"：
**决策缓存省的是"分类 LLM 往返"，检索/答案缓存省的是"检索 + 生成"。它们不再抢同一次请求的控制权。**

---

## 3. 改进项

### 3.1 【P0】意图驱动的缓存准入（CachePolicy）

在 `intent_gate.py` 增加策略表，由意图直接决定答案缓存是否可用：

```python
# agent/intent_gate.py
ALLOW = "allow"        # 允许答案缓存
DENY  = "deny"         # 禁用（有副作用 / 实时性）
SHORT = "short"        # 允许但短 TTL（创作类、时效类）

def cache_policy_for(intents: list[str]) -> tuple[str, int]:
    """按意图给出答案缓存准入策略与 TTL（秒）。"""
    s = set(intents or [])
    if "task" in s:
        # task 有副作用/实时读；并行场景下 kb 分支也必须禁用，
        # 否则「刚删除又问」会拿到写之前的答案。
        return DENY, 0
    if s == {"knowledge_base"}:
        return ALLOW, 86400
    if s == {"chat"}:
        return SHORT, 900          # 创作类/时效类，短 TTL 或按配置关闭
    return DENY, 0
```

配套改动：
- `agent/state.py` 增加 `cache_policy: str`、`cache_ttl_seconds: int`、`query_embedding` 语义不变；
- `intent_router_node` 返回这两个字段；
- `knowledge_node` 在 `ks.get_cached()` 之前判 `if state.get("cache_policy") == DENY: 跳过缓存`。

**验收**：`["task","knowledge_base"]` 并行请求不再出现语义缓存命中（日志可查）。

### 3.2 【P0】缓存键升维（payload + 强过滤）

`semantic_cache.put()` payload 扩字段，`get()` 用 Qdrant `Filter(must=...)` 强过滤：

```python
payload = {
    "query_text": query,
    "answer": answer,
    "login_username": effective_user,
    "created_at": ...,
    "hit_count": 0,
    # ↓ 新增：缓存维度（任一不匹配即 miss，等价于自动失效）
    "intent": intent,                     # 产生该答案时的意图
    "intents_key": "+".join(sorted(intents)),
    "kb_version": kb_version,             # 知识库版本指纹（见 3.4）
    "model": model,                       # 生成模型名
    "prompt_version": settings.prompt_version,
    "schema_version": CACHE_SCHEMA_VERSION,   # 本结构版本，代码升级直接 bump
    "entity_tags": entity_tags,           # 抽取的实体/表名，用于定向失效
    "ttl_class": ttl_class,
    "context_preview": context_preview,
    "source": "rag",
}
```

```python
# get() 中的强过滤（与权限过滤合并）
must = [
    FieldCondition(key="schema_version", match=MatchValue(value=CACHE_SCHEMA_VERSION)),
    FieldCondition(key="kb_version",     match=MatchValue(value=current_kb_version)),
    FieldCondition(key="intent",         match=MatchValue(value=intent)),
    FieldCondition(key="model",          match=MatchValue(value=model)),
    FieldCondition(key="prompt_version", match=MatchValue(value=settings.prompt_version)),
]
```

> 版本 bump = 逻辑失效，无需删数据；旧条目自然不再被命中，由定时清理回收。
> 这比 `invalidate_all()` 全量删集合温和得多，且可以灰度回滚（把版本号改回去即可）。

### 3.3 【P0】一致性校验：防止"相似却错误"的命中

阈值收紧 + 实体校验，二者同时通过才算命中：

```python
ANSWER_CACHE_THRESHOLD = 0.95   # 答案缓存档：从严（原 0.92）
RETRIEVE_CACHE_THRESHOLD = 0.90 # 检索缓存档：可放宽（见 3.5）

_ENTITY_RE = re.compile(r"[0-9]{2,}|[a-z_][a-z0-9_]{2,}|[\u4e00-\u9fff]{2,6}(?=[的\s]|$)")

def entity_consistent(query: str, cached_query: str) -> bool:
    """关键实体一致性校验：数字/表名/专名必须对齐，否则判 miss。"""
    a, b = set(_ENTITY_RE.findall(query)), set(_ENTITY_RE.findall(cached_query))
    if not a and not b:
        return True
    if not a or not b:
        return False
    return len(a & b) / len(a | b) >= 0.6
```

**可选加强（推荐）**：项目里已有 `bge-reranker-base`，命中后用 reranker 对
`(当前 query, 缓存 query)` 打一次分（本地推理，几毫秒，零 API 成本），低于阈值即判 miss。
这比纯余弦相似度可靠一个量级。

### 3.4 【P1】失效策略：从"全量清"到"按维度失效"

| 触发事件 | 失效动作 |
|---------|---------|
| 文档上传/删除/更新 | `kb_version` bump（知识库内容指纹：`max(updated_at)` + 文档数 + 集合名 hash），旧条目自动失配 |
| 任务写操作成功（insert/update/delete） | 按 `entity_tags` 定向删除（表名/实体命中即删），并 bump 该用户的 task 相关缓存 |
| 模型 / 提示词升级 | bump `model` / `prompt_version`（字段已在 key 中，天然失效） |
| 缓存结构变更 | bump `CACHE_SCHEMA_VERSION` |
| 定时清理 | 按 `created_at < now - ttl_class` 分批 scroll 删除（替换现有全量 scroll，限制单批处理量） |

`entity_tags` 抽取建议：从 task 分支的 `tool_calls` 里取表名与 WHERE 关键值，
从 kb 分支取 `context_docs` 的 `source` 文件名，统一小写后入 payload。

### 3.5 ✅【P1】用"检索缓存"作为默认加速层，答案缓存降为可选高档位

> **已落地（2026-09-21）**，见 `docs/lingxi_agent_langgraph_redesign.md` §18.13。
> 实现：`rag/retrieval_cache.py`（`RetrievalCache`）+ `agent/knowledge_service.py::retrieve()`。

这是本方案里**最重要的一处架构调整**。

- **检索缓存（推荐默认开启）**：缓存 `doc_ids + context_text + 完整 docs`，命中后**仍然走 LLM 生成**。
  - 省掉：向量检索 + RRF 融合 + rerank + 上下文格式化（占 RAG 链路 30%~50% 延迟与主要检索成本）
  - 保住：引用溯源、审计落痕、分支语义、答案与当前上下文的一致性
  - 阈值可放宽到 0.90（错了只是多检索一次，不会给出错误答案）
- **答案缓存（高档位，谨慎开启）**：阈值 0.95 + 实体校验 + reranker 校验 + 意图准入，
  且按 3.6 做灰度与错误命中监控。

这样做之后，"意图缓存有没有意义"的问题自然消失：
每轮请求都照常做意图判定、照常进分支、照常落审计，缓存只是把**最贵的那一段**换成内存/向量查询。

**落地时对原方案的三处修正（均为收紧方向，更安全）**：

| 原方案 | 落地实现 | 理由 |
| --- | --- | --- |
| 缓存 `doc_ids + scores + context_text` | 缓存 `doc_ids + context_text + 完整 docs`；**不存 `scores`** | `HybridRetriever` 的 RRF 融合分只用于排序、未回写 `Document.metadata`，存一堆 0 是误导；溯源靠 `doc_ids`（`chunk_id`）。待 retriever 侧埋点后可启用 |
| 省掉 embedding 调用 | 不省：查检索缓存本身需要 query 向量 | 但入口 §18.5 已预取一次并全链路复用，因此**实际不额外付费**；省的是检索与重排 |
| （未提权限） | 检索缓存**严格用户隔离**，不共享匿名条目 | 匿名检索走的是无权限过滤路径（既有实现），共享会把越权结果固化并放大（§18.13.4） |

**保真不变式**：单文档超 `cache_retrieve_max_doc_chars`（默认 1200）、文档数超
`cache_retrieve_max_docs`（默认 8）、metadata 不可序列化 —— **一律放弃写入而不是截断**。
截断会让「命中」与「重跑」产生不一致的上下文，直接违反 P2。

### 3.6 【P1】修复 embedding 复用链路（让 L2 真正生效）

现状：`query_embedding` 由 `knowledge_node` 写，而 `intent_router` 更早执行 → L2 恒不触发。

改法：在入口 `api/routes/agent.py` 或 `intent_router_node` 开头**预计算一次** query 双向量写入 state，
下游（意图 L2 / 知识检索 / 语义缓存）全部复用，全链路只调一次 embedding API：

```python
# intent_router_node 开头（或入口）
if not state.get("query_embedding"):
    emb, sparse = await KnowledgeService(db).compute_embedding_with_sparse(user_input)
    state_updates = {"query_embedding": emb, "query_sparse": sparse}
```

注意：稀疏向量序列化进 state 时需确认 `SparseVector` 可安全传递（checkpoint 落 MySQL 会序列化），
否则只传稠密向量，稀疏在 knowledge 分支内单独算。

### 3.7 【P2】决策缓存 Redis 化

`_DECISION_CACHE` 换成 Redis（`agent:intent:v1:{hash(normalized_text)}`），value 存
`{kind, rule, intents, confidence, reason, source, ts}`，TTL 沿用分级（forced 3600 / chitchat 1800 / llm 600）。
收益：多 worker 命中率一致、可统计、可运营（命中率/来源分布直接从 Redis 侧采样）。
降级：Redis 不可用时静默回落到进程内 dict，行为与今天一致。

### 3.8 【P2】可观测 + 影子模式灰度

指标（接进现有 `agent/observability.py`）：

```
cache.decision.{hit|miss|skip_intent_deny|skip_version|reject_entity|reject_rerank}
cache.layer.{retrieve|answer}
cache.latency_ms
cache.bad_hit            # 命中后 30s 内用户重问/点踩/重新生成
cache.token_saved
```

灰度路径（三阶段，每阶段看指标再推进）：

1. **影子模式** `CACHE_SHADOW=true`：只算相似度与是否可命中，**不返回缓存答案**，
   记录 `would_hit` 与"缓存答案 vs 实时答案"的差异率。跑 3~7 天，确认差异率 < 1%。
2. **检索缓存开**：只开启检索缓存层，答案照常生成。
3. **答案缓存开灰度**：按 `user_id hash % 100 < cache_answer_percent` 放量，从 5% 起。

自动熔断：`bad_hit_rate` 连续超阈值（建议 0.5%）→ 自动把答案缓存降级为影子模式并告警。

---

## 4. 落地步骤

| 阶段 | 改动 | 工作量 | 风险 |
|------|------|--------|------|
| P0-1 ✅ | `cache_policy_for()` + state 字段 + knowledge_node 准入判断 | 0.5d | 低（只做减法，命中率下降但正确性上升） |
| P0-2 ✅ | payload 扩字段 + `get()` 强过滤 + schema_version | 1d | 低（bump 版本即全量逻辑失效，可回滚） |
| P0-3 ✅ | 阈值收紧（`cache_answer_threshold=0.95`）+ 实体一致性校验 | 0.5d | 低 |
| P0-4 ✅ | embedding 前置，修复 L2 死代码（经 `configurable` 传递，不落 state） | 0.5d | 中（入口多一次 embedding 调用，已做失败降级） |
| P1-1 | `kb_version` 指纹 + 文档变更 bump（替换 `invalidate_all`） | 1d | 中（需 ingestion 侧埋点）。**检索缓存同样依赖它，应优先做** |
| P1-2 ✅ | 检索缓存层 + 答案缓存分档 | 2~3d | 中（新增一层，需影子验证）。已落地，见 §18.13 |
| P1-3 | 任务写操作后按 `entity_tags` 定向失效 | 1d | 中 |
| P2 | 决策缓存 Redis 化 + 影子模式 + 指标看板 | 2~3d | 低 |

**建议顺序**：先做 P0 全套（2.5 天，纯收益、可回滚）→ 检索缓存层（本次已完成）→
`kb_version` 真实指纹 → 按影子模式推进答案缓存放量。

> **注意**：检索缓存目前依赖手工 bump 的 `cache_kb_version`。在 P1-1 落地前，
> 知识库更新后**必须**手动 bump 该配置或调 `/cache/clear`，否则两层缓存都会返回旧内容。
> 这不是新引入的风险（答案缓存同样如此），但检索缓存命中率更高、体感更明显。

---

## 5. 验收指标

| 指标 | 当前 | 目标 |
|------|------|------|
| 语义答案缓存命中量 / 总请求量（"掩盖率"） | 未知，需先埋点 | < 15% |
| task 意图 / 并行意图下答案缓存命中数 | 可能 > 0 | **恒为 0** |
| knowledge 分支检索缓存命中率 | 0 | 25%~45% |
| knowledge 分支答案缓存命中率 | 未知 | 15%~30% |
| 错误命中率 bad_hit_rate | 未观测 | < 0.5% |
| P95 首字延迟降幅 | — | ≥ 30%（检索缓存贡献） |
| LLM token 消耗降幅 | — | ≥ 20% |
| 意图层 LLM 调用占比（L3 占比） | 未知 | < 40%（L0/L1/L2 合计 > 60%） |

观测口径：检索缓存命中率 = `cache.retrieve.hit / (cache.retrieve.hit + cache.retrieve.miss)`；
答案缓存命中率 = `cache.get.hit / (cache.get.hit + cache.get.miss)`。
两层拒绝原因分别看各自前缀下的 `reject.threshold / expired / entity`，
「因保真放弃写入」看 `cache.retrieve.skip.oversize`（该值偏高说明 chunk 尺寸偏大，
应调 `cache_retrieve_max_doc_chars` 或优化切分粒度）。

---

## 6. 取舍说明（明确写清楚，避免团队误读）

- **命中率会下降，这是设计目标而不是退步。** 收紧阈值 + 意图准入 + 版本过滤，
  短期命中率必然降；换来的是"命中即可信"。企业系统里，一次错误答案的成本远高于十次 LLM 调用。
- **答案缓存不是默认项。** 只有"只读、可复现、有明确引用"的知识库问答才配得上答案级缓存。
  task / 实时查询 / 创作类一律不缓存或短 TTL。
- **检索缓存是主力，答案缓存是甜点。** 先保证检索缓存跑通并观测到延迟收益，再考虑答案缓存放量。
- **不要靠 `invalidate_all()` 兜底。** 它是运维级的核弹按钮，不是一致性方案；一致性必须靠 key 维度 + 定向失效。
