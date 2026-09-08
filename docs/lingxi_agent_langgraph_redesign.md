# lingxi-agent 智能化改造开发设计文档（LangChain + LangGraph）

> 版本：v1.0
> 日期：2026-09-04
> 状态：待评审

***

## 目录

1. [文档说明](#1-文档说明)
2. [现状分析](#2-现状分析)
3. [改造目标与技术选型](#3-改造目标与技术选型)
4. [总体架构设计](#4-总体架构设计)
5. [核心模块设计](#5-核心模块设计)
6. [流程图](#6-流程图)
7. \#7-关键功能实现
8. [数据表设计](#8-数据表设计)
9. [API 设计](#9-api-设计)
10. [配置项设计](#10-配置项设计)
11. [安全设计](#11-安全设计)
12. [影响范围与改动清单](#12-影响范围与改动清单)
13. [分阶段实施计划](#13-分阶段实施计划)
14. [风险与注意事项](#14-风险与注意事项)

***

## 1. 文档说明

### 1.1 背景

当前 lingxi-agent 项目是一个基于 FastAPI 的 RAG 问答系统，已具备知识库混合检索（Qdrant 稀疏向量(基于text-embedding-v4) + 稠密向量 + RRF 融合 + Cross-Encoder 重排序）、语义缓存、对话记忆（MySQL）、文件清洗切分与向量入库等能力。

本次改造的核心诉求：

1. 引入 **LangChain + LangGraph** 技术栈，实现基于**意图识别**的路由能力，将用户输入路由到：**知识库问答 / 任务执行 / 普通聊天** 三条链路；
2. **接入飞书审批**：当任务执行涉及**数据库删除或修改**操作时，必须经过**人工审批**后才能执行；
3. 知识库检索**完整保留现有检索流程**（混合检索 + 语义缓存 + 权限过滤）；
4. 技术栈使用**最新且稳定**的版本，版本间保证兼容；
5. 代码使用**最新 API**，不使用官方明确废弃的接口；
6. 代码设计符合**企业级**标准，关键点与函数均有详细注释；
7. 版本变更只改 `requirements.txt`，由用户安装依赖后再进入开发；
8. 先输出本设计文档（含流程图、关键功能实现方案、影响范围），评审通过后再实施。

### 1.2 已确认的设计决策

| 决策点          | 结论                                                                    |
| ------------ | --------------------------------------------------------------------- |
| 审批触发条件       | 仅当「路由到任务执行」且「LLM 决策是对数据库某张表做**删除或修改**」时触发审批；查询/插入不触发                  |
| 审批人确定方式      | 使用**飞书审批流自动路由**（审批流编码 approval\_code + 流程内配置的审批人/角色）                  |
| 接口兼容策略       | **新增接口，保留旧接口**（`/api/conversations/chat` 维持不变，新增 `/api/agent/chat`）   |
| 任务执行工具范围（初期） | 只读查询、数据插入、数据更新（触发审批）、数据删除（触发审批），全部基于**结构化参数 + 表/列白名单**，禁止 LLM 直接拼 SQL |

***

## 2. 现状分析

### 2.1 当前技术栈

| 层次        | 技术                                                                            | 说明                       |
| --------- | ----------------------------------------------------------------------------- | ------------------------ |
| Web 框架    | FastAPI 0.115 + Uvicorn                                                       | 异步 API 服务                |
| ORM       | SQLAlchemy 2.0 (asyncio) + aiomysql                                           | 异步数据库访问                  |
| 配置        | Pydantic v2 + pydantic-settings                                               | `.env.dev` 环境配置          |
| LLM       | `langchain-openai.ChatOpenAI`（通义千问 DashScope 兼容接口）                            | 对话/摘要/意图                 |
| Embedding | 自研 `DashScopeEmbedding`（DashScope 原生 SDK 直连）                            | text-embedding-v4，1024 维（稠密+稀疏双输出） |
| 向量库       | Qdrant（本地磁盘模式）                                                                | 文档向量 + 语义缓存两个集合          |
| 检索        | 自研 `HybridRetriever`：Qdrant 稀疏向量(text-embedding-v4) + 稠密向量 + RRF 融合 + Cross-Encoder 重排 | 核心检索链路                   |
| 记忆        | 自研 `MySQLChatMessageHistory` / `MySQLConversationSummaryMemory`               | 消息持久化 + 长对话摘要压缩          |
| 缓存        | 自研 `SemanticCache`（Qdrant 独立集合）                                               | 语义级问答缓存                  |
| 认证        | JWT + bcrypt + 图片验证码                                                          | 用户体系                     |
| 存储        | 腾讯云 COS                                                                       | 文件持久化                    |
| 追踪        | LangSmith                                                                     | 已接入                      |

### 2.2 当前模块结构

```
lingxi-agent/
├── app/main.py                 # FastAPI 入口：lifespan 初始化、路由注册、全局异常
├── api/routes/                 # 路由层
│   ├── auth.py                 # 验证码/注册/登录/me
│   ├── conversation.py         # 会话 CRUD + 流式对话（use_rag 参数二选一）
│   ├── file_process.py         # 文件清洗切分 + 双向量入库（同 doc_id 先删后插）+ 语义缓存失效
│   ├── metadata.py             # 元数据定义 CRUD
│   ├── attachments.py          # 附件
│   └── cache.py                # 语义缓存统计
├── core/                       # config / database / exceptions
├── models/                     # ORM 模型 + Pydantic Schema
├── rag/
│   ├── hybrid_retriever.py     # HybridRetriever（稀疏+稠密双路） / CrossEncoderReranker
│   ├── rag_conversation_service.py  # RAG 对话服务（检索 + 记忆 + 流式）
│   ├── conversation_service.py # 普通对话服务
│   ├── semantic_cache.py       # 语义缓存
│   └── memory_mysql.py         # MySQL 记忆
├── embeddings/embedding_deal.py # Embedding / Qdrant 客户端
├── ingestion/                  # COS / 解析 / 清洗 / 分块
├── prompt/prompt_storage.py    # 提示词集中管理
├── utils/                      # auth / captcha
└── requirements.txt
```

### 2.3 现有对话 API 流程（保持兼容，不修改）

```
POST /api/conversations/chat   (SSE)
  ├─ use_rag=true  → RAGConversationService.chat_with_rag
  │                  ├─ 预计算 query embedding
  │                  ├─ 语义缓存命中？→ 直接流式返回缓存回答
  │                  ├─ 混合检索（权限过滤 + 稀疏向量(text-embedding-v4) + 稠密向量 + RRF + 重排）
  │                  ├─ 上下文压缩 + 摘要（MySQL）
  │                  ├─ RAG Prompt → LLM 流式生成
  │                  └─ 保存消息 + 写缓存
  └─ use_rag=false → ConversationService.chat_with_memory（普通聊天）
```

### 2.4 现有检索流程（本次改造**原样保留**）

```
用户问题
  ├─ 1. 权限过滤：metadata.auth_option ∈ {public} ∪ {private ∧ create_username=当前用户}
  ├─ 2. 并行召回：稀疏向量召回（text-embedding-v4, top_k=k*5）  ∥  稠密向量（相似度搜索, top_k=k*5）
  ├─ 3. RRF 融合：score(d) = Σ 1/(k + rank_i(d))，k=60
  ├─ 4. Cross-Encoder 重排序（bge-reranker-base 本地模型, 候选 12 条）
  └─ 5. 返回 Top-K 文档（新知识优先，按 created_at 倒序）
```

***

## 3. 改造目标与技术选型

### 3.1 依赖版本矩阵（已写入 requirements.txt）

| 包                          | 版本     | 说明 / 兼容性说明                                               |
| -------------------------- | ------ | -------------------------------------------------------- |
| langchain-core             | 1.6.1  | 最新稳定版，提供 Message / Prompt / Document 等核心抽象               |
| langchain                  | 1.3.10 | 锁定原因：1.4.0 依赖已被官方 **yanked** 的 `langgraph==1.2.11`，安装会失败 |
| langchain-openai           | 1.6.0  | ChatOpenAI 官方适配（通义千问 DashScope 兼容接口）                     |
| langchain-qdrant           | 1.1.0  | QdrantVectorStore 集成，沿用现有向量库                             |
| langgraph                  | 1.2.10 | 锁定原因：1.2.11 已被官方 **yanked**（broken），1.2.10 为最新可用稳定版      |
| langgraph-checkpoint       | 4.2.0  | 图状态持久化基础库（由 langgraph 自动拉取）                              |
| langgraph-checkpoint-mysql | 3.0.0  | MySQL 检查点保存器，用于跨请求恢复审批中断点                                |
| openai                     | 3.8.0  | 最新稳定版（embedding\_deal.py 直连使用，旧 1.x 语法已废弃）               |
| lark-oapi                  | 1.7.3  | 飞书开放平台官方 SDK（审批流对接）                                      |
| qdrant-client              | 1.19.0 | 向量库客户端                                                   |
| langsmith                  | 0.12.1 | 链路追踪                                                     |

> 说明：`langchain-community` 官方已宣布 sunset（停止维护），本项目未使用，已从依赖中移除，避免安全隐患。

### 3.2 技术选型原则

1. **LangGraph 承担编排**：意图路由、条件分支、多意图**并行执行与自动汇聚**、任务执行、Human-in-the-loop 审批，全部由 StateGraph 表达，状态显式、可回放、可持久化；
2. **LangChain 承担模型抽象**：ChatOpenAI、Prompt、Message、Structured Output、Document 等；
3. **复用现有检索资产**：`HybridRetriever`、`SemanticCache` 等检索组件不重写，通过薄适配层接入图节点；稀疏向量与稠密向量均由 **text-embedding-v4** 一次调用双输出并存储于 Qdrant，**不再维护独立 BM25 内存索引**；
4. **审批采用官方 HITL 机制**：`create_agent` + `HumanInTheLoopMiddleware` + LangGraph `interrupt()` + `MySQLAsyncSaver` 检查点持久化，天然支持服务重启后继续审批流；
5. **API 全部使用最新官方接口**：`StateGraph/add_node/add_conditional_edges/interrupt/Command/with_structured_output` 等。任务 Agent 使用 `langchain.agents.create_agent`（LangGraph v1 官方推荐），**不使用已弃用的** **`langgraph.prebuilt.create_react_agent`**，也不使用已废弃的 `Chain`/`LLMChain`/`SequentialChain` 等旧 API。

***

## 4. 总体架构设计

### 4.1 架构分层

```
┌─────────────────────────────────────────────────────────────────────┐
│                         接入层（FastAPI）                            │
│  POST /api/agent/chat(SSE)   POST /api/approval/callback  查询接口  │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────┐
│                      Agent 编排层（LangGraph）                       │
│  ┌──────────────┐    ┌──────────────────────────────────────────┐  │
│  │ IntentRouter │───►│          AgentStateGraph                 │  │
│  │  意图识别     │    │  intent_router → 条件路由                │  │
│  └──────────────┘    │   ├→ knowledge_subgraph（知识库问答）      │  │
│                      │   ├→ chat_node（普通聊天）                │  │
│                      │   └→ task_subgraph（任务执行 + 审批）      │  │
│                      └──────────────────────────────────────────┘  │
└───────────────┬────────────────────┬────────────────────┬──────────┘
                │                    │                    │
┌───────────────▼──────┐  ┌──────────▼─────────┐  ┌───────▼───────────────┐
│  知识服务（复用现状）  │  │  Agent 工具层       │  │ 审批服务（飞书）       │
│  HybridRetriever      │  │  query/insert/     │  │  lark-oapi            │
│  SemanticCache        │  │  update/delete     │  │  ApprovalService      │
│  MySQL 记忆           │  │  +表/列白名单       │  │  Feishu 回调           │
└───────────────────────┘  └───────────────────┘  └───────────────────────┘
                │                    │                    │
┌───────────────▼────────────────────▼────────────────────▼──────────────┐
│                       基础设施层                                       │
│  Qdrant(向量+缓存)  MySQL(业务+检查点+审批单)  DashScope LLM/Embedding  │
│  COS(文件)  LangSmith(追踪)  JWT(认证)                                │
└───────────────────────────────────────────────────────────────────────┘
```

### 4.2 新增模块结构

```
lingxi-agent/
├── agent/                              # 新增：Agent 编排层
│   ├── __init__.py
│   ├── state.py                        # LangGraph 图状态定义（TypedDict + Reducer）
│   ├── graph_builder.py                # 组装主图 + 条件路由 + 编译（checkpointer）
│   ├── intent_router.py                # 意图识别节点（Structured Output）
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── knowledge_node.py           # 知识库问答节点（复用现有检索流程）
│   │   ├── chat_node.py                # 普通聊天节点
│   │   └── task_node.py                # 任务执行节点（create_agent + HITL 中间件）
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── registry.py                 # 工具注册表（名称/风险等级/审批开关）
│   │   └── db_tools.py                 # 数据库工具（结构化参数 + 白名单 + 参数化执行）
│   ├── approval/
│   │   ├── __init__.py
│   │   ├── feishu_client.py            # 飞书审批客户端（lark-oapi）
│   │   ├── approval_service.py         # 审批业务（建单/查状态/回调处理/恢复图）
│   │   └── callback.py                 # 飞书事件回调签名校验与解密
│   └── knowledge_service.py            # 知识服务薄适配层（调用现有 RAG 组件）
├── models/
│   ├── task_model.py                   # 新增：任务执行记录表
│   ├── task_schema.py                  # 新增：任务相关 Schema
│   ├── approval_model.py               # 新增：审批单表
│   └── approval_schema.py              # 新增：审批相关 Schema
├── api/routes/
│   ├── agent.py                        # 新增：/api/agent/chat (SSE) 等
│   └── approval.py                     # 新增：/api/approval/*（含飞书回调）
└── prompt/
    ├── prompt_storage.py               # 扩展：意图识别/任务执行提示词
```

***

## 5. 核心模块设计

### 5.1 图状态（AgentState）

```python
# agent/state.py
from typing import TypedDict, Annotated, Optional, Any
from langgraph.graph.message import add_messages

class AgentState(TypedDict, total=False):
    # ---- 对话上下文 ----
    conversation_id: str                # 会话 ID（= LangGraph thread_id）
    user_id: Optional[str]              # 用户 ID
    username: Optional[str]             # 用户名（用于知识库权限过滤）
    model: str                          # 模型名
    messages: Annotated[list, add_messages]   # 消息列表（含历史 + 当前输入）

    # ---- 意图识别（多标签） ----
    intents: list                      # 多意图列表，如 ["task","knowledge_base"] / ["chat"]
    intent_reason: Optional[str]       # LLM 分类理由（可观测）
    intent_confidence: Optional[float] # 置信度

    # ---- 知识库问答 ----
    context_docs: list                 # 检索到的文档
    context_text: Optional[str]        # 格式化后的上下文
    cached_answer: Optional[str]       # 语义缓存命中时直接使用
    rag_answer: Optional[str]          # 知识库分支产出（并行场景下与任务分支隔离）

    # ---- 任务执行 ----
    task_execution_id: Optional[str]    # 任务执行记录 ID
    tool_calls: list                    # 本次任务的工具调用序列（审计）
    tool_results: list                  # 工具执行结果
    pending_write: Optional[dict]       # 待审批的写操作（工具名/表/参数）
    approval_required: bool             # 是否需要审批
    approval_status: Optional[str]      # pending / approved / rejected
    approval_id: Optional[str]          # 审批单 ID

    # ---- 输出 ----
    task_answer: Optional[str]          # 任务分支产出（含审批后结果，供汇总节点合并）
    final_response: Optional[str]      # 汇总节点合并后的最终回答（供落库与回溯）
    error: Optional[str]               # 错误信息
```

### 5.2 意图识别（IntentRouter）

- 采用 `ChatOpenAI.with_structured_output()`（官方最新结构化输出 API，非废弃的 `output_parser` 手工解析）；
- **多标签分类**：同一输入可能同时命中「任务执行 + 知识库问答」，输出意图**列表**而非单一意图：

```python
class IntentResult(BaseModel):
    intents: list[Literal["knowledge_base", "task", "chat"]]  # 可含多个，如 ["task","knowledge_base"]
    reason: str          # 分类依据（可观测/审计）
    confidence: float    # 0~1（整体置信度，取最低子意图置信度）
```

- 提示词（放入 `prompt/prompt_storage.py`，集中管理）：
  - `knowledge_base`：与知识库中文档内容相关的提问（政策、说明书、FAQ 等）；
  - `task`：需要对**业务数据库数据**执行操作的请求（查询/新增/修改/删除某条数据、某张表）；
  - `chat`：寒暄、闲聊、通用知识问答，不涉及知识库文档与数据库操作。
- **意图组合规则**（路由函数执行，见 5.8）：
  - 仅 `["chat"]` → 普通聊天；
  - 仅 `["knowledge_base"]` / 仅 `["task"]` → 单分支；
  - `["task","knowledge_base"]` 同时命中 → **并行**执行任务子图与知识库子图，最后汇总；
  - `chat` 与其它意图并存时**丢弃 chat**（chat 仅作兜底，不参与并行）；
  - 异常输入（如同时命中 3 个）→ 按 `task > knowledge_base > chat` 优先级收敛。
- **降级策略**：LLM 调用失败、输出非法、置信度 < 0.5 时，默认路由到 `chat`，保证服务可用性；
- **安全策略**：命中 `task` 的输入，在进入任务执行前会再次由任务 Agent 判断是否真的需要写操作，双重确认降低误判。

### 5.3 知识库问答节点（KnowledgeNode）

**复用现有检索流程**，通过薄适配层 `agent/knowledge_service.py` 调用现有组件，不改动 `rag/` 内部实现：

```python
# agent/knowledge_service.py（伪代码）
class KnowledgeService:
    def __init__(self, db): self.rag = RAGConversationService(db_session=db)

    async def get_cached(self, user_input, username, embedding):
        return await get_semantic_cache().get(user_input, username, precomputed_embedding=embedding)

    async def retrieve(self, user_input, username, embedding, k=4):
        return await self.rag.retrieve_context_public(...)   # 调用现有混合检索

    async def build_prompt_inputs(self, conversation_id):
        history = await self.rag.get_compressed_history_public(conversation_id)  # 复用压缩+摘要
        return history
```

节点逻辑：

1. 预计算 query embedding；
2. 语义缓存检查（命中 → 直接作为回答，写入消息）；
3. 未命中 → 混合检索（权限过滤 + 稀疏向量(text-embedding-v4) + 稠密向量 + RRF + 重排，`k=4`）；
4. 组装 RAG Prompt（复用 `RAG_SYSTEM_PROMPT_WITH_CONTEXT/WITHOUT_CONTEXT`）；
5. LLM 流式生成（供 API 层 `stream_mode="messages"` 转发 SSE），产出写入 `rag_answer`（并行场景下供 merge 节点合并）；
6. 回答落库（MySQL） + 写入语义缓存。

### 5.4 普通聊天节点（ChatNode）

- 复用 `ConversationService.chat_with_memory` 的 Prompt 组装与记忆逻辑；
- 不使用 RAG 检索，走 `CHAT_SYSTEM_PROMPT` + 最近历史；
- 支持流式输出。

### 5.5 任务执行节点（TaskNode）与工具层

#### 5.5.1 工具清单（初期）

| 工具名           | 能力              | 风险等级 | 是否触发审批       |
| ------------- | --------------- | ---- | ------------ |
| `list_tables` | 列出可操作的表（白名单）    | 只读   | 否            |
| `query_data`  | 结构化条件查询（SELECT） | 只读   | 否            |
| `insert_data` | 新增记录（INSERT）    | 写    | 否（可配置，默认不审批） |
| `update_data` | 修改记录（UPDATE）    | 写    | **是**        |
| `delete_data` | 删除记录（DELETE）    | 写    | **是**        |

> 工具通过注册表 `agent/tools/registry.py` 声明，每个工具带有 `requires_approval` 标记，未来扩展新工具只需在注册表声明，无需改动图逻辑。

#### 5.5.2 结构化参数与安全执行

- **禁止 LLM 拼接 SQL**：工具入参为结构化字段（`table`、`columns`、`filters`、`values`），工具内部用 **SQLAlchemy Core + 绑定参数**构造语句（天然防注入）；
- **表/列白名单**：`AGENT_DB_ALLOWED_TABLES` 配置允许操作的表；列名校验防止越权访问敏感字段；
- **查询限制**：SELECT 默认 `LIMIT 50`、超时保护、结果字段裁剪；
- **审计**：每次工具调用（含参数、结果摘要）写入 `task_execution` 表。

#### 5.5.3 任务执行节点流程

1. 图路由到 `task` → 进入任务子图；
2. 使用 `langchain.agents.create_agent`（LangGraph v1 官方推荐的最新 Agent API，`langgraph.prebuilt.create_react_agent` 已在 v1 中弃用）创建任务 Agent，绑定工具 + LLM，并挂载 `HumanInTheLoopMiddleware`（官方 HITL 中间件，内置写操作审批中断能力）；
3. Agent 规划 → 若仅调用只读工具，直接执行并汇总结果；
4. 若 Agent 决定调用 `update_data` / `delete_data` → 中间件依据 `interrupt_on` 策略在**工具执行前**触发 `interrupt`（写操作审批门，见 5.6）；
5. 审批通过 → 恢复执行写工具 → 汇总结果；审批拒绝 → 终止并告知用户；
6. 任务结束，生成人类可读的执行报告作为回答。

### 5.6 飞书审批（Human-in-the-loop）

基于 LangChain v1 官方 HITL 机制：`create_agent` + `HumanInTheLoopMiddleware`（内部触发 LangGraph `interrupt()`）+ `MySQLAsyncSaver` 检查点持久化：

```
任务 Agent（create_agent + HumanInTheLoopMiddleware）
  ├─ Agent 决策调用 update_data / delete_data
  ├─ 中间件在工具执行前 interrupt()
  │    （携带 HITLRequest：action_requests + review_configs，
  │     为每个待审批工具调用生成 action request）
  ├─ 中断状态持久化（thread_id=conversation_id，MySQL 检查点）
        │
        ▼
┌───────────── 执行包装层（TaskExecutionService）─────────────┐
│  检测到 interrupt 后：                                       │
│  ① 从 action_requests 提取工具名/参数                        │
│  ② 落库审批单（approval_request，status=pending）            │
│  ③ 创建飞书审批实例（表单摘要，审批流自动路由）               │
│  ④ SSE 推送 event=approval_required + approval_id           │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
┌───────────── 飞书端 ─────────────┐
│  审批人收到审批，点击通过/拒绝    │
│  → 审批流回调 → POST /api/approval/callback
└──────────────────────────────────┘
                               ▼
回调处理器：验签 → 解密 → 更新 approval_request 状态
                               ▼
以 Command(resume=HITLResponse(decisions=[approve/reject])) 恢复图执行（后台任务）
   ├─ approved → 执行写工具 → 完成任务
   └─ rejected → 终止任务，生成"已拒绝"回答
```

**关键点**：

- 中断点状态（含待审批的工具调用）由 MySQL 检查点保存，**服务重启后仍可恢复**；
- `interrupt_on` 策略按工具名配置：`update_data` / `delete_data` → `allowed_decisions=["approve","reject"]`（不允许 edit/respond，保证审批语义严格）；`insert_data` 默认 `False`（自动放行），如需按运行时条件审批可通过 `when` 回调动态判断；
- 审批回调与恢复为异步后台任务，SSE 连接通过心跳保活，完成后推送最终结果（同时提供轮询接口兜底）；
- 审批单状态与图状态双写，通过 `approval_request.id` 关联，保证幂等（重复回调/重复恢复均被状态守卫拦截）。

### 5.7 记忆与消息持久化

- 图内 `messages` 为工作态；每次请求从 MySQL 加载历史（复用 `MySQLChatMessageHistory`），结束后写入新增消息；
- 长对话压缩与摘要复用现有 `MySQLConversationSummaryMemory` 逻辑（通过薄适配层暴露公开方法）；
- LangGraph 检查点仅服务于审批中断/恢复场景，与业务消息表互不冲突。

### 5.8 多意图并行路由与汇总（核心新增）

当用户输入**同时包含任务执行与知识库问答**（如"查一下订单数据，并对照知识库里的售后政策给出处理建议"）时，利用 LangGraph 原生 DAG 能力实现**并行 fan-out + 自动汇聚（barrier）**：

```
intent_router（多标签）
        │  条件路由（route_by_intents）
        ├─ 仅单意图 ──────────────► 对应单分支（knowledge / task_agent / chat）
        └─ ["task","knowledge_base"] ─► 同一 superstep 并行执行：
                                        ├─ knowledge 节点（复用现有检索，输出 rag_answer）
                                        └─ task_agent 节点（create_agent + HITL，输出 task_answer）
                                                 │
                                                 ▼
                                        merge 节点（等待所有前驱完成后自动汇聚，输出 final_response）
```

#### 5.8.1 并行机制（LangGraph 原生能力，无需引入额外并发框架）

| 能力   | LangGraph 机制                                                            | 本项目用法                                               |
| ---- | ----------------------------------------------------------------------- | --------------------------------------------------- |
| 并行执行 | 条件路由映射值可为**节点名列表**，同一 superstep 中无依赖节点并行运行                              | `route_by_intents` 返回 `["knowledge", "task_agent"]` |
| 自动汇聚 | 节点有**多条入边**时，LangGraph 在所有前驱节点完成后才执行（barrier/join）                      | `merge` 节点收敛两个分支                                    |
| 状态隔离 | 并行分支写入**不同字段**，避免 reducer 冲突                                            | 知识库分支写 `rag_answer`，任务分支写 `task_answer`             |
| 中断语义 | 任一分支触发 `interrupt` 会**暂停整个图**，其余分支已产出的结果保留在 state 中，resume 后继续执行到 merge | 任务分支审批中断时，`rag_answer` 已落 state，审批通过后直接进入汇总         |

#### 5.8.2 与飞书审批的交互（并行 + 中断）

- 知识库分支通常先完成：`rag_answer` 写入 state，检索与生成结果**不因审批中断而丢失**；
- 任务分支命中写操作 → HITL 中间件 interrupt → 整个图暂停（thread\_id=conversation\_id 持久化）；
- SSE 行为：知识库答案的 token 在中断前已可流式下发 → 随后推送 `event=approval_required` → 审批通过后恢复 → 任务分支完成 → merge 汇总 → 推送最终回答；
- 兜底：若前端连接断开，可通过 `GET /api/agent/tasks/{task_execution_id}` 轮询获得合并后的 `final_response`。

#### 5.8.3 汇总节点（merge）

- **合并策略**：两条分支结果做最终 LLM 摘要（保证上下文连贯、去重、自然衔接），而非简单字符串拼接：
  - 输入：`rag_answer` + `task_answer` + 原用户输入；
  - 输出：`final_response`（结构化为「知识库结论 + 任务执行结果 + 综合建议」三段式，可视场景省略段）；
  - 若某分支失败/无结果，则汇总节点对剩余分支结果直接润色输出（容错）；
- **可观测性**：`final_response` 连同各分支原始结果一并落库 `task_execution`（含 `rag_answer`、`task_answer` 快照），便于审计与回溯；
- 汇总节点为普通 LLM 调用（非 Agent），无工具、无审批，天然适合作为 DAG 汇聚点。

***

## 6. 流程图

### 6.1 总体路由流程图

```mermaid
flowchart TD
    A[用户输入] --> B[意图识别节点<br/>LLM + 多标签结构化输出]
    B --> C{意图组合}
    C -->|仅 knowledge_base| D[知识库问答子图]
    C -->|仅 task| E[任务执行子图]
    C -->|仅 chat| F[普通聊天节点]
    C -->|task + knowledge_base| G[知识库问答子图]
    C -->|task + knowledge_base| H[任务执行子图]
    C -->|异常/低置信度| F
    D --> I[汇总节点 merge]
    E --> I
    F --> I
    G --> I
    H --> I
    I --> J[SSE 流式返回]
    J --> K[消息落库 MySQL + 语义缓存写入<br/>并行结果快照落 task_execution]
```

> 说明：`G`（知识库）与 `H`（任务）在同一 superstep 并行执行；`merge` 节点有多条入边，LangGraph 自动在其所有前驱完成后才执行（barrier）。

### 6.2 知识库问答子图（复用现有检索流程）

```mermaid
flowchart TD
    A[当前用户输入] --> B[预计算 query embedding]
    B --> C{语义缓存命中?}
    C -->|是| D[直接返回缓存回答]
    C -->|否| E[构建权限过滤<br/>public 或 private-本人]
    E --> F[混合检索<br/>稀疏向量 并行 稠密向量]
    F --> G[RRF 融合 k=60]
    G --> H[Cross-Encoder 重排]
    H --> I[组装 RAG Prompt<br/>上下文压缩 + 摘要]
    I --> J[LLM 流式生成]
    J --> K[保存消息 + 写入语义缓存]
    D --> K
```

### 6.3 任务执行子图（含审批门）

```mermaid
flowchart TD
    A[路由到 task] --> B[任务 Agent<br/>create_agent + HITL中间件]
    B --> C{需要写库?<br/>update_data/delete_data}
    C -->|否| D[执行只读/插入工具]
    D --> E[生成执行报告]
    C -->|是| F[中间件在工具执行前中断<br/>HITLRequest 携带待审批工具]
    F --> G[执行包装层<br/>落库审批单 + 创建飞书审批实例]
    G --> H[中断状态持久化<br/>thread_id=conversation_id]
    H --> I[等待飞书回调]
    I --> K{审批结果}
    K -->|通过| L[恢复执行写工具]
    L --> E
    K -->|拒绝| M[终止任务<br/>生成拒绝回答]
    E --> N[SSE 返回 + 落库]
    M --> N
```

### 6.4 飞书审批交互时序图

```mermaid
sequenceDiagram
    participant U as 用户(前端)
    participant B as Agent服务(FastAPI+LangGraph)
    participant F as 飞书审批流
    participant A as 审批人

    U->>B: POST /api/agent/chat (SSE)
    B->>B: 意图识别 → task
    B->>B: 任务 Agent 决策写库 → HITL中间件 interrupt()
    B->>B: 执行包装层：落库审批单 + 创建飞书审批实例
    B-->>U: SSE event=approval_required<br/>(approval_id)
    B->>F: 创建审批实例(approval_code)
    F-->>A: 推送审批任务（自动路由）
    A->>F: 通过 / 拒绝
    F->>B: 回调 POST /api/approval/callback
    B->>B: 验签解密 → 更新审批单
    B->>B: Command(resume=结果) 恢复图
    B-->>U: SSE 推送任务结果/拒绝原因
```

### 6.5 审批触发判定逻辑

```mermaid
flowchart TD
    A["路由到 task 子图"] --> B{"LLM 决策?"}
    B -->|update_data| C["写操作 触发审批"]
    B -->|delete_data| C
    B -->|query_data| D["只读 不审批"]
    B -->|insert_data| E["写操作<br/>默认不审批-可配置"]
    B -->|list_tables| D
```

### 6.6 多意图并行路由与汇总流程图

```mermaid
flowchart TD
    A["用户输入"] --> B["意图识别<br/>多标签分类"]
    B --> C{"同时含 task 与 knowledge_base?"}
    C -->|是| D["并行 fan-out 到两分支"]
    C -->|否| E["单意图路由<br/>knowledge / task / chat"]
    D --> F["知识库分支<br/>检索+生成 产出 rag_answer"]
    D --> G["任务分支<br/>create_agent+HITL 产出 task_answer"]
    E --> H["单分支执行"]
    F --> I["merge 汇总节点<br/>barrier 汇聚两分支"]
    G --> I
    H --> I
    I --> J["LLM 合并润色<br/>知识库结论 + 任务结果 + 综合建议"]
    J --> K["final_response 落库 + SSE 返回"]
```

### 6.7 多意图并行 + 审批中断时序图

```mermaid
sequenceDiagram
    participant U as 用户(前端)
    participant G as LangGraph 主图
    participant K as 知识库分支
    participant T as 任务分支(HITL)
    participant F as 飞书审批流
    participant M as merge 汇总节点

    U->>G: POST /api/agent/chat (SSE)
    G->>G: 意图识别 → [task, knowledge_base]
    G->>K: 并行执行（同一 superstep）
    G->>T: 并行执行（同一 superstep）
    K-->>G: 返回 rag_answer（token 已流式下发）
    T->>T: 决策写库 → HITL interrupt
    G-->>U: SSE event=approval_required
    G->>F: 创建飞书审批实例
    F-->>T: 审批通过回调 → Command(resume=approve)
    T-->>G: 返回 task_answer
    G->>M: barrier 汇聚两分支结果
    M-->>G: 返回 final_response（合并润色）
    G-->>U: SSE 推送最终回答
```

***

## 7. 关键功能实现

### 7.1 意图识别节点实现

```python
# agent/intent_router.py（设计示意，最终以实际编码为准）
from typing import Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from core.config import settings
from prompt.prompt_storage import INTENT_CLASSIFICATION_PROMPT

class IntentResult(BaseModel):
    """意图识别结构化输出（多标签：同一输入可命中多个意图）"""
    intents: list[Literal["knowledge_base", "task", "chat"]] = Field(description="意图列表")
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
        ).with_structured_output(IntentResult)   # 官方最新结构化输出 API

    async def classify(self, user_input: str, history_text: str = "") -> IntentResult:
        """
        对用户输入进行多标签意图分类。
        失败或异常时降级为 ["chat"]，保证主流程可用。
        """
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", INTENT_CLASSIFICATION_PROMPT),
                ("human", "{input}"),
            ])
            result = await (prompt | self._llm).ainvoke({"input": user_input})
            if result.confidence < 0.5:          # 低置信度兜底
                return IntentResult(intents=["chat"], reason="置信度低于阈值", confidence=result.confidence)
            return self._normalize(result)
        except Exception:
            return IntentResult(intents=["chat"], reason="意图识别异常，降级为普通聊天", confidence=0)

    @staticmethod
    def _normalize(result: IntentResult) -> IntentResult:
        """
        意图归一化：去重、丢弃 chat（chat 仅作兜底）、异常多意图按优先级收敛。
        - ["task","knowledge_base"] → 保留两个，触发并行分支
        - 含 chat 的其它组合 → 丢弃 chat
        - 空列表 / 三个全命中 → 按 task > knowledge_base > chat 收敛
        """
        intents = list(dict.fromkeys(result.intents))          # 去重保序
        if "chat" in intents and len(intents) > 1:             # chat 不参与并行
            intents.remove("chat")
        if not intents or len(intents) > 2:
            intents = [i for i in ("task", "knowledge_base") if i in intents] or ["chat"]
        return IntentResult(intents=intents, reason=result.reason, confidence=result.confidence)
```

```python
# agent/graph_builder.py —— 意图组合路由函数（多标签 → 节点列表，实现并行 fan-out）
def route_by_intents(state: AgentState) -> str | list[str]:
    """
    根据多意图返回路由目标。
    - 返回节点名**列表**时，LangGraph 会在同一 superstep 并行执行多个节点；
    - 返回单个节点名则走单分支。
    """
    intents = set(state.get("intents", ["chat"]))
    if intents == {"task", "knowledge_base"}:
        return ["knowledge", "task_agent"]     # 并行 fan-out（关键）
    if "task" in intents:
        return "task_agent"
    if "knowledge_base" in intents:
        return "knowledge"
    return "chat"                              # 兜底
```

### 7.2 主图组装与审批中断/恢复

```python
# agent/graph_builder.py（设计示意）
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.mysql.aio import MySQLAsyncSaver
from langgraph.types import Command
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from agent.state import AgentState   # 外层图状态（含意图/审批字段）

def build_task_agent(llm, tools, checkpointer):
    """
    构建任务执行 Agent（LangGraph v1 官方推荐 API）。

    说明：
    - `langgraph.prebuilt.create_react_agent` 已在 LangGraph v1 中弃用，
      统一使用 `langchain.agents.create_agent`（底层仍运行在 LangGraph 上）；
    - `HumanInTheLoopMiddleware` 在写工具**执行前**触发 interrupt，
      通过 interrupt_on 策略实现审批门；中断状态由 checkpointer 持久化。
    """
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=TASK_AGENT_SYSTEM_PROMPT,
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    # 写操作：仅允许 approve / reject，禁止 edit/respond，语义严格
                    "update_data": {"allowed_decisions": ["approve", "reject"]},
                    "delete_data": {"allowed_decisions": ["approve", "reject"]},
                    # 插入默认自动放行（可配置；如需动态判断可用 when 回调）
                    "insert_data": False,
                },
                description_prefix="数据库写操作待审批",
            ),
        ],
        checkpointer=checkpointer,   # 与主图共用 MySQL 检查点（thread_id=conversation_id）
    )

def build_graph(checkpointer) -> StateGraph:
    """构建主图：意图路由 + 知识库/聊天/任务分支 + 并行汇总"""
    g = StateGraph(AgentState)
    g.add_node("intent_router", intent_router_node)
    g.add_node("knowledge", knowledge_node)
    g.add_node("chat", chat_node)
    # 任务 Agent 作为子图节点嵌入主图；中间件 interrupt 会冒泡到主图 invoke
    g.add_node("task_agent", build_task_agent(llm, tools, checkpointer))
    g.add_node("merge", merge_node)           # 汇总节点（barrier：多入边自动汇聚）
    g.add_node("finalize", finalize_node)     # 收尾：落库 + 快照

    g.add_edge(START, "intent_router")
    g.add_conditional_edges(
        "intent_router",
        route_by_intents,                     # 多标签路由：可返回节点名列表 → 并行 fan-out
        # 映射值可为单个节点名或节点名列表；列表中的节点在同一 superstep 并行执行
        {"knowledge": "knowledge", "task_agent": "task_agent", "chat": "chat"},
    )
    # 三条分支统一汇入 merge（barrier），单分支时 merge 仅透传润色
    g.add_edge("knowledge", "merge")
    g.add_edge("chat", "merge")
    g.add_edge("task_agent", "merge")
    g.add_edge("merge", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)
```

**汇总节点（merge\_node）实现**：

```python
# agent/nodes/merge_node.py（设计示意）
from langchain_core.prompts import ChatPromptTemplate
from prompt.prompt_storage import MERGE_ANSWERS_PROMPT

async def merge_node(state: AgentState) -> dict:
    """
    汇总节点：合并知识库分支（rag_answer）与任务分支（task_answer）结果。

    执行策略：
    1. 仅一个分支有结果 → 直接作为 final_response（免 LLM 调用，省时省 token）；
    2. 两个分支均有结果 → LLM 合并润色（知识库结论 + 任务结果 + 综合建议）；
    3. 分支失败 → 由异常处理兜底为可读错误信息。
    """
    rag = state.get("rag_answer")
    task = state.get("task_answer")
    if rag and task:
        prompt = ChatPromptTemplate.from_messages([
            ("system", MERGE_ANSWERS_PROMPT),
            ("human", "用户问题：{question}\n知识库结论：{rag}\n任务执行结果：{task}"),
        ])
        final_response = await (prompt | llm).ainvoke({
            "question": state["messages"][-1].content,
            "rag": rag, "task": task,
        })
        return {"final_response": final_response.content}
    return {"final_response": rag or task or "暂无可用结果"}
```

**执行包装层（TaskExecutionService）—— 审批中断检测与恢复**：

```python
# agent/approval/approval_service.py（设计示意）
async def run_and_handle_approval(graph, inputs, config, sse_queue):
    """
    运行图并处理审批中断：
    1. 正常完成 → 直接输出；
    2. 命中 HITL interrupt → 落库审批单 + 创建飞书审批实例 + SSE 通知，
       等待回调恢复（回调处理器以 Command(resume=...) 恢复同一 thread）。
    """
    interrupted = None
    async for chunk in graph.astream(inputs, config=config, stream_mode="updates"):
        if "__interrupt__" in chunk:
            interrupted = chunk["__interrupt__"]
    if interrupted is None:
        return

    # 解析 HITLRequest：提取待审批工具调用（action_requests）
    hitl = interrupted[0].value                 # HITLRequest（action_requests + review_configs）
    approval_id = await save_pending_approval(hitl)     # ① 落库审批单
    instance_code = await feishu.create_instance(...)   # ② 创建飞书审批实例
    await update_instance_code(approval_id, instance_code)
    await sse_queue.put({"event": "approval_required", "approval_id": approval_id})  # ③ SSE

# 审批回调后恢复（飞书回调处理器调用）：
# approved → decisions=[{"type": "approve"}]
# rejected → decisions=[{"type": "reject", "message": "审批人拒绝原因"}]
# 注：HITLResponse/Decision 的具体构造方式以 langchain.agents.middleware 实际 API 为准，编码阶段核对官方文档
await graph.ainvoke(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config={"configurable": {"thread_id": conversation_id}},
)
```

### 7.3 飞书审批客户端

```python
# agent/approval/feishu_client.py（设计示意）
import lark_oapi as lark
from lark_oapi.api.approval.v4 import (
    CreateInstanceRequestBody, CreateInstanceRequest, InstanceForm, NodeApprover,
)

class FeishuApprovalClient:
    """飞书审批客户端：创建审批实例（审批流自动路由）"""

    def __init__(self, app_id: str, app_secret: str):
        self.client = lark.Client.builder() \
            .app_id(app_id).app_secret(app_secret) \
            .log_level(lark.LogLevel.INFO).build()

    def create_instance(self, approval_code: str, user_id: str, form_fields: list[dict]) -> str:
        """
        创建审批实例，返回 instance_code。
        - approval_code: 飞书审批流编码（在审批流中配置审批人/角色实现自动路由）
        - form_fields: 审批表单，如 [{"name": "操作类型", "value": "删除数据"}, ...]
        """
        body = CreateInstanceRequestBody.builder() \
            .approval_code(approval_code) \
            .user_id(user_id) \
            .form([InstanceForm.builder().name(f["name"]).value(f["value"]).build()
                   for f in form_fields]) \
            .build()
        req = CreateInstanceRequest.builder().request_body(body).build()
        resp = self.client.approval.v4.instance.create(req)
        # resp.code == 0 表示成功；校验失败抛异常
        return resp.data.instance_code
```

### 7.4 数据库工具（结构化参数 + 白名单）

```python
# agent/tools/db_tools.py（设计示意）
from sqlalchemy import text
from core.database import async_engine

class DbToolExecutor:
    """
    数据库工具执行器：只接收结构化参数，禁止拼接 SQL。
    - table 必须命中白名单 AGENT_DB_ALLOWED_TABLES
    - 全部语句使用 SQLAlchemy Core / text() 绑定参数
    - SELECT 强制 LIMIT，防止拖库
    """

    async def query_data(self, table: str, columns: list[str], filters: dict, limit: int = 50) -> list[dict]:
        self._assert_table_allowed(table)                    # 白名单校验
        self._assert_columns_allowed(table, columns)         # 列名校验
        sql = text(f"SELECT {','.join(columns)} FROM `{table}` "
                   f"WHERE {self._build_where(filters)} LIMIT :limit")
        async with async_engine.connect() as conn:
            rows = (await conn.execute(sql, {**filters, "limit": limit})).mappings().all()
        return [dict(r) for r in rows]
```

### 7.5 飞书回调验签与解密

- 飞书事件订阅需校验 `verification_token` 与 `encrypt_key`（AES-256-CBC）；
- 回调路由收到事件后，先验签，再解密 `event` JSON，提取 `approval.instance` 事件中的 `instance_code` 与 `status`（`APPROVED` / `REJECTED` / `CANCELED`）；
- 用 `instance_code` 反查 `approval_request` 记录，幂等更新状态，触发图恢复；
- 回调失败重试机制：更新失败或图恢复失败时，记录日志并支持定时补偿任务（以 `approval_request.status` 为准）。

### 7.6 SSE 流式协议（新增接口）

沿用现有 `data: {...}\n\n` 分帧，扩展 `event` 字段区分消息类型：

```json
// 普通 token
data: {"event": "content", "content": "……"}

// 任务状态
data: {"event": "status", "status": "task_started"}
data: {"event": "status", "status": "approval_required", "approval_id": "xxx"}

// 心跳（审批等待期，防连接超时）
data: {"event": "ping"}

// 完成
data: {"event": "done"}
```

***

## 8. 数据表设计

### 8.1 task\_execution（任务执行记录，审计）

| 字段                        | 类型             | 说明                                                         |
| ------------------------- | -------------- | ---------------------------------------------------------- |
| id                        | varchar(36) PK | UUID                                                       |
| conversation\_id          | varchar(36)    | 关联会话                                                       |
| user\_id                  | varchar(36)    | 发起人                                                        |
| intent                    | varchar(64)    | 路由意图（task / knowledge\_base / task,knowledge\_base / chat） |
| status                    | varchar(32)    | running / completed / rejected / failed / canceled         |
| tool\_calls               | JSON           | 工具调用序列（名称、参数、结果摘要）                                         |
| rag\_answer               | text           | 知识库分支产出（并行场景快照，供审计/回溯）                                     |
| task\_answer              | text           | 任务分支产出（含审批后结果快照）                                           |
| result                    | text           | 最终回答（final\_response，含合并润色结果）                              |
| error                     | text           | 错误信息                                                       |
| created\_at / updated\_at | datetime       | 时间戳                                                        |

### 8.2 approval\_request（审批单）

| 字段                        | 类型             | 说明                                       |
| ------------------------- | -------------- | ---------------------------------------- |
| id                        | varchar(36) PK | UUID                                     |
| task\_execution\_id       | varchar(36)    | 关联任务                                     |
| conversation\_id          | varchar(36)    | 关联会话（= thread\_id）                       |
| requester\_id             | varchar(64)    | 发起人（飞书 user\_id/open\_id）                |
| approval\_type            | varchar(32)    | update / delete                          |
| target\_table             | varchar(64)    | 目标表                                      |
| tool\_params              | JSON           | 待执行工具参数（恢复时使用）                           |
| status                    | varchar(32)    | pending / approved / rejected / canceled |
| feishu\_instance\_code    | varchar(128)   | 飞书审批实例编码（回调关联）                           |
| approved\_by              | varchar(64)    | 审批人                                      |
| callback\_payload         | JSON           | 飞书回调原文（审计）                               |
| created\_at / updated\_at | datetime       | 时间戳                                      |

> 索引：`idx_appr_instance(instance_code)`、`idx_appr_conv(conversation_id, status)`。
> 新表由 `Base.metadata.create_all` 在启动时自动创建（沿用现有机制）。

***

## 9. API 设计

### 9.1 新增接口

| 方法   | 路径                                     | 说明                            | 鉴权          |
| ---- | -------------------------------------- | ----------------------------- | ----------- |
| POST | `/api/agent/chat`                      | 智能对话（意图路由 + 知识库/任务/聊天），SSE 流式 | JWT         |
| POST | `/api/approval/callback`               | 飞书审批事件回调（验签 + 解密）             | 飞书验签（无 JWT） |
| GET  | `/api/agent/tasks/{task_execution_id}` | 查询任务执行状态与结果（轮询兜底）             | JWT         |
| GET  | `/api/approvals/{approval_id}`         | 查询审批单状态                       | JWT         |

请求体（`/api/agent/chat`）与现有 `ChatRequest` 对齐：

```json
{
  "conversation_id": "xxx",
  "messages": [{"role": "user", "content": "把用户张三的账号删除"}],
  "model": "qwen-turbo",
  "stream": true
}
```

> 说明：不再需要 `use_rag` 布尔开关，意图由系统自动识别；旧接口 `/api/conversations/chat` 与 `use_rag` 参数**原样保留**，供存量前端使用。

### 9.2 保留接口（零改动）

- `/api/conversations/chat`（含 `use_rag`）
- `/api/conversations/*` 会话 CRUD、消息列表
- `/api/auth/*`、`/api/file/process`、`/api/metadata/*`、`/api/attachments/*`、`/api/cache/*`

***

## 10. 配置项设计

在 `core/config.py` 增加（`.env.dev` / 生产 `.env` 注入）：

```ini
# ---- 飞书审批 ----
FEISHU_APP_ID=cli_xxxxxxxx
FEISHU_APP_SECRET=xxxxxxxx
FEISHU_APPROVAL_CODE=xxxxxxxx            # 审批流编码（审批人自动路由配置在飞书侧）
FEISHU_ENCRYPT_KEY=xxxxxxxx              # 事件回调 AES 密钥
FEISHU_VERIFICATION_TOKEN=xxxxxxxx       # 事件回调验签 token
FEISHU_CALLBACK_URL=...                  # 事件订阅地址（可选，用于配置提示）

# ---- 任务执行 ----
AGENT_DB_ALLOWED_TABLES=user,conversation,conversation_message   # 表白名单，逗号分隔
AGENT_QUERY_MAX_ROWS=50                  # 单次查询最大返回行数
AGENT_TASK_TIMEOUT_SECONDS=120           # 任务执行超时
AGENT_INSERT_REQUIRES_APPROVAL=false     # 插入操作是否审批（默认否）
```

***

## 11. 安全设计

| 风险点         | 措施                                                     |
| ----------- | ------------------------------------------------------ |
| SQL 注入      | 工具仅接收结构化参数，语句全部绑定参数化执行；禁止 LLM 拼接 SQL                   |
| 越权访问数据      | 表/列白名单 + 权限过滤（复用现有 metadata.auth\_option 机制）+ 查询 LIMIT |
| 任务 Agent 失控 | 工具集最小化、每次调用审计落库、超时保护、审批门兜底                             |
| 飞书回调伪造      | 验签（verification\_token）+ 事件解密（encrypt\_key）+ 幂等状态守卫    |
| 提示词注入       | 意图识别与工具描述中显式声明"仅处理授权范围内的数据库操作"；知识库内容注入的防御沿用现有提示词分层隔离   |
| 敏感信息泄露      | 日志与审计中不记录完整参数值（脱敏），审批表单仅展示必要摘要                         |
| 并发/重复审批     | 审批单状态机（pending→approved/rejected 单向流转）+ 恢复时再次校验状态      |

***

## 12. 影响范围与改动清单

> 遵循**最小改动原则**：现有已稳定的模块（检索、嵌入、文件处理、认证等）**不重写、不改逻辑**，仅新增少量公开方法供图节点复用。

### 12.1 新增文件（无风险）

| 文件                                                | 内容                   |
| ------------------------------------------------- | -------------------- |
| `agent/**`                                        | 图状态、节点、工具、审批、知识服务适配层 |
| `models/task_model.py` / `task_schema.py`         | 任务表与 Schema          |
| `models/approval_model.py` / `approval_schema.py` | 审批表与 Schema          |
| `api/routes/agent.py` / `approval.py`             | 新接口路由                |
| `docs/lingxi_agent_langgraph_redesign.md`         | 本文档                  |

### 12.2 修改文件（改动极小，需评审确认）

| 文件                                | 改动点                                                                                                                                                                                                                       | 影响范围               |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ |
| `requirements.txt`                | 已改：新增 langgraph 生态、openai 3.x、lark-oapi；升级 langchain 系列                                                                                                                                                                   | 依赖安装（由用户执行）        |
| `core/config.py`                  | 新增飞书/任务执行配置项（纯新增字段，默认值兜底）                                                                                                                                                                                                 | 无行为影响              |
| `.env.dev`                        | 新增飞书配置占位                                                                                                                                                                                                                  | 仅本地开发              |
| `app/main.py`                     | lifespan 中初始化 Agent 图（含 MySQL 检查点）；注册 2 个新路由                                                                                                                                                                              | 启动流程扩展，失败降级不影响现有功能 |
| `prompt/prompt_storage.py`        | 新增意图识别/任务执行提示词（纯追加）                                                                                                                                                                                                       | 无                  |
| `rag/rag_conversation_service.py` | 新增 2\~3 个**公开薄方法**（如 `retrieve_context_public`、`get_compressed_history_public`），内部委托现有私有方法；移除 BM25 索引管理器初始化/重建，`rebuild_hybrid_index` 改为 `invalidate_kb_semantic_cache`（仅失效语义缓存），检索链路预计算稠密+稀疏双向量并全链路传递 | 检索行为不变，仅实现载体变化     |
| `rag/hybrid_retriever.py`         | 移除内存 `BM25Indexer`/持久化/同步机制，BM25 路改为 Qdrant **稀疏向量**查询（`query_points` + `using="sparse"`，查询侧与写入侧共用 text-embedding-v4 生成的稀疏向量，支持 `precomputed_sparse` 预计算复用）；RRF 双路融合与 Cross-Encoder 重排保留                                                                  | 检索行为不变，依赖更少        |
| `embeddings/embedding_deal.py`    | 集合同时配置**稠密 + 稀疏**向量（兼容存量无名稠密向量，增量补充 sparse 配置）；`save_to_vectors` 双向量写入，同 `doc_id` 先删后插（Qdrant 删点即删全部向量）；`DashScopeEmbedding` 基于 text-embedding-v4 一次调用双输出（`embed_documents_with_sparse` / `embed_query_with_sparse`）                                                          | 写入侧新增稀疏向量，检索侧无需感知  |
| `api/routes/file_process.py`      | 移除 BM25 索引重建步骤，改为知识库变更后失效语义缓存（失败仅告警不阻断）                                                                                                                                                                                   | 同 doc_id 自动先删后插    |
| `rag/memory_mysql.py`             | 如需可暴露历史加载公开方法（可选）                                                                                                                                                                                                         | 仅新增方法              |

### 12.3 明确不改动

- `rag/semantic_cache.py`、`rag/memory_mysql.py`、`ingestion/**`、`utils/**`（原样保留）
- `rag/hybrid_retriever.py`、`embeddings/embedding_deal.py` 已完成 **BM25→text-embedding-v4 稀疏向量**改造（见 §12.2）；`rag/bm25_index_manager.py` 已删除
- `api/routes/conversation.py`、`auth.py`、`metadata.py`、`attachments.py`、`cache.py`（旧接口原样保留）

***

## 13. 分阶段实施计划

| 阶段  | 内容                             | 产出                              | 验收标准                                       |
| --- | ------------------------------ | ------------------------------- | ------------------------------------------ |
| 0   | 用户按 requirements.txt 安装依赖      | 可运行环境                           | `pip install -r requirements.txt` 成功，服务可启动 |
| 1   | 意图识别 + 主图骨架（路由到知识库/聊天）         | `agent/` 基础模块、`/api/agent/chat` | 知识库问答与普通聊天流式正常，检索结果与旧接口一致                  |
| 2   | 任务执行子图 + 工具层（只读/插入）            | 工具注册表、db\_tools                 | 查询/插入任务可完成并出审计报告                           |
| 3   | 飞书审批（写操作拦截 + interrupt + 回调恢复） | 审批模块、审批表、回调接口                   | update/delete 触发审批，通过后执行、拒绝后终止，重启可恢复       |
| 4   | 观测与加固                          | LangSmith 追踪完善、日志、超时、幂等         | 全链路可追踪，异常可降级                               |

***

## 14. 风险与注意事项

1. **依赖兼容**：已通过 PyPI 版本核对锁定版本组合（langchain 1.3.10 + langgraph 1.2.10 + langchain-core 1.6.1），langgraph 1.2.11 与 langchain 1.4.0 组合不可用（yanked），后续升级需重新验证；
2. **LLM 意图误判**：采用结构化输出 + 低置信度降级 + 任务写操作双重确认（意图识别 + Agent 工具决策）降低误判概率；上线初期可开启意图结果日志抽查。多意图并行场景下，若误判为 `task+knowledge_base` 会导致一次额外检索/任务开销，通过意图归一化与置信度阈值控制，且单分支结果缺失时 merge 节点可容错输出；
   - **并行分支 + 审批中断（T3-9 已验证）**：LangGraph 中某节点触发 `GraphInterrupt` 会取消同超步中仍在执行的其他并行分支。`task+knowledge_base` 场景下任务分支先触发审批中断时，知识分支（检索 + LLM 生成较慢）会被取消，导致 `rag_answer` 缺失。实现上由 merge 节点在检测到「意图含 knowledge_base 但 rag_answer 为空」时重新执行知识分支以恢复结果（抑制流式、仅恢复文本），保证审批恢复后汇总完整。生产建议使用 Qdrant Server 模式（本地路径模式不允许多进程并发访问）。
3. **审批恢复依赖 MySQL 检查点**：`langgraph-checkpoint-mysql` 需独立的数据库表权限（自动建表）；生产环境建议与业务库隔离或单独 Schema；
4. **SSE 长连接**：审批等待期间连接保持，通过心跳保活；若连接断开，前端可轮询 `/api/agent/tasks/{id}` 兜底；
5. **飞书审批流配置**：审批人自动路由在飞书审批流管理后台配置，应用侧仅传入 `approval_code` 与表单数据；
6. **数据库写操作范围**：初期白名单表由配置控制，务必在生产环境收敛到最小集合；
7. **API 生命周期监控**：`langgraph.prebuilt.create_react_agent` 已在 LangGraph v1 弃用，本项目统一使用 `langchain.agents.create_agent`；LangChain/LangGraph 迭代较快，开发与升级时应以官方文档为准持续跟进（`langchain.agents.middleware` 中 HITL 相关类的构造方式以实际版本 API 为准）；
8. **存量数据稀疏向量为空（T 系列新增）**：改造前写入 Qdrant 的点仅含稠密向量，稀疏路（text-embedding-v4 关键词）对这些点召回为空。集合已增量补充 `sparse` 向量配置（`update_collection`），但存量点需**重新上传文档**（走 `process_file` 同 `doc_id` 先删后插）后才会生成稀疏向量；检索侧稀疏路为空时自动降级为仅稠密路，不影响服务可用性。
