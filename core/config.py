"""
应用配置

使用 Pydantic Settings 实现类型安全的配置管理
所有敏感信息通过环境变量注入
开发环境可复用 .env.dev，生产环境使用 .env.pro
"""

import os
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env.dev" if os.path.exists(".env.dev") else ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 数据库配置
    db_host: str = "localhost"
    db_port: int = 3306
    db_name: str = "lingxi_agent"
    db_user: str = "root"
    db_password: str = "123456"

    # JWT 认证配置
    jwt_secret_key: str = "lingxi-default-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expires_hours: int = 24

    # 验证码配置
    captcha_expires_seconds: int = 120

    # 腾讯云 COS 配置（密钥经 .env.dev / 环境变量注入，禁止硬编码）
    cos_secret_id: str = ""
    cos_secret_key: str = ""
    cos_region: str = "ap-chengdu"
    cos_bucket: str = "lingxi-agent-persistence-1314815866"
    cos_domain: str = "https://lingxi-agent-persistence-1314815866.cos.ap-chengdu.myqcloud.com"

    # 文件处理配置
    file_max_size: int = 52428800  # 50MB
    file_chunk_size: int = 1000  # 默认分块大小
    file_chunk_overlap: int = 200  # 默认分块重叠大小
    file_default_separators: str = "\n\n,\n"  # 默认分隔符
    file_temp_dir: str = "./temp"  # 临时文件目录

    # 通义千问 DashScope API Key（经 .env.dev 的 OPENAI_API_KEY 注入，禁止硬编码）
    api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")

    # Cross-Encoder 重排序配置
    reranker_model: str = "./bge-reranker-base"  # 本地路径
    reranker_device: str = ""  # 运行设备: cuda / mps / cpu，空字符串自动选择
    reranker_max_length: int = 512  # 输入最大 token 长度
    reranker_batch_size: int = 16  # 推理批大小

    # 语义缓存配置（基于 Qdrant）
    cache_enabled: bool = True
    cache_collection_name: str = "lingxi-cache"  # Qdrant 缓存集合名
    cache_similarity_threshold: float = 0.92  # 余弦相似度阈值（越高越严格）
    cache_ttl_seconds: int = 86400  # 缓存过期时间 24h
    cache_max_entries: int = 10000  # 最大缓存条目数

    # 应用配置
    debug: bool = False

    # ============================================================
    # 前台审批（Agent 任务写操作人工审批，设计文档 §5.6）
    # ============================================================
    approval_wait_timeout: int = 7200             # 审批等待超时（秒），超时后 SSE 结束等待、轮询兜底

    # ============================================================
    # 任务执行（Agent 数据库工具）配置
    # ============================================================
    agent_db_allowed_tables: str = "user,conversation,conversation_message"  # 表白名单，逗号分隔
    agent_query_max_rows: int = 50                 # 单次查询最大返回行数
    agent_task_timeout_seconds: int = 120          # 任务执行超时（秒），非审批模式下 Agent 循环整体超时
    agent_insert_requires_approval: bool = False   # 插入操作是否触发审批（默认否）
    agent_tool_timeout_seconds: int = 30           # 单次工具调用总时长上限（秒），统一包裹所有工具
    agent_llm_timeout_seconds: int = 60            # 单次 LLM 推理超时（秒）

    # ============================================================
    # 工具注册（启动时扫描 @tool 装饰器工具并同步工具注册表
    # ============================================================
    agent_tool_scan_packages: str = "tools"        # 扫描 @tool 装饰器工具的包（逗号分隔，相对项目根目录）
    tool_registry_sync_on_start: bool = True       # 启动时是否自动同步工具注册表（失败降级不阻塞启动）

    # ============================================================
    # 工具熔断器
    # ============================================================
    agent_circuit_failure_threshold: int = 5       # 连续失败阈值：连续失败达此值触发熔断（status=2）
    agent_circuit_failure_ratio: float = 0.5       # 时间窗口失败率阈值：窗口内失败率超过且达到最小调用量时熔断
    agent_circuit_min_calls: int = 10              # 失败率判定所需的最小窗口调用量（防小样本误熔断）
    agent_circuit_window_seconds: int = 60         # 失败率统计窗口（秒）
    agent_circuit_cooldown_seconds: int = 60       # 熔断冷却期（秒），到期后进入半开探测
    agent_circuit_half_open_max_trials: int = 3    # 半开探测最大放行次数（成功即恢复，失败即重新熔断）
    agent_circuit_enabled: bool = True             # 熔断器总开关（false 时仅统计不熔断）
    agent_circuit_flush_interval_seconds: float = 1.0  # 熔断计数批量落库间隔（秒）：后台合并写回，降低每调用一次 DB 写
    agent_probe_enabled: bool = True               # 熔断恢复定时探测总开关（false 时熔断工具只能人工恢复）
    agent_probe_interval_seconds: int = 300        # 熔断工具扫描探测间隔（秒），默认每 5 分钟一轮
    agent_probe_timeout_seconds: float = 30.0      # 单个工具探测调用的超时上限（秒）
    agent_probe_exclude_tools: str = "ask_user"    # 定时探测排除的工具名（逗号分隔）：交互式工具重放会再次触发 interrupt，禁止自动重放

    # ============================================================
    # 观测与加固（阶段 4，设计文档 §15）
    # ============================================================
    trace_id_header: str = "X-Trace-Id"            # 响应头中的链路追踪 ID 字段名
    agent_metrics_enabled: bool = True             # 进程内指标采集总开关
    langsmith_tracing: bool = False                # LangSmith 链路追踪开关（需配合 LANGSMITH_API_KEY 环境变量）
    langsmith_project: str = "lingxi-agent"        # LangSmith 项目名（控制台过滤维度）
    langsmith_endpoint: str = "https://api.smith.langchain.com"  # LangSmith 服务端点（自建填自托管地址）

    # 超时体系（三层工具/LLM/任务 之外的补全：整体请求 / 汇聚 LLM / 恢复执行）
    agent_request_timeout_seconds: int = 180       # 单次 /api/agent/chat 主图执行整体超时（秒）
    agent_merge_timeout_seconds: int = 60          # merge 汇总节点 LLM 调用超时（秒）
    agent_merge_llm_retries: int = 1               # merge 汇总 LLM 失败重试次数（不含首次）
    agent_resume_timeout_seconds: int = 300        # 审批/追问恢复图执行的整体超时（秒）
    agent_graph_recursion_limit: int = 100         # 主图 / 任务 Agent 递归步数上限（防死循环）

    # SSE 背压与丢帧保护（防慢客户端拖垮服务端）
    agent_sse_queue_maxsize: int = 1000            # SSE 事件队列容量（0 表示无界，不推荐）
    agent_sse_put_timeout_seconds: float = 2.0     # 队列满时投递的最长等待时间（超时丢弃并记录）

    # ============================================================
    # LLM 思考过程与任务流式输出（设计文档 §21）
    # ============================================================
    # 统一 LLM 服务地址（收敛此前散落在 11 处 ChatOpenAI 实例化里的硬编码字面量）
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    agent_reasoning_enabled: bool = True           # 思考过程总开关（关闭后不请求也不推送）
    # 支持思考过程的模型前缀白名单（逗号分隔，前缀匹配）。
    # 只放白名单是刻意的：非思考模型传 enable_thinking 可能直接报错，
    # 宁可不传（退回改造前行为），也不要为少数模型破坏多数请求。
    agent_reasoning_models: str = "qwen3,qwen-plus,qwen-max,deepseek-r1,glm-4.5"
    agent_reasoning_max_chars: int = 20000         # 单轮思考过程推送上限（防超长思考刷屏/撑爆前端）

    # 任务分支流式输出（task_node 由 ainvoke 改为 astream）
    agent_task_streaming_enabled: bool = True      # 任务分支是否流式输出（关闭则退回一次性整段推送）
    agent_task_tool_events_enabled: bool = True    # 是否推送工具调用进度事件（tool 帧）

    @property
    def agent_reasoning_model_prefixes(self) -> list[str]:
        """将思考模型白名单解析为小写前缀列表（过滤空项）。"""
        return [
            p.strip().lower()
            for p in (self.agent_reasoning_models or "").split(",")
            if p.strip()
        ]

    # ============================================================
    # 意图识别分层优化（设计文档 §16）
    # ============================================================
    intent_gate_enabled: bool = True               # 规则快速通道总开关（关闭后全部走 LLM 分类）
    intent_slash_enabled: bool = True              # 斜杠命令本地处理总开关（关闭后当普通文本走 LLM）
    intent_slash_shortcut: bool = True             # 斜杠命令是否短路图执行（False 则仍进图但跳过意图 LLM）
    intent_trivial_keywords: str = ""              # 寒暄关键词表覆盖（空串表示用 intent_gate 默认表）
    intent_escalation_model: str = "qwen-plus"     # 低置信度升级重判模型（空串或与主模型相同则跳过）
    intent_llm_cache_confidence: float = 0.85      # LLM 结果进入进程内缓存的置信度门槛
    intent_embed_threshold: float = 0.86           # 向量就近判定阈值（低于则回落 LLM）

    # ============================================================
    # 性能优化（设计文档 §17）
    # ============================================================
    intent_llm_timeout_seconds: int = 15           # 意图判别单独超时（秒）。与 agent_llm_timeout_seconds
                                                   # 解耦：意图只需输出极短结构化对象，等 60s 无意义，
                                                   # 超时即降级 chat。设回 60 即恢复旧行为（§17.3）
    intent_llm_max_tokens: int = 1024              # 意图判别输出上限（§19.1）。
                                                   # 注意：qwen 思考型模型会先消耗 reasoning_tokens 再产出
                                                   # 正文，256 会被思考过程吃满导致 finish_reason=length，
                                                   # 结构化输出解析抛 LengthFinishReasonError。1024 为思考留足余量。
    intent_llm_disable_thinking: bool = True       # 意图判别是否关闭模型思考模式（§19.1）。
                                                   # 分类任务无需长链推理，关掉可省掉全部 reasoning 开销
                                                   # （既避免撑爆 max_tokens，也显著降低首字延迟）。
                                                   # 不支持该参数的模型会忽略它；若出现兼容问题置 False。
    intent_list_patterns_enabled: bool = True      # 「有哪些X / 列出X / X清单」句式强信号开关（§17.3）
    intent_proto_warmup_enabled: bool = True       # 启动时预热原型向量（§17.3）
    agent_merge_cross_turn_guard: bool = True      # 跨轮状态隔离：merge 按本轮分支标记判定，
                                                   # 忽略检查点里的历史残留（§17.2）。
                                                   # False 则回落「字段非空」旧判据
    agent_db_table_name_fuzzy: bool = True         # DB 工具表名单复数保守归一（§17.4）

    # ============================================================
    # 缓存治理：语义缓存 × 意图缓存（设计文档 §18）
    # ============================================================
    # 答案缓存阈值独立于 cache_similarity_threshold：答案缓存一旦命中就
    # **跳过整条执行链路**（不再检索、不再生成），因此对相似度要求更严；
    # 0.92 对短中文文本过松（「删除张三」与「删除李四」余弦常在 0.95+）。
    cache_answer_threshold: float = 0.95           # 答案缓存命中阈值（§18.3）
    cache_intent_policy_enabled: bool = True       # 意图驱动的缓存准入开关（§18.2）
                                                   # False 时回到「所有意图都可缓存」旧行为
    cache_entity_check_enabled: bool = True        # 命中实体一致性校验（§18.3）
    cache_ttl_knowledge_seconds: int = 86400       # knowledge_base 单分支答案缓存 TTL
    cache_ttl_short_seconds: int = 900             # chat / 时效类短 TTL

    # 缓存维度版本：任一 bump 后旧条目**自动失配**（逻辑失效，无需删数据）
    cache_schema_version: str = "1"                # 缓存结构版本（字段变更时 bump）
    cache_prompt_version: str = "1"                # 提示词版本（RAG/系统提示变更时 bump）
    cache_kb_version: str = "v1"                   # 知识库内容版本（§18 P1 改为真实指纹前，
                                                   # 由文档变更处显式 bump）

    cache_embed_prefetch_enabled: bool = True      # 入口预计算 query 双向量并复用（§18.5）
                                                   # 修复 L2 向量层拿不到 embedding 的缺陷

    # ---- §18.13 检索缓存层（P1-2）：缓存检索结果，命中后仍照常走 LLM 生成 ----
    cache_retrieve_enabled: bool = True            # 检索缓存总开关（False 即回到无检索缓存行为）
    cache_retrieve_threshold: float = 0.90         # 检索缓存命中阈值（宽松档：错了只是多检索一次）
    cache_retrieve_collection_name: str = "lingxi-retrieve-cache"  # 独立集合，不与答案缓存混用
    cache_retrieve_ttl_seconds: int = 86400        # 检索结果 TTL（与知识库答案缓存同档）
    cache_retrieve_max_entries: int = 5000         # 独立容量上限（条目比答案缓存大，故更保守）
    cache_retrieve_max_doc_chars: int = 1200       # 单文档 page_content 上限，超出则**不缓存**
                                                   # （宁可不缓存，也不存被截断的上下文）
    cache_retrieve_max_docs: int = 8               # 单条目最多缓存文档数

    # ---- §19 嵌套图检查点隔离 ----
    agent_task_thread_isolate: bool = True         # 任务 Agent（内层图）使用独立 thread_id，
                                                   # 不再与主图共用 conversation_id 检查点。
                                                   # False → 回到「内层图读写主图检查点」旧行为
                                                   # （跨轮串味：本轮 LLM 看到上一轮的问题）

    # ---- §23 任务 Agent 跨轮上下文注入 ----
    agent_task_history_inject: bool = True         # 任务 Agent 是否注入最近历史对话，
                                                   # 供模型消解「这个用户 / 刚才那个」等指代。
                                                   # 内层 thread 隔离后内层图拿不到任何历史，
                                                   # 关闭 → 回到改动前行为（模型可能重复追问
                                                   # 用户已在上一轮给出的实体）。
    agent_task_history_recent_num: int = 10        # 注入的历史条数上限。任务场景的实体通常
                                                   # 出现在紧邻上一轮，10 条足够；再多会被
                                                   # 内层的大段工具结果挤爆上下文预算。

    # ---- §20 收尾补推 ----
    agent_push_final_response: bool = True         # finalize 节点是否补推 final_response。
                                                   # merge 合并后的最终回答只在该节点产生，
                                                   # 原实现不推送 → 降级路径下前端收不到任何内容。
                                                   # False → 回到改动前行为（不补推）

    # ---- §20 删除策略（delete_data） ----
    agent_delete_mode: str = "logical"             # physical（真删）| logical（标记删除）
                                                   # 用户答复「逻辑删除合理」→ 默认 logical。
                                                   # 需与 agent_delete_flag_column 配合。
    agent_delete_flag_column: str = ""             # §20 F3.4b：逻辑删除标记列名（仅 logical 模式使用）。
                                                   # **留空 = 让工具按候选集自动探测**
                                                   # （deleted / is_deleted / delete_flag / del_flag…）。
                                                   # 曾默认写死 "is_deleted"，而 `user` 表真实列名是
                                                   # `deleted`（SHOW COLUMNS 确认）→ 逻辑删除静默回落
                                                   # 物理删除。故改为「显式配置优先、探测兜底」。
                                                   # 显式配置但该列不存在时也会转入探测并告警。
    agent_delete_flag_true_value: int = 1          # 标记「已删除」时写入的值（0/1 或时间戳均可配置）

    @property
    def agent_db_allowed_table_list(self) -> list[str]:
        """将逗号分隔的表白名单解析为列表（过滤空项）。"""
        return [t.strip() for t in self.agent_db_allowed_tables.split(",") if t.strip()]


settings = Settings()
