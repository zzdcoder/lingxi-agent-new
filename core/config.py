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
    # 意图识别分层优化（设计文档 §16）
    # ============================================================
    intent_gate_enabled: bool = True               # 规则快速通道总开关（关闭后全部走 LLM 分类）
    intent_slash_enabled: bool = True              # 斜杠命令本地处理总开关（关闭后当普通文本走 LLM）
    intent_slash_shortcut: bool = True             # 斜杠命令是否短路图执行（False 则仍进图但跳过意图 LLM）
    intent_trivial_keywords: str = ""              # 寒暄关键词表覆盖（空串表示用 intent_gate 默认表）
    intent_escalation_model: str = "qwen-plus"     # 低置信度升级重判模型（空串或与主模型相同则跳过）
    intent_llm_cache_confidence: float = 0.85      # LLM 结果进入进程内缓存的置信度门槛
    intent_embed_threshold: float = 0.86           # 向量就近判定阈值（低于则回落 LLM）

    @property
    def agent_db_allowed_table_list(self) -> list[str]:
        """将逗号分隔的表白名单解析为列表（过滤空项）。"""
        return [t.strip() for t in self.agent_db_allowed_tables.split(",") if t.strip()]


settings = Settings()
