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

    @property
    def agent_db_allowed_table_list(self) -> list[str]:
        """将逗号分隔的表白名单解析为列表（过滤空项）。"""
        return [t.strip() for t in self.agent_db_allowed_tables.split(",") if t.strip()]


settings = Settings()
