"""
应用配置

使用 Pydantic Settings 实现类型安全的配置管理
所有敏感信息通过环境变量注入
开发环境可复用 .env.dev，生产环境使用 .env.pro
"""

import os
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

    # 腾讯云 COS 配置
    cos_secret_id: str = "***REMOVED***"
    cos_secret_key: str = "***REMOVED***"
    cos_region: str = "ap-chengdu"
    cos_bucket: str = "lingxi-agent-persistence-1314815866"
    cos_domain: str = "https://lingxi-agent-persistence-1314815866.cos.ap-chengdu.myqcloud.com"

    # 文件处理配置
    file_max_size: int = 52428800  # 50MB
    file_chunk_size: int = 1000  # 默认分块大小
    file_chunk_overlap: int = 200  # 默认分块重叠大小
    file_default_separators: str = "\n\n,\n"  # 默认分隔符
    file_temp_dir: str = "./temp"  # 临时文件目录

    api_key :str ="***REMOVED***"

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
    # 飞书审批配置（Agent 任务写操作人工审批）
    # 未配置时相关功能自动降级：写操作拒绝并提示"审批未配置"
    # ============================================================
    feishu_app_id: str = ""                        # 飞书应用 App ID
    feishu_app_secret: str = ""                    # 飞书应用 App Secret
    feishu_approval_code: str = ""                 # 审批流编码（审批人/角色在飞书后台配置，自动路由）
    feishu_encrypt_key: str = ""                   # 事件回调 AES 加密密钥
    feishu_verification_token: str = ""            # 事件回调验签 token
    feishu_callback_url: str = ""                  # 事件订阅地址（可选，仅用于配置提示）

    # ============================================================
    # 任务执行（Agent 数据库工具）配置
    # ============================================================
    agent_db_allowed_tables: str = "user,conversation,conversation_message"  # 表白名单，逗号分隔
    agent_query_max_rows: int = 50                 # 单次查询最大返回行数
    agent_task_timeout_seconds: int = 120          # 任务执行超时（秒）
    agent_insert_requires_approval: bool = False   # 插入操作是否触发审批（默认否）

    @property
    def agent_db_allowed_table_list(self) -> list[str]:
        """将逗号分隔的表白名单解析为列表（过滤空项）。"""
        return [t.strip() for t in self.agent_db_allowed_tables.split(",") if t.strip()]


settings = Settings()
