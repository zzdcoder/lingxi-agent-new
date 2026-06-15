# 项目重构建议

## 📋 概述

本文档记录了当前项目的结构问题和重构建议，旨在提升代码质量、可维护性和企业级规范。

## 🏗️ 后端重构建议

### 1. 目录结构优化

**当前结构**:
```
lingxi-agent-backend/
├── api/
├── app/
├── core/
├── embeddings/
├── ingestion/
├── llm/
├── models/
├── prompt/
├── rag/
├── scripts/
└── utils/
```

**建议结构**:
```
lingxi-agent-backend/
├── src/                          # 源代码根目录
│   ├── api/                      # API 层
│   │   ├── routes/               # 路由定义
│   │   ├── middleware/           # 中间件
│   │   └── dependencies/         # 依赖注入
│   ├── core/                     # 核心配置
│   │   ├── config.py             # 配置管理
│   │   ├── database.py           # 数据库连接
│   │   └── exceptions.py         # 异常定义
│   ├── models/                   # 数据模型
│   │   ├── orm/                  # ORM 模型
│   │   └── schemas/              # Pydantic Schema
│   ├── services/                 # 业务逻辑层
│   │   ├── conversation_service.py
│   │   ├── rag_service.py
│   │   └── file_service.py
│   ├── repositories/             # 数据访问层
│   │   ├── conversation_repo.py
│   │   └── file_repo.py
│   ├── integrations/             # 外部集成
│   │   ├── llm/                  # LLM 提供商
│   │   ├── vectorstore/          # 向量数据库
│   │   └── storage/              # 对象存储
│   └── utils/                    # 工具函数
├── tests/                        # 测试代码
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── migrations/                   # 数据库迁移
├── scripts/                      # 运维脚本
└── docs/                         # 文档
```

**优势**:
- ✅ 分层更清晰（API → Service → Repository）
- ✅ 便于单元测试
- ✅ 符合企业级项目规范

### 2. 配置管理重构

**当前问题**:
- `.env.dev` 中的敏感信息硬编码
- 配置分散在多个文件

**建议**:
```python
# core/config.py
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # 数据库配置
    db_host: str
    db_port: int = 3306
    db_name: str
    db_user: str
    db_password: str
    
    # LLM 配置
    llm_api_key: str
    llm_api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_model: str = "qwen-turbo"
    
    # ChromaDB 配置
    chroma_host: str = "localhost"
    chroma_port: int = 8399
    
    # JWT 配置
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }

settings = Settings()
```

**.env.example** (提交到 Git):
```env
# 数据库配置
DB_HOST=localhost
DB_PORT=3306
DB_NAME=lingxi_agent
DB_USER=root
DB_PASSWORD=

# LLM 配置
LLM_API_KEY=
LLM_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1

# JWT 配置
JWT_SECRET_KEY=change-this-to-a-random-string
```

**.env** (不提交到 Git):
```env
# 填入真实配置
DB_PASSWORD=your-password
LLM_API_KEY=your-api-key
JWT_SECRET_KEY=your-secret-key
```

### 3. 异常处理规范化

**当前问题**:
- 异常定义不完整
- 错误码不统一

**建议**:
```python
# core/exceptions.py
from fastapi import HTTPException
from typing import Optional

class BaseAppException(HTTPException):
    """应用基础异常"""
    def __init__(
        self,
        status_code: int = 500,
        message: str = "服务器内部错误",
        error_code: Optional[str] = None
    ):
        super().__init__(status_code=status_code, detail=message)
        self.error_code = error_code or f"ERR_{status_code}"

class ConversationException(BaseAppException):
    """对话相关异常"""
    def __init__(self, message: str, error_code: str = "ERR_CONVERSATION"):
        super().__init__(status_code=400, message=message, error_code=error_code)

class RAGException(BaseAppException):
    """RAG 相关异常"""
    def __init__(self, message: str, error_code: str = "ERR_RAG"):
        super().__init__(status_code=500, message=message, error_code=error_code)

class AuthenticationException(BaseAppException):
    """认证异常"""
    def __init__(self, message: str = "认证失败"):
        super().__init__(status_code=401, message=message, error_code="ERR_AUTH")
```

**使用示例**:
```python
@router.post("/chat")
async def chat(...):
    try:
        # 业务逻辑
        ...
    except ConversationException as e:
        logger.error(f"对话异常: {e.message}")
        raise
    except Exception as e:
        logger.exception("未捕获的异常")
        raise BaseAppException(message=f"服务器错误: {str(e)}")
```

### 4. 日志系统增强

**当前问题**:
- 日志格式不统一
- 缺少日志级别规范
- 没有日志轮转

**建议**:
```python
# core/logging_config.py
import logging
import logging.handlers
import os
from pathlib import Path

def setup_logging():
    """配置全局日志"""
    
    # 日志目录
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # 日志格式
    log_format = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # 根日志配置
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            # 控制台输出
            logging.StreamHandler(),
            # 文件输出（带轮转）
            logging.handlers.RotatingFileHandler(
                filename=log_dir / "app.log",
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
                encoding="utf-8"
            ),
            # 错误日志
            logging.handlers.RotatingFileHandler(
                filename=log_dir / "error.log",
                level=logging.ERROR,
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8"
            )
        ]
    )
    
    # 设置第三方库日志级别
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.INFO)

# 在 main.py 中调用
setup_logging()
```

**日志级别规范**:
- `DEBUG`: 调试信息（开发环境）
- `INFO`: 关键操作成功（创建会话、对话完成）
- `WARNING`: 潜在问题（检索为空、配置缺失）
- `ERROR`: 操作失败（数据库错误、LLM 调用失败）
- `CRITICAL`: 系统级错误（服务启动失败）

### 5. 依赖注入优化

**当前问题**:
- 服务实例化分散
- 单例模式实现不统一

**建议**:
```python
# api/dependencies/services.py
from functools import lru_cache
from sqlalchemy.ext.asyncio import AsyncSession

from services.conversation_service import ConversationService
from services.rag_service import RAGConversationService

@lru_cache()
def get_conversation_service_factory():
    """获取对话服务工厂（缓存）"""
    return ConversationService

async def get_conversation_service(db: AsyncSession) -> ConversationService:
    """对话服务依赖注入"""
    return ConversationService(db_session=db)

async def get_rag_conversation_service(db: AsyncSession) -> RAGConversationService:
    """RAG 对话服务依赖注入"""
    return RAGConversationService(db_session=db)
```

**路由中使用**:
```python
from api.dependencies.services import get_conversation_service

@router.post("/chat")
async def chat(
    chat_request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    service: ConversationService = Depends(get_conversation_service)
):
    # 直接使用 service
    ...
```

### 6. 数据库迁移工具

**当前问题**:
- 手动执行 SQL 脚本
- 没有版本管理

**建议**: 使用 Alembic

```bash
# 安装
pip install alembic

# 初始化
alembic init migrations

# 配置 alembic.ini
sqlalchemy.url = mysql+aiomysql://root:123456@localhost/lingxi_agent

# 创建迁移
alembic revision --autogenerate -m "add conversation tables"

# 执行迁移
alembic upgrade head

# 回滚
alembic downgrade -1
```

## 🎨 前端重构建议

### 1. 状态管理

**当前问题**:
- 使用多个 useState，状态分散
- 缺少全局状态管理

**建议**: 使用 Zustand 或 Redux Toolkit

```typescript
// store/conversationStore.ts
import { create } from 'zustand';
import { listConversations, createConversation, deleteConversation } from '../services/llm';

interface Conversation {
  id: string;
  title: string;
  status: string;
  // ...
}

interface ConversationState {
  conversations: Conversation[];
  activeConversationId: string | null;
  isLoading: boolean;
  
  // Actions
  loadConversations: () => Promise<void>;
  createNewConversation: (title: string) => Promise<void>;
  deleteConversation: (id: string) => Promise<void>;
  setActiveConversation: (id: string) => void;
}

export const useConversationStore = create<ConversationState>((set, get) => ({
  conversations: [],
  activeConversationId: null,
  isLoading: false,
  
  loadConversations: async () => {
    set({ isLoading: true });
    try {
      const conversations = await listConversations();
      set({ conversations, isLoading: false });
    } catch (error) {
      set({ isLoading: false });
      console.error('加载会话失败:', error);
    }
  },
  
  createNewConversation: async (title: string) => {
    const conversation = await createConversation(title, 'qwen-turbo');
    set((state) => ({
      conversations: [conversation, ...state.conversations],
      activeConversationId: conversation.id,
    }));
  },
  
  deleteConversation: async (id: string) => {
    await deleteConversation(id);
    set((state) => ({
      conversations: state.conversations.filter(c => c.id !== id),
    }));
  },
  
  setActiveConversation: (id: string) => {
    set({ activeConversationId: id });
  },
}));
```

### 2. API 请求封装

**当前问题**:
- fetch 调用分散
- 缺少统一的错误处理
- 没有请求重试机制

**建议**: 使用 Axios + React Query

```typescript
// lib/api.ts
import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
  timeout: 30000,
});

// 请求拦截器
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('access_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// 响应拦截器
api.interceptors.response.use(
  (response) => response.data,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('access_token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export default api;

// hooks/useConversations.ts
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import api from '../lib/api';

export function useConversations() {
  const queryClient = useQueryClient();
  
  return useQuery({
    queryKey: ['conversations'],
    queryFn: () => api.get('/conversations'),
  });
}

export function useCreateConversation() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (title: string) => 
      api.post('/conversations', null, { params: { title } }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['conversations'] });
    },
  });
}
```

### 3. 组件拆分

**当前问题**:
- 组件过大（如 `KnowledgeBaseCreatePage.tsx` 92KB）
- 职责不清晰

**建议**: 拆分为更小的组件

```
components/
├── conversation/
│   ├── ConversationList.tsx
│   ├── ConversationItem.tsx
│   ├── MessageList.tsx
│   ├── MessageBubble.tsx
│   ├── ChatInput.tsx
│   └── ThinkingIndicator.tsx
├── layout/
│   ├── Sidebar.tsx
│   ├── Header.tsx
│   └── MainContent.tsx
└── common/
    ├── Button.tsx
    ├── Input.tsx
    ├── Modal.tsx
    └── Loading.tsx
```

### 4. 类型安全增强

**建议**: 使用更严格的 TypeScript 配置

```json
// tsconfig.json
{
  "compilerOptions": {
    "strict": true,
    "noImplicitAny": true,
    "strictNullChecks": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true
  }
}
```

## 🧪 测试建议

### 1. 后端测试

```python
# tests/test_conversation.py
import pytest
from httpx import AsyncClient
from app.main import app

@pytest.mark.asyncio
async def test_create_conversation():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/conversations",
            params={"title": "测试对话", "model": "qwen-turbo"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "测试对话"

@pytest.mark.asyncio
async def test_chat_stream():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/conversations/chat",
            json={
                "messages": [{"role": "user", "content": "你好"}],
                "model": "qwen-turbo",
                "conversation_id": "test-id",
                "use_rag": False
            }
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
```

### 2. 前端测试

```typescript
// __tests__/ConversationList.test.tsx
import { render, screen, waitFor } from '@testing-library/react';
import { ConversationList } from '../components/conversation/ConversationList';

test('renders conversations', async () => {
  render(<ConversationList />);
  
  await waitFor(() => {
    expect(screen.getByText('新对话')).toBeInTheDocument();
  });
});
```

## 📊 性能优化建议

### 1. 数据库查询优化

**添加索引**:
```sql
-- 已经在模型中定义，确保已创建
CREATE INDEX idx_conv_user_id_status ON conversation(user_id, status);
CREATE INDEX idx_msg_conv_id_seq ON conversation_message(conversation_id, sequence_number);
```

**查询优化**:
```python
# 使用分页
async def list_conversations(
    self,
    user_id: Optional[str],
    offset: int = 0,
    limit: int = 50
) -> List[ConversationDefinition]:
    stmt = (
        select(ConversationDefinition)
        .where(...)
        .order_by(ConversationDefinition.updated_at.desc())
        .offset(offset)
        .limit(limit)
    )
```

### 2. 缓存策略

**建议**: 使用 Redis 缓存热点数据

```python
import redis.asyncio as redis
from functools import lru_cache

redis_client = redis.Redis(host='localhost', port=6379, db=0)

async def get_conversation_cache(conversation_id: str):
    """从缓存获取会话"""
    cache_key = f"conversation:{conversation_id}"
    
    # 尝试从缓存获取
    cached = await redis_client.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # 从数据库查询
    conversation = await db.get(ConversationDefinition, conversation_id)
    
    # 写入缓存（5分钟过期）
    await redis_client.setex(
        cache_key,
        300,
        json.dumps(conversation)
    )
    
    return conversation
```

## 📝 代码规范

### 1. Python 代码规范

使用 Black + isort + flake8:

```bash
# 安装
pip install black isort flake8

# 格式化代码
black .
isort .

# 检查代码质量
flake8 .
```

**pyproject.toml**:
```toml
[tool.black]
line-length = 88
target-version = ['py310']

[tool.isort]
profile = "black"
multi_line_output = 3
```

### 2. TypeScript 代码规范

使用 ESLint + Prettier:

```bash
# 安装
npm install --save-dev eslint prettier eslint-config-prettier

# package.json scripts
{
  "scripts": {
    "lint": "eslint src --ext .ts,.tsx",
    "format": "prettier --write src/**/*.{ts,tsx}"
  }
}
```

## 🎯 优先重构清单

### 高优先级（立即执行）
1. ✅ 添加 `.env.example` 文件
2. ✅ 统一异常处理
3. ✅ 增强日志系统
4. ✅ 前端 API 请求封装

### 中优先级（1-2 周内）
1. 引入 Alembic 管理数据库迁移
2. 前端引入状态管理（Zustand）
3. 拆分大组件
4. 添加单元测试

### 低优先级（1 个月内）
1. 重构目录结构
2. 引入 Redis 缓存
3. 添加 API 文档（Swagger 增强）
4. CI/CD 流水线

## 📚 参考资料

- [FastAPI Best Practices](https://github.com/zhanymkanov/fastapi-best-practices)
- [React Query Documentation](https://tanstack.com/query/latest)
- [Zustand Documentation](https://zustand-demo.pmnd.rs/)
- [Alembic Tutorial](https://alembic.sqlalchemy.org/en/latest/tutorial.html)
- [Python Logging Cookbook](https://docs.python.org/3/howto/logging-cookbook.html)

---

**创建时间**: 2024-01-01  
**版本**: v1.0.0  
**状态**: 待评审
