# 灵犀智能助手 - 对话功能开发文档

## 📋 项目概述

本次开发完成了基于 MySQL Memory 和 RAG 知识检索的企业级对话系统，包括：
- ✅ 后端对话服务（带 Memory）
- ✅ RAG 知识检索集成
- ✅ SSE 流式响应
- ✅ 前端对接指南
- ✅ 企业级代码规范

## 🏗️ 架构设计

### 后端架构

```
┌─────────────────────────────────────────────────────┐
│                   FastAPI Application                │
├─────────────────────────────────────────────────────┤
│  API Routes (api/routes/conversation.py)            │
│  ├── GET    /api/conversations          (列表)      │
│  ├── POST   /api/conversations          (创建)      │
│  ├── DELETE /api/conversations/{id}     (删除)      │
│  ├── GET    /api/conversations/{id}/messages (消息) │
│  └── POST   /api/conversations/chat     (对话)      │
├─────────────────────────────────────────────────────┤
│  Service Layer                                       │
│  ├── ConversationService (普通对话)                 │
│  └── RAGConversationService (RAG 对话)              │
├─────────────────────────────────────────────────────┤
│  Memory Layer                                        │
│  ├── MySQLChatMessageHistory (MySQL Memory)         │
│  └── MySQLConversationSummaryMemory (摘要管理)      │
├─────────────────────────────────────────────────────┤
│  Data Layer                                          │
│  ├── MySQL (会话和消息存储)                         │
│  └── ChromaDB (向量数据库)                          │
└─────────────────────────────────────────────────────┘
```

### 数据库表结构

#### 1. `conversation` - 会话表
```sql
- id (VARCHAR 36, PK)
- user_id (VARCHAR 36, 可选)
- title (VARCHAR 255)
- status (VARCHAR 32): active/archived/deleted
- deleted (INT): 0/1
- created_at (DATETIME)
- updated_at (DATETIME)
```

#### 2. `conversation_message` - 消息表
```sql
- id (VARCHAR 36, PK)
- conversation_id (VARCHAR 36, FK)
- message_type (VARCHAR 32): human/ai/system
- content (TEXT)
- additional_kwargs (JSON)
- sequence_number (INT)
- token_count (INT)
- created_at (DATETIME)
```

#### 3. `conversation_summary` - 摘要表
```sql
- id (VARCHAR 36, PK)
- conversation_id (VARCHAR 36, UNIQUE)
- summary (TEXT)
- message_count (INT)
- last_message_id (VARCHAR 36)
- summary_token_count (INT)
- created_at (DATETIME)
- updated_at (DATETIME)
```

## 🚀 快速开始

### 1. 数据库迁移

```bash
cd d:\lingxi-agent-backend

# 创建表
python scripts/migrate_conversation.py --action create

# 验证表创建
mysql -u root -p lingxi_agent -e "SHOW TABLES;"
```

### 2. 启动后端服务

```bash
# 安装依赖（如果还没有）
pip install -r requirements.txt

# 启动服务
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

访问 API 文档: http://localhost:8000/docs

### 3. 启动前端服务

```bash
cd d:\lingxi-agent-portal

# 安装依赖
npm install

# 启动开发服务器
npm run dev
```

访问前端: http://localhost:5173

## 📝 API 接口文档

### 1. 获取会话列表

**请求**:
```http
GET /api/conversations
Authorization: Bearer <token>
```

**响应**:
```json
[
  {
    "id": "uuid",
    "user_id": "user-uuid",
    "title": "新对话",
    "status": "active",
    "created_at": "2024-01-01T00:00:00",
    "updated_at": "2024-01-01T00:00:00"
  }
]
```

### 2. 创建会话

**请求**:
```http
POST /api/conversations?title=新对话&model=qwen-turbo
Authorization: Bearer <token>
```

**响应**:
```json
{
  "id": "uuid",
  "title": "新对话",
  "status": "active",
  "created_at": "2024-01-01T00:00:00",
  "updated_at": "2024-01-01T00:00:00"
}
```

### 3. 获取消息历史

**请求**:
```http
GET /api/conversations/{conversation_id}/messages
Authorization: Bearer <token>
```

**响应**:
```json
[
  {
    "id": "uuid",
    "conversation_id": "uuid",
    "message_type": "human",
    "content": "你好",
    "sequence_number": 1,
    "created_at": "2024-01-01T00:00:00"
  }
]
```

### 4. 流式对话（核心接口）

**请求**:
```http
POST /api/conversations/chat
Authorization: Bearer <token>
Content-Type: application/json

{
  "messages": [
    {"role": "user", "content": "你好"}
  ],
  "model": "qwen-turbo",
  "conversation_id": "uuid",
  "stream": true,
  "use_rag": true
}
```

**响应** (SSE 流):
```
data: {"content": "你"}

data: {"content": "好"}

data: {"content": "！"}

data: {"done": true}
```

## 🔧 前端对接

### 修改文件清单

由于前端项目不在当前工作区，请手动修改以下文件：

1. **`src/services/llm.ts`** - 重构服务层
   - 移除 Mock 实现
   - 更新 API 路径：`/api/chat` → `/api/conversations/chat`
   - 更新请求参数格式

2. **`src/hooks/useChat.ts`** - 更新消息映射
   - 修改 `mapBackendMessage` 函数
   - 适配新的消息类型：`message_type: 'human' | 'ai' | 'system'`

详细修改指南请参考: [前端重构指南](./frontend_refactor_guide.md)

### 关键代码片段

#### 流式对话调用示例

```typescript
import { streamChat } from './services/llm';

const stream = streamChat(
  messages,        // 消息列表
  'qwen-turbo',    // 模型
  conversationId,  // 会话ID
  {
    onThinking: (text) => {
      // 处理思考过程
      setThinkingContent(text);
    },
    onContent: (text) => {
      // 处理回答内容
      setContent(text);
    },
    onDone: () => {
      // 对话完成
      setIsLoading(false);
    },
    onError: (err) => {
      // 错误处理
      setError(err.message);
    },
  }
);

// 可以中断对话
stream.abort();
```

## 📊 核心功能说明

### 1. MySQL Memory 实现

**位置**: `rag/memory_mysql.py`

**特性**:
- ✅ 实现 LangChain `BaseChatMessageHistory` 接口
- ✅ 异步操作支持
- ✅ 自动消息排序
- ✅ 会话摘要管理

**使用示例**:
```python
from rag.memory_mysql import MySQLChatMessageHistory

# 创建历史存储
history = MySQLChatMessageHistory(
    session=db_session,
    conversation_id="conv-123"
)

# 添加消息
await history.add_message(HumanMessage(content="你好"))
await history.add_message(AIMessage(content="你好！"))

# 获取消息
messages = await history.aget_messages()
```

### 2. RAG 知识检索

**位置**: `rag/rag_conversation_service.py`

**流程**:
1. 用户发送问题
2. 向量数据库检索相关文档
3. 构建增强 Prompt（包含检索到的上下文）
4. LLM 生成回答
5. 保存到 MySQL Memory

**配置**:
```python
# .env.dev
api_key=your-dashscope-api-key

# embeddings/embedding_deal.py
CHROMA_PORT = 8399  # ChromaDB 端口
```

### 3. 流式响应 (SSE)

**位置**: `api/routes/conversation.py`

**特性**:
- ✅ Server-Sent Events 协议
- ✅ 实时流式输出
- ✅ 支持思考过程输出
- ✅ 自动错误处理

## 🏢 企业级特性

### 1. 日志系统

所有关键操作都有完整的日志记录：

```python
logger.info(f"创建会话成功: id={conversation.id}")
logger.error(f"对话失败: {e}")
logger.debug(f"检索完成: 找到 {len(docs)} 个文档")
```

### 2. 异常处理

统一的异常处理机制：

```python
try:
    # 业务逻辑
    ...
except ConversationException:
    raise
except Exception as e:
    logger.error(f"操作失败: {e}")
    raise ConversationException(f"操作失败: {str(e)}")
```

### 3. 代码注释

完整的函数文档字符串：

```python
async def create_conversation(
    self,
    user_id: Optional[str],
    title: str = "新对话",
    model: str = "qwen-turbo"
) -> ConversationDefinition:
    """
    创建新会话
    
    :param user_id: 用户ID
    :param title: 会话标题
    :param model: 使用的模型
    :return: 会话对象
    """
```

### 4. 类型安全

使用 Python 类型提示：

```python
from typing import List, Optional, Dict, Any, AsyncGenerator

async def chat_with_rag(
    self,
    conversation_id: str,
    messages: List[Dict[str, Any]],
    model: str = "qwen-turbo"
) -> AsyncGenerator[bytes, None]:
    ...
```

## 🐛 故障排查

### 问题 1: ChromaDB 连接失败

**症状**: `ConnectionError: ChromaDB 服务未启动`

**解决**:
```bash
# 检查 ChromaDB 是否运行
chroma run --path ./chroma_data --host localhost --port 8399

# 检查端口
netstat -an | findstr 8399
```

### 问题 2: MySQL 连接失败

**症状**: `ConnectionRefusedError`

**解决**:
```bash
# 检查 MySQL 服务
mysql -u root -p

# 检查数据库是否存在
SHOW DATABASES;
USE lingxi_agent;
```

### 问题 3: SSE 流不工作

**症状**: 前端收不到流式响应

**解决**:
1. 检查浏览器 Network 面板
2. 确认响应头包含 `Content-Type: text/event-stream`
3. 检查后端日志是否有错误
4. 确认 Nginx 配置（如果使用）禁用了 buffering

### 问题 4: 向量检索为空

**症状**: 检索不到相关文档

**解决**:
```python
# 检查向量数据库是否有数据
from embeddings.embedding_deal import EmbeddingHandler, DashScopeEmbedding

embeddings = DashScopeEmbedding(api_key=settings.api_key)
# ... 连接到 ChromaDB
results = vectorstore.similarity_search("测试查询", k=3)
print(f"检索到 {len(results)} 个文档")
```

## 📈 性能优化建议

### 1. 数据库索引

已自动创建关键索引：
- `idx_conv_user_id_status`: 快速查询用户会话
- `idx_msg_conv_id_seq`: 按顺序获取消息
- `idx_summary_conv_id`: 快速查找摘要

### 2. 连接池

SQLAlchemy 已配置连接池：
```python
async_engine = create_async_engine(
    database_url,
    pool_pre_ping=True,  # 连接前检测
    pool_size=10,         # 连接池大小
    max_overflow=20,      # 最大溢出连接数
)
```

### 3. 异步操作

所有数据库操作都是异步的：
```python
async with AsyncSessionLocal() as session:
    result = await session.execute(stmt)
```

## 📚 相关文档

- [Memory MySQL 使用指南](./memory_mysql_guide.md)
- [前端重构指南](./frontend_refactor_guide.md)
- [LangChain 官方文档](https://python.langchain.com/docs/)
- [FastAPI 文档](https://fastapi.tiangolo.com/)

## ✅ 开发清单

- [x] 创建数据库表模型
- [x] 实现 MySQL Memory
- [x] 创建对话服务层
- [x] 集成 RAG 检索
- [x] 实现流式响应
- [x] 创建 API 路由
- [x] 编写前端对接指南
- [x] 添加完整日志
- [x] 添加异常处理
- [x] 编写使用文档

## 🎯 下一步优化

1. **消息重试机制**: 网络错误时自动重试
2. **离线缓存**: 前端缓存消息历史
3. **对话导出**: 支持导出对话记录
4. **多模态支持**: 图片、文件上传
5. **对话分析**: 统计对话数据、用户行为
6. **权限控制**: 基于角色的访问控制

## 📞 技术支持

如有问题，请查看：
1. 后端日志（控制台输出）
2. 浏览器 Developer Tools
3. MySQL 慢查询日志
4. ChromaDB 日志

---

**最后更新**: 2024-01-01  
**版本**: v1.0.0  
**作者**: 灵犀团队
