# LangChain Memory MySQL 实现文档

## 📋 概述

本项目实现了基于 MySQL 的 LangChain Memory 功能，支持：
- ✅ 持久化存储对话历史
- ✅ 支持 LangChain 的 `BaseChatMessageHistory` 接口
- ✅ 会话摘要管理（长对话压缩）
- ✅ 异步操作支持
- ✅ 完整的索引优化

## 🗄️ 数据库表结构

### 1. `conversation` - 会话定义表

存储会话的基本信息。

| 字段 | 类型 | 说明 |
|------|------|------|
| id | VARCHAR(36) | 会话ID (UUID) |
| user_id | VARCHAR(36) | 关联用户ID（可选） |
| title | VARCHAR(255) | 会话标题 |
| status | VARCHAR(32) | 状态: active, archived, deleted |
| deleted | INT | 逻辑删除: 0-正常, 1-删除 |
| created_at | DATETIME | 创建时间 |
| updated_at | DATETIME | 更新时间 |

**索引**:
- `idx_conv_user_id_status`: (user_id, status) - 查询用户的活跃会话
- `idx_conv_created_at`: (created_at) - 按时间排序

### 2. `conversation_message` - 会话消息表

存储对话历史消息（核心 Memory 表）。

| 字段 | 类型 | 说明 |
|------|------|------|
| id | VARCHAR(36) | 消息ID (UUID) |
| conversation_id | VARCHAR(36) | 关联会话ID |
| message_type | VARCHAR(32) | 类型: human, ai, system, function, tool |
| content | TEXT | 消息内容 |
| additional_kwargs | JSON | 附加参数（工具调用等） |
| sequence_number | INT | 消息序号（用于排序） |
| token_count | INT | Token数量 |
| created_at | DATETIME | 创建时间 |

**索引**:
- `idx_msg_conv_id_seq`: (conversation_id, sequence_number) - 按顺序获取消息
- `idx_msg_conv_id_type`: (conversation_id, message_type) - 按类型过滤
- `idx_msg_created_at`: (created_at) - 按时间查询

### 3. `conversation_summary` - 会话摘要表

存储对话摘要，用于长对话压缩。

| 字段 | 类型 | 说明 |
|------|------|------|
| id | VARCHAR(36) | 摘要ID (UUID) |
| conversation_id | VARCHAR(36) | 关联会话ID（唯一） |
| summary | TEXT | 摘要内容 |
| message_count | INT | 已总结的消息数量 |
| last_message_id | VARCHAR(36) | 最后一条被总结的消息ID |
| summary_token_count | INT | 摘要Token数量 |
| created_at | DATETIME | 创建时间 |
| updated_at | DATETIME | 更新时间 |

**索引**:
- `idx_summary_conv_id`: (conversation_id) - 唯一索引

## 🚀 快速开始

### 1. 执行数据库迁移

```bash
# 创建表
python scripts/migrate_conversation.py --action create

# 删除表（谨慎使用）
python scripts/migrate_conversation.py --action drop
```

### 2. 基础使用

```python
from rag.memory_mysql import MySQLChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

# 创建 MySQL 历史存储
history = MySQLChatMessageHistory(
    session=db_session,
    conversation_id="your-conversation-id"
)

# 添加消息
await history.add_message(HumanMessage(content="你好！"))
await history.add_message(AIMessage(content="你好！有什么可以帮助你的？"))

# 获取消息
messages = await history.aget_messages()
for msg in messages:
    print(f"{msg.type}: {msg.content}")

# 清空历史
await history.aclear()
```

### 3. 在 LangChain Chain 中使用

```python
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from rag.memory_mysql import MySQLChatMessageHistory

# 创建 MySQL 历史存储
mysql_history = MySQLChatMessageHistory(
    session=db_session,
    conversation_id="conv-123"
)

# 创建 LangChain Memory
memory = ConversationBufferMemory(
    chat_memory=mysql_history,
    return_messages=True,
    memory_key="history"
)

# 创建对话链
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# 执行对话（自动从 MySQL 加载历史）
response = await conversation.ainvoke("你好，我之前问了什么？")
print(response["response"])
```

### 4. 使用摘要 Memory（长对话压缩）

```python
from langchain.memory import ConversationSummaryBufferMemory
from rag.memory_mysql import MySQLChatMessageHistory, MySQLConversationSummaryMemory

# 创建历史存储
mysql_history = MySQLChatMessageHistory(
    session=db_session,
    conversation_id="conv-456"
)

# 创建摘要管理器
summary_manager = MySQLConversationSummaryMemory(db_session)

# 加载现有摘要
existing_summary = await summary_manager.get_summary("conv-456")

# 创建带摘要的 Memory
memory = ConversationSummaryBufferMemory(
    llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
    chat_memory=mysql_history,
    max_token_limit=2000,  # 超过 2000 token 时自动摘要
    memory_key="history",
    return_messages=True
)

# 注入现有摘要
if existing_summary:
    memory.buffer = existing_summary

# 创建对话链并执行
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
conversation = ConversationChain(llm=llm, memory=memory)

response = await conversation.ainvoke("继续我们的对话")

# 保存摘要到 MySQL
messages = await mysql_history.aget_messages()
await summary_manager.save_summary(
    conversation_id="conv-456",
    summary=memory.buffer,
    message_count=len(messages)
)
```

## 📖 API 参考

### MySQLChatMessageHistory

继承自 LangChain 的 `BaseChatMessageHistory`，提供异步方法：

| 方法 | 说明 |
|------|------|
| `add_message(message)` | 添加单条消息 |
| `aadd_messages(messages)` | 批量添加消息 |
| `aget_messages()` | 获取消息列表 |
| `aclear()` | 清空消息历史 |

**同步方法**（不推荐，仅用于兼容）：
- `messages` (property)
- `add_user_message(message)`
- `add_ai_message(message)`
- `clear()`

### MySQLConversationSummaryMemory

管理会话摘要：

| 方法 | 说明 |
|------|------|
| `get_summary(conversation_id)` | 获取会话摘要 |
| `save_summary(...)` | 保存或更新摘要 |
| `delete_summary(conversation_id)` | 删除摘要 |

## 🔧 高级用法

### 自定义消息类型

支持 LangChain 的所有消息类型：

```python
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    FunctionMessage,
    ToolMessage
)

# 系统消息
await history.add_message(SystemMessage(content="你是一个助手"))

# 工具调用消息
await history.add_message(AIMessage(
    content="",
    additional_kwargs={
        "tool_calls": [{
            "id": "call_123",
            "type": "function",
            "function": {"name": "search", "arguments": '{"query": "xxx"}'}
        }]
    }
))
```

### 批量导入历史对话

```python
from langchain_core.messages import HumanMessage, AIMessage

# 准备消息列表
messages = [
    HumanMessage(content="第一条消息"),
    AIMessage(content="第一条回复"),
    HumanMessage(content="第二条消息"),
    AIMessage(content="第二条回复"),
]

# 批量添加
await history.aadd_messages(messages)
```

### 查询特定类型的消息

```python
from sqlalchemy import select
from models.conversation_model import ConversationMessage

# 只获取用户消息
stmt = select(ConversationMessage).where(
    ConversationMessage.conversation_id == "conv-123",
    ConversationMessage.message_type == "human"
).order_by(ConversationMessage.sequence_number.asc())

result = await db_session.execute(stmt)
human_messages = result.scalars().all()
```

## 📊 性能优化建议

1. **定期清理旧会话**:
```python
from sqlalchemy import delete, func
from datetime import datetime, timedelta

# 删除 30 天前的会话
cutoff_date = datetime.utcnow() - timedelta(days=30)
stmt = delete(ConversationDefinition).where(
    ConversationDefinition.created_at < cutoff_date,
    ConversationDefinition.status == "archived"
)
await db_session.execute(stmt)
```

2. **限制消息数量**:
```python
# 只保留最近 100 条消息
stmt = select(ConversationMessage).where(
    ConversationMessage.conversation_id == "conv-123"
).order_by(ConversationMessage.sequence_number.desc()).limit(100)
```

3. **使用摘要压缩长对话**:
   - 当对话超过一定长度时，使用 `ConversationSummaryBufferMemory`
   - 自动将历史压缩为摘要，减少 token 消耗

## ⚠️ 注意事项

1. **异步操作**: 所有数据库操作都是异步的，确保使用 `await`
2. **事务管理**: 操作失败时会自动 rollback
3. **消息顺序**: 使用 `sequence_number` 保证消息顺序
4. **JSON 字段**: MySQL 5.7+ 支持 JSON 类型，用于存储 `additional_kwargs`

## 🐛 故障排查

### 问题 1: 消息加载为空

**原因**: 可能 `conversation_id` 不匹配

**解决**:
```python
# 检查数据库中的记录
from sqlalchemy import select
stmt = select(ConversationMessage).where(
    ConversationMessage.conversation_id == "your-id"
)
result = await db.execute(stmt)
print(result.scalars().all())
```

### 问题 2: 消息顺序错乱

**原因**: `sequence_number` 计算错误

**解决**: 确保使用 `await history._get_next_sequence_number()`

### 问题 3: JSON 字段报错

**原因**: MySQL 版本 < 5.7 不支持 JSON 类型

**解决**: 修改模型，将 JSON 改为 TEXT，手动序列化：
```python
additional_kwargs: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
```

## 📝 示例代码

完整的 API 示例请参考:
- [memory_mysql_example.py](file://d:\lingxi-agent-backend\rag\memory_mysql_example.py)

## 📚 相关资源

- [LangChain Memory 文档](https://python.langchain.com/docs/modules/memory/)
- [LangChain ChatMessageHistory](https://python.langchain.com/api_reference/core/chat_history/langchain_core.chat_history.BaseChatMessageHistory.html)

