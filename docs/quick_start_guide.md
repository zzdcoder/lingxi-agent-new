# 对话功能快速启动指南

## 📋 前置检查清单

在启动服务之前，请确保：

- [ ] MySQL 服务已启动并运行
- [ ] 数据库 `lingxi_agent` 已创建
- [ ] Python 虚拟环境已激活
- [ ] 所有依赖已安装

## 🚀 启动步骤

### 步骤 1: 创建数据库表

```bash
cd d:\lingxi-agent-backend

# 执行数据库迁移
python scripts/migrate_conversation.py --action create
```

**预期输出**:
```
INFO - 开始创建数据库表...
INFO - ✓ conversation 表创建成功
INFO - ✓ conversation_message 表创建成功
INFO - ✓ conversation_summary 表创建成功
INFO - 所有表创建完成！
```

**验证**:
```bash
mysql -u root -p lingxi_agent -e "SHOW TABLES;"
```

应该看到：
```
+------------------------+
| Tables_in_lingxi_agent |
+------------------------+
| conversation           |
| conversation_message   |
| conversation_summary   |
+------------------------+
```

### 步骤 2: 启动后端服务

```bash
# 确保在正确的目录
cd d:\lingxi-agent-backend

# 启动 FastAPI 服务
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**预期输出**:
```
INFO:     Will watch for changes in these directories: ['d:\\lingxi-agent-backend']
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [12345]
INFO:     Started server process [12346]
INFO:     Waiting for application startup.
INFO - Chroma 服务已启动，PID=12347, 端口=8399
INFO:     Application startup complete.
```

### 步骤 3: 验证后端服务

#### 3.1 访问 API 文档

打开浏览器访问: http://localhost:8000/docs

你应该能看到完整的 Swagger UI 界面，包括：
- 用户认证
- 对话管理
- 文件处理
- 健康检查

#### 3.2 测试健康检查

```bash
curl http://localhost:8000/health
```

**预期响应**:
```json
{
  "status": "ok"
}
```

#### 3.3 运行自动化测试

```bash
# 安装测试依赖（如果还没有）
pip install httpx

# 运行测试脚本
python scripts/test_conversation.py
```

**预期输出**:
```
🚀 开始对话功能测试

=== 测试 1: 健康检查 ===
INFO - 状态码: 200
INFO - 响应: {'status': 'ok'}
INFO - ✓ 健康检查通过

=== 测试 2: 用户登录 ===
INFO - ✓ 登录成功，获取 Token: eyJhbGciOiJIUzI1NiIs...

=== 测试 3: 创建会话 ===
INFO - 状态码: 200
INFO - ✓ 创建会话成功: ID=uuid-xxxx
INFO -   标题: 测试对话
INFO -   状态: active

=== 测试 4: 获取会话列表 ===
INFO - 状态码: 200
INFO - ✓ 获取会话列表成功: 共 1 个会话

=== 测试 5: 流式对话 ===
INFO - 状态码: 200
INFO - ✓ 流式对话请求成功
响应内容:
你好！我是灵犀智能助手...
INFO - ✓ 对话完成，共 150 个字符

=== 测试 6: 获取消息历史 ===
INFO - 状态码: 200
INFO - ✓ 获取消息历史成功: 共 2 条消息

✅ 所有测试完成！
```

## 🔧 常见问题排查

### 问题 1: 数据库连接失败

**错误信息**:
```
ConnectionRefusedError: [WinError 10061] 无法连接
```

**解决方案**:
```bash
# 1. 检查 MySQL 服务是否运行
# Windows: 打开服务管理器，找到 MySQL 服务

# 2. 检查数据库是否存在
mysql -u root -p
SHOW DATABASES;

# 3. 如果数据库不存在，创建它
CREATE DATABASE lingxi_agent CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

### 问题 2: ChromaDB 端口被占用

**错误信息**:
```
RuntimeError: ChromaDB 端口 8399 被占用
```

**解决方案**:
```bash
# 1. 查找占用端口的进程
netstat -ano | findstr 8399

# 2. 结束进程（替换 PID）
taskkill /F /PID <PID>

# 3. 或者修改 .env.dev 中的端口
CHROMA_PORT=8400
```

### 问题 3: get_current_user_id 导入错误

**错误信息**:
```
ImportError: cannot import name 'get_current_user_id' from 'utils.auth'
```

**解决方案**:

这个函数已经在 `utils/auth.py` 中添加。如果仍然报错，请检查：

```bash
# 1. 确认文件内容
cat utils/auth.py | grep -A 20 "def get_current_user_id"

# 2. 重启服务（确保重新加载）
# 停止当前服务（Ctrl+C）
python -m uvicorn app.main:app --reload --port 8000
```

### 问题 4: ConversationException 未定义

**错误信息**:
```
ImportError: cannot import name 'ConversationException' from 'core.exceptions'
```

**解决方案**:

这个异常已经在 `core/exceptions.py` 中定义。如果仍然报错：

```bash
# 1. 确认文件内容
cat core/exceptions.py | grep -A 3 "class ConversationException"

# 2. 检查是否有语法错误
python -m py_compile core/exceptions.py
```

### 问题 5: SSE 流不工作

**症状**: 前端收不到流式响应

**排查步骤**:

1. **检查后端日志**:
```bash
# 查看是否有错误日志
# 应该看到类似：
INFO - 开始流式对话: conversation_id=xxx
INFO - 对话完成: 耗时=2.34s
```

2. **检查浏览器 Network 面板**:
- 打开 Developer Tools (F12)
- 切换到 Network 标签
- 发起对话请求
- 检查响应头是否包含: `Content-Type: text/event-stream`

3. **测试 SSE 端点**:
```bash
curl -N -X POST http://localhost:8000/api/conversations/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "messages": [{"role": "user", "content": "你好"}],
    "model": "qwen-turbo",
    "conversation_id": "test-id",
    "use_rag": false
  }'
```

## 📊 验证数据库数据

### 查看会话记录

```sql
USE lingxi_agent;

-- 查看所有会话
SELECT id, title, status, created_at 
FROM conversation 
ORDER BY created_at DESC 
LIMIT 10;

-- 查看某个会话的消息
SELECT message_type, content, sequence_number, created_at
FROM conversation_message
WHERE conversation_id = 'your-conversation-id'
ORDER BY sequence_number ASC;
```

### 查看向量数据

```python
# Python 脚本查看 ChromaDB
import chromadb

client = chromadb.HttpClient(host="localhost", port=8399)
collections = client.list_collections()
print(f"集合数量: {len(collections)}")

for collection in collections:
    print(f"集合名称: {collection.name}")
    print(f"  文档数量: {collection.count()}")
```

## 🎯 前端对接

### 步骤 1: 修改前端代码

参考 [前端重构指南](./frontend_refactor_guide.md)，主要修改：

1. **更新 `src/services/llm.ts`**
   - 移除 Mock 实现
   - 更新 API 路径为 `/api/conversations/chat`

2. **更新 `src/hooks/useChat.ts`**
   - 修改消息映射逻辑

### 步骤 2: 启动前端

```bash
cd d:\lingxi-agent-portal

# 安装依赖（首次）
npm install

# 启动开发服务器
npm run dev
```

### 步骤 3: 测试完整流程

1. 打开浏览器: http://localhost:5173
2. 登录系统
3. 创建新对话
4. 发送消息，观察流式响应
5. 检查消息历史是否正确保存

## 🔍 调试技巧

### 1. 查看实时日志

```bash
# 后端日志（控制台输出）
# 应该看到类似：
2024-01-01 12:00:00 - rag.conversation_service - INFO - 开始对话: conversation_id=xxx
2024-01-01 12:00:02 - rag.conversation_service - INFO - 对话完成: 耗时=2.34s
```

### 2. 使用 Postman 测试 API

导入以下集合：

```json
{
  "info": {
    "name": "灵犀对话 API",
    "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
  },
  "item": [
    {
      "name": "健康检查",
      "request": {
        "method": "GET",
        "url": "http://localhost:8000/health"
      }
    },
    {
      "name": "创建会话",
      "request": {
        "method": "POST",
        "url": "http://localhost:8000/api/conversations?title=测试&model=qwen-turbo",
        "header": [
          {
            "key": "Authorization",
            "value": "Bearer {{token}}"
          }
        ]
      }
    },
    {
      "name": "流式对话",
      "request": {
        "method": "POST",
        "url": "http://localhost:8000/api/conversations/chat",
        "header": [
          {
            "key": "Content-Type",
            "value": "application/json"
          },
          {
            "key": "Authorization",
            "value": "Bearer {{token}}"
          }
        ],
        "body": {
          "mode": "raw",
          "raw": "{\n  \"messages\": [{\"role\": \"user\", \"content\": \"你好\"}],\n  \"model\": \"qwen-turbo\",\n  \"conversation_id\": \"{{conversation_id}}\",\n  \"use_rag\": false\n}"
        }
      }
    }
  ]
}
```

### 3. 数据库查询调试

```sql
-- 查看最新的对话
SELECT 
    c.id,
    c.title,
    c.status,
    COUNT(m.id) as message_count,
    c.created_at
FROM conversation c
LEFT JOIN conversation_message m ON c.id = m.conversation_id
GROUP BY c.id
ORDER BY c.created_at DESC
LIMIT 10;
```

## ✅ 验收标准

完成以下检查，确认功能正常：

- [ ] 数据库表创建成功（3 张表）
- [ ] 后端服务启动成功，无报错
- [ ] API 文档可访问（http://localhost:8000/docs）
- [ ] 健康检查返回 `{"status": "ok"}`
- [ ] 用户登录成功，获取 Token
- [ ] 创建会话成功，返回会话 ID
- [ ] 获取会话列表成功
- [ ] 流式对话正常，响应实时输出
- [ ] 消息历史正确保存和查询
- [ ] 前端可以正常对话（完成前端修改后）

## 📚 相关文档

- [完整功能文档](./conversation_feature_guide.md)
- [MySQL Memory 使用指南](./memory_mysql_guide.md)
- [前端重构指南](./frontend_refactor_guide.md)
- [项目重构建议](./refactor_suggestions.md)

## 🆘 获取帮助

如果遇到问题：

1. **查看日志**: 后端控制台输出
2. **检查文档**: 上面的故障排查章节
3. **测试 API**: 使用 `scripts/test_conversation.py`
4. **数据库检查**: 使用 SQL 查询验证数据

---

**最后更新**: 2024-01-01  
**版本**: v1.0.0
