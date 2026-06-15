# 前端重构指南

## 📋 概述

由于前端项目 (`lingxi-agent-portal`) 不在当前工作区，以下是需要手动应用的代码修改。

## 🔧 需要修改的文件

### 1. `src/services/llm.ts` - 重构服务层

**文件路径**: `d:\lingxi-agent-portal\src\services\llm.ts`

**修改内容**: 将文件内容替换为以下内容

```typescript
/**
 * LLM 对话 API 服务层
 *
 * 提供统一的对话管理接口，支持：
 * - 会话 CRUD 操作
 * - 消息历史查询
 * - SSE 流式响应
 * - 自动 JWT 认证
 *
 * @module services/llm
 */

import type { Attachment } from '../types';

// API 基础路径（从环境变量读取，默认 /api）
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || '/api').replace(/\/$/, '');

// ==========================================
// 类型定义
// ==========================================

/** 后端会话模型 */
export interface BackendConversation {
  id: string;
  user_id?: string;
  title: string;
  status: string;
  created_at: string;
  updated_at: string;
}

/** 后端消息模型 */
export interface BackendMessage {
  id: string;
  conversation_id: string;
  message_type: 'human' | 'ai' | 'system';
  content: string;
  additional_kwargs?: Record<string, any>;
  sequence_number: number;
  token_count?: number;
  created_at: string;
}

/** 聊天消息（发送给后端） */
export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  attachment_ids?: string[];
}

/** 流式回调接口 */
export interface StreamCallbacks {
  onThinking?: (text: string) => void;
  onContent?: (text: string) => void;
  onDone?: () => void;
  onError?: (error: Error) => void;
}

// ==========================================
// 会话管理 API
// ==========================================

/**
 * 获取会话列表
 * @returns 会话列表（按更新时间倒序）
 */
export async function listConversations(): Promise<BackendConversation[]> {
  return requestJson<BackendConversation[]>('/conversations');
}

/**
 * 创建新会话
 * @param title 会话标题
 * @param model 使用的模型
 * @returns 创建的会话
 */
export async function createConversation(title: string, model: string): Promise<BackendConversation> {
  const params = new URLSearchParams({ title, model });
  return requestJson<BackendConversation>(`/conversations?${params.toString()}`, {
    method: 'POST',
  });
}

/**
 * 删除会话
 * @param id 会话ID
 */
export async function deleteConversation(id: string): Promise<void> {
  await requestJson<void>(`/conversations/${encodeURIComponent(id)}`, {
    method: 'DELETE',
  });
}

// ==========================================
// 消息管理 API
// ==========================================

/**
 * 获取会话的消息历史
 * @param conversationId 会话ID
 * @returns 消息列表（按序号排序）
 */
export async function listMessages(conversationId: string): Promise<BackendMessage[]> {
  return requestJson<BackendMessage[]>(`/conversations/${encodeURIComponent(conversationId)}/messages`);
}

// ==========================================
// 对话 API（流式）
// ==========================================

/**
 * 发起流式对话
 * 
 * @param messages 消息列表（包含历史消息和当前消息）
 * @param model 使用的模型
 * @param conversationId 会话ID
 * @param callbacks 流式回调
 * @returns 包含 abort 方法的对象，用于中断请求
 */
export function streamChat(
  messages: ChatMessage[],
  model: string,
  conversationId: string,
  callbacks: StreamCallbacks
): { abort: () => void } {
  const controller = new AbortController();

  // 发起 POST 请求
  fetch(`${API_BASE_URL}/conversations/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...getAuthHeaders(),
    },
    body: JSON.stringify({
      messages,
      model,
      conversation_id: conversationId,
      stream: true,
    }),
    signal: controller.signal,
  })
    .then(async (res) => {
      // 检查响应状态
      if (!res.ok || !res.body) {
        throw new Error(`请求失败，请稍后重试（${res.status}）`);
      }

      // 读取 SSE 流
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        // 解码数据块
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        // 处理每一行
        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || !trimmed.startsWith('data:')) continue;

          const data = trimmed.slice(5).trim();
          
          // 检查是否完成
          if (data === '[DONE]') {
            callbacks.onDone?.();
            return;
          }

          try {
            // 解析 JSON 数据
            const parsed = JSON.parse(data);
            
            // 处理思考过程
            if (parsed.thinking) {
              callbacks.onThinking?.(parsed.thinking);
            }
            
            // 处理回答内容
            if (parsed.content) {
              callbacks.onContent?.(parsed.content);
            }
            
            // 检查完成标记
            if (parsed.done) {
              callbacks.onDone?.();
              return;
            }
          } catch (error) {
            // 解析失败，直接作为内容输出
            console.warn('SSE 数据解析失败:', data, error);
            callbacks.onContent?.(data);
          }
        }
      }

      // 流结束
      callbacks.onDone?.();
    })
    .catch((err) => {
      // 忽略中止错误
      if (err.name !== 'AbortError') {
        callbacks.onError?.(err);
      }
    });

  return { abort: () => controller.abort() };
}

// ==========================================
// 附件 API
// ==========================================

/**
 * 上传附件
 * @param conversationId 会话ID（可选）
 * @param file 文件对象
 * @returns 附件信息
 */
export async function uploadAttachment(conversationId: string | undefined, file: File): Promise<Attachment> {
  const formData = new FormData();
  formData.append('file', file);
  if (conversationId) {
    formData.append('conversation_id', conversationId);
  }

  const attachment = await requestJson<BackendAttachment>('/attachments', {
    method: 'POST',
    body: formData,
  });

  return mapBackendAttachment(attachment);
}

// ==========================================
// 辅助函数
// ==========================================

/** 后端附件模型 */
interface BackendAttachment {
  id: string | number;
  name?: string;
  original_name?: string;
  mimeType?: string;
  content_type?: string;
  size: number;
  url: string;
}

/**
 * 映射后端附件模型到前端模型
 */
export function mapBackendAttachment(attachment: BackendAttachment): Attachment {
  return {
    id: String(attachment.id),
    name: attachment.name || attachment.original_name || '未命名附件',
    mimeType: attachment.mimeType || attachment.content_type || 'application/octet-stream',
    size: attachment.size,
    url: attachment.url,
  };
}

/**
 * 从 localStorage 读取 JWT Token 并构建 Authorization 请求头
 * 所有受保护的 API 请求都会自动携带此头部
 */
function getAuthHeaders(): Record<string, string> {
  const token = localStorage.getItem('access_token');
  return token ? { Authorization: `Bearer ${token}` } : {};
}

/**
 * 通用 JSON 请求函数
 * 
 * @param path API 路径
 * @param options Fetch 选项
 * @returns 解析后的 JSON 数据
 */
async function requestJson<T>(path: string, options: RequestInit = {}): Promise<T> {
  // 自动合并认证请求头
  const headers: Record<string, string> = {
    ...(options.headers as Record<string, string> || {}),
    ...getAuthHeaders(),
  };

  const res = await fetch(`${API_BASE_URL}${path}`, { ...options, headers });

  // 检查响应状态
  if (!res.ok) {
    throw new Error(`请求失败，请稍后重试（${res.status}）`);
  }

  // 解析 JSON
  const text = await res.text();
  return text ? (JSON.parse(text) as T) : (undefined as T);
}
```

### 2. `src/hooks/useChat.ts` - 更新消息映射

**文件路径**: `d:\lingxi-agent-portal\src\hooks\useChat.ts`

**修改内容**: 找到 `mapBackendMessage` 函数（约第 412 行），替换为：

```typescript
function mapBackendMessage(message: BackendMessage): Message {
  return {
    id: String(message.id),
    role: message.message_type === 'human' ? 'user' : 
          message.message_type === 'ai' ? 'assistant' : 'system',
    content: message.content,
    attachments: message.additional_kwargs?.attachments?.map(mapBackendAttachment),
    thinkingContent: message.additional_kwargs?.thinking_content,
    timestamp: parseBackendDate(message.created_at),
  };
}
```

## 📝 主要改动说明

### 1. API 路径变更
- **旧**: `/api/chat`
- **新**: `/api/conversations/chat`

### 2. 请求参数变更
- **旧**: `{ messages, model, conversation_id }`
- **新**: `{ messages, model, conversation_id, stream: true }`

### 3. 响应格式变更
- **旧**: `role: 'user' | 'assistant' | 'system'`
- **新**: `message_type: 'human' | 'ai' | 'system'`

### 4. 移除 Mock 实现
- 删除了 `USE_MOCK_CHAT` 相关代码
- 所有请求直接访问真实后端

## ✅ 测试步骤

1. **启动后端服务**:
```bash
cd d:\lingxi-agent-backend
python -m uvicorn app.main:app --reload --port 8000
```

2. **启动前端服务**:
```bash
cd d:\lingxi-agent-portal
npm run dev
```

3. **测试对话功能**:
   - 打开浏览器访问 `http://localhost:5173`
   - 登录系统
   - 创建新对话
   - 发送消息，验证流式响应

## 🐛 常见问题

### Q1: CORS 错误
**解决**: 确保后端已配置 CORS 中间件（已在 `main.py` 中配置）

### Q2: 401 Unauthorized
**解决**: 检查 `localStorage` 中是否有有效的 `access_token`

### Q3: SSE 流不工作
**解决**: 
- 检查浏览器 Network 面板，确认响应头包含 `Content-Type: text/event-stream`
- 检查后端日志，确认没有报错

## 📚 下一步

完成前端修改后，你可以：
1. 添加错误边界处理
2. 实现消息重试机制
3. 添加加载状态优化
4. 实现离线缓存
