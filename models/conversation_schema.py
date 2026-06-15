"""
对话相关 Pydantic Schema

用于 API 请求校验与响应序列化
"""
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


class ConversationCreate(BaseModel):
    """创建会话请求参数"""
    title: str = Field(default="新对话", description="会话标题")
    model: str = Field(default="qwen-turbo", description="模型名称")


class ConversationOut(BaseModel):
    """会话响应模型"""
    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: Optional[str] = None
    title: str
    status: str
    model: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class MessageCreate(BaseModel):
    """发送消息请求参数"""
    conversation_id: str = Field(..., description="会话ID")
    messages: List[dict] = Field(..., description="消息历史（包含当前消息）")
    model: str = Field(default="qwen-turbo", description="模型名称")


class MessageOut(BaseModel):
    """消息响应模型"""
    model_config = ConfigDict(from_attributes=True)

    id: str
    conversation_id: str
    message_type: str
    content: str
    additional_kwargs: Optional[dict] = None
    sequence_number: int
    token_count: Optional[int] = None
    created_at: datetime


class ChatRequest(BaseModel):
    """流式对话请求参数"""
    messages: List[dict] = Field(..., description="消息列表")
    model: str = Field(default="qwen-turbo", description="模型名称")
    conversation_id: str = Field(..., description="会话ID")
    stream: bool = Field(default=True, description="是否使用流式响应")
    use_rag: bool = Field(default=True, description="是否使用RAG知识检索")


class ChatResponse(BaseModel):
    """流式对话响应（SSE 格式）"""
    thinking: Optional[str] = Field(default=None, description="思考过程")
    content: Optional[str] = Field(default=None, description="回答内容")
    done: bool = Field(default=False, description="是否完成")
