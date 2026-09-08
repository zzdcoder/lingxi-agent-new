"""
任务执行相关 Pydantic Schema

用于任务执行记录创建与序列化
"""
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class TaskStatus(str, Enum):
    """任务执行状态"""
    RUNNING = "running"
    COMPLETED = "completed"
    REJECTED = "rejected"
    FAILED = "failed"
    CANCELED = "canceled"


class TaskExecutionCreate(BaseModel):
    """创建任务执行记录请求"""
    conversation_id: Optional[str] = Field(default=None, description="会话ID")
    user_id: Optional[str] = Field(default=None, description="发起用户ID")
    intent: Optional[str] = Field(default=None, description="路由意图")


class TaskExecutionOut(BaseModel):
    """任务执行记录响应"""
    model_config = ConfigDict(from_attributes=True)

    id: str
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None
    intent: Optional[str] = None
    status: str
    tool_calls: Optional[list] = None
    rag_answer: Optional[str] = None
    task_answer: Optional[str] = None
    result: Optional[str] = None
    error: Optional[str] = None
