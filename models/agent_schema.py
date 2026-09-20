"""
Agent 相关 Pydantic Schema

/ api / agent / chat 请求与任务/审批查询的响应结构
"""
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class AgentChatRequest(BaseModel):
    """智能对话请求（意图自动路由，无需 use_rag 开关）"""
    conversation_id: str = Field(..., description="会话ID")
    messages: List[dict] = Field(..., description="消息列表（末条必须为 user）")
    model: str = Field(default="qwen-turbo", description="模型名称")
    stream: bool = Field(default=True, description="是否使用流式响应")

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v: List[dict]) -> List[dict]:
        """校验消息列表非空且末条为 user 消息"""
        if not v:
            raise ValueError("messages 不能为空")
        if v[-1].get("role") != "user":
            raise ValueError("messages 最后一条必须是 user 消息")
        return v


class TaskExecutionOut(BaseModel):
    """任务执行记录查询响应"""
    id: str
    conversation_id: str
    intent: Optional[str] = None
    status: str
    tool_calls: Optional[list] = None
    rag_answer: Optional[str] = None
    task_answer: Optional[str] = None
    result: Optional[str] = None
    error: Optional[str] = None


class ApprovalOut(BaseModel):
    """审批单查询响应"""
    id: str
    task_execution_id: Optional[str] = None
    conversation_id: Optional[str] = None
    approval_type: Optional[str] = None
    target_table: Optional[str] = None
    tool_params: Optional[dict] = None
    actions: Optional[list] = None
    status: str
    approved_by: Optional[str] = None
    decision_reason: Optional[str] = None


class ClarifyAnswerIn(BaseModel):
    """追问答案提交请求（支持批量回答多问题；兼容旧单问题请求）"""
    answer: Optional[str] = Field(default=None, max_length=500, description="用户对追问的回答（单问题兼容字段）")
    answers: Optional[List[str]] = Field(default=None, description="批量回答列表（与 questions 一一对应）")

    @field_validator("answers")
    @classmethod
    def _validate_answers(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """清洗批量回答：去除空字符项，单条非空且截断到 500 字。"""
        if v is None:
            return v
        cleaned = [str(item).strip() for item in v if item is not None]
        if not cleaned or any(not item for item in cleaned):
            raise ValueError("answers 不能为空，且每个回答不能为空")
        return [item[:500] for item in cleaned]
