"""
用户相关 Pydantic Schema

用于登录注册请求校验与响应序列化
"""

from pydantic import BaseModel, Field, field_serializer
from datetime import datetime


class UserRegister(BaseModel):
    """用户注册请求"""
    username: str = Field(..., min_length=3, max_length=32, pattern=r"^[a-zA-Z0-9_]+$")
    password: str = Field(..., min_length=6, max_length=64)
    user_type: str = Field(..., pattern=r"^(manufacturer|mobile)$")
    captcha_id: str = Field(..., min_length=1, max_length=64)
    captcha_text: str = Field(..., min_length=1, max_length=16)


class UserLogin(BaseModel):
    """用户登录请求"""
    username: str = Field(..., min_length=1, max_length=32)
    password: str = Field(..., min_length=1, max_length=64)
    captcha_id: str = Field(..., min_length=1, max_length=64)
    captcha_text: str = Field(..., min_length=1, max_length=16)


class UserOut(BaseModel):
    """用户信息响应"""
    id: int
    username: str
    user_type: str
    created_at: datetime

    @field_serializer("id")
    def _serialize_id(self, value: int) -> str:
        return str(value)

    class Config:
        from_attributes = True


class Token(BaseModel):
    """登录成功 Token 响应"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserOut


class CaptchaResponse(BaseModel):
    """验证码响应"""
    captcha_id: str
    image_url: str
