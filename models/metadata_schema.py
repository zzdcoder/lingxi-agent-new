"""
元数据相关 Pydantic Schema

用于 API 请求校验与响应序列化
"""

from pydantic import BaseModel, Field
from datetime import datetime


class MetadataFieldType:
    STRING = "string"
    NUMBER = "number"
    TIME = "time"


class MetadataCreate(BaseModel):
    """创建元数据定义请求参数"""
    name: str = Field(..., min_length=1, max_length=64, description="字段名称（英文标识）")
    display_name: str = Field(..., min_length=1, max_length=128, description="显示名称")
    field_type: str = Field(..., description="字段类型：string / number / time")


class MetadataUpdate(BaseModel):
    """更新元数据定义请求参数"""
    display_name: str | None = Field(default=None, min_length=1, max_length=128, description="显示名称")
    is_enabled: bool | None = Field(default=None, description="是否启用")


class MetadataOut(BaseModel):
    """元数据定义响应模型"""
    id: str
    name: str
    display_name: str
    field_type: str
    is_builtin: bool
    is_enabled: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        populate_by_name = True
        from_attributes = True
