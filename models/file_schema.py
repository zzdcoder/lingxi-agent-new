"""
文件相关 Pydantic Schema

用于 API 请求校验与响应序列化
"""
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime


class AttachmentCreate(BaseModel):
    """创建附件请求参数"""
    conversation_id: str | None = Field(default=None, description="关联对话 ID")


class AttachmentOut(BaseModel):
    """附件响应模型（与前端 BackendAttachment 对齐）

    序列化时输出 name / mimeType / url，
    从 ORM 验证时映射 original_name / content_type / cos_url
    """
    model_config = ConfigDict(
        populate_by_name=True,
        from_attributes=True,
    )

    id: str
    name: str = Field(..., validation_alias="original_name")
    mimeType: str = Field(..., validation_alias="content_type")
    size: int
    url: str = Field(..., validation_alias="cos_url")
    created_at: datetime


class FileDDLAndSplitInput(BaseModel):
    """
    文件处理清洗请求参数
    """
    save_embedding: bool = Field(default=False,description="是否存入向量库")
    file_id: str = Field(description="文件的COS ID")
    file_type: str = Field(description="文件类型")
    separators: str = Field(description="分段标识符")
    chunk_size: int = Field(description="分块大小")
    chunk_overlap: int = Field(description="分块重叠大小")
    ddl_option1: bool = Field(default=False, description="替换掉连续的空格、换行符和制表符")
    ddl_option2: bool = Field(default=False, description="删除所有 URL 和电子邮件地址")
    metadata_list: list[dict] = Field(default=[], description="元数据")
    auth_option: str = Field(description="权限设置")
    belong_demain: list[str] = Field(default=[], description="所属领域")


class ChunkResult(BaseModel):
    """单个分块结果"""
    page_content: str = Field(description="分块文本内容")
    metadata: dict = Field(default={}, description="分块元数据")


class FileDDLAndSplitOutput(BaseModel):
    """
    文件清洗切片响应参数
    """
    deal_result: list[ChunkResult] = Field(default=[], description="切片结果")