"""
API 路由定义包

按资源域组织端点处理器（对话、文档、健康检查）
每个模块定义 FastAPI 路由，由主应用挂载
"""

from api.routes import attachments
from api.routes import auth
from api.routes import metadata
from api.routes import file_process

__all__ = ["attachments", "auth", "metadata", "file_process"]
