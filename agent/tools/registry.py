"""
工具注册表（薄封装）

工具定义与元数据已收敛至 agent/tools/agent_tool.py（ToolSpec / REGISTRY 及
get_* 查询函数），本文件仅 re-export 公共符号，保证既有 import 零改动。
"""
from agent.tools.agent_tool import (
    REGISTRY,
    RiskLevel,
    ToolSpec,
    get_all_tools,
    get_approval_write_tools,
    get_read_tools,
    get_tool,
)

__all__ = [
    "REGISTRY",
    "RiskLevel",
    "ToolSpec",
    "get_all_tools",
    "get_approval_write_tools",
    "get_read_tools",
    "get_tool",
]
