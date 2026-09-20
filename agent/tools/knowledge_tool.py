"""
知识库检索工具执行器（薄封装）

工具定义已收敛至 agent/tools/agent_tool.py 的 KnowledgeToolExecutor 类，
本文件仅 re-export 公共符号，保证既有 import 零改动。
"""
from agent.tools.agent_tool import (
    DEFAULT_K,
    DOC_CONTENT_LIMIT,
    MAX_K,
    TOTAL_CONTEXT_LIMIT,
    KnowledgeToolExecutor,
)

__all__ = [
    "DEFAULT_K",
    "DOC_CONTENT_LIMIT",
    "MAX_K",
    "TOTAL_CONTEXT_LIMIT",
    "KnowledgeToolExecutor",
]
