"""
任务追问工具（薄封装）

工具定义已收敛至 agent/tools/agent_tool.py 的 ClarifyTool 类别下的
ask_user（@tool），本文件仅 re-export 公共符号：
- 保证既有 import 零改动（agent/nodes/task_node.py 引用 ASK_USER_TOOL_NAME）；
- ask_user 作为模块级 BaseTool 属性被 re-export，仍会被启动时
  agent/tools/registrar.py 对 tools 包的扫描发现并注册。
"""
from agent.tools.agent_tool import (
    ASK_USER_TOOL_NAME,
    CLARIFY_TYPE,
    ask_user,
)

__all__ = [
    "ASK_USER_TOOL_NAME",
    "CLARIFY_TYPE",
    "ask_user",
]
