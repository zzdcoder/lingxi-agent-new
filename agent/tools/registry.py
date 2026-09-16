"""
工具注册表

集中声明 Agent 可用工具的元数据（名称 / 描述 / 风险等级 / 审批开关 / 执行方法），
供任务节点与审批模块复用。未来新增工具只需在此声明并在 DbToolExecutor 中实现。

安全基线（设计文档 §5.5.1、§11）：
- 只读工具（list_tables / query_data）：不触发审批；
- 写工具（insert_data / update_data / delete_data）：insert 的审批开关由配置
  AGENT_INSERT_REQUIRES_APPROVAL 动态控制，update/delete 默认强制审批。

说明：ToolSpec.handler 记录 DbToolExecutor 上的方法名，任务节点在运行时
以具体 executor 实例绑定为 LangChain Tool（executor 依赖运行时 db session）。
"""
from dataclasses import dataclass
from typing import Optional

from core.config import settings


class RiskLevel:
    """工具风险等级"""
    READ = "read"        # 只读，不涉及数据变更
    WRITE = "write"      # 写操作，可能变更数据


@dataclass(frozen=True)
class ToolSpec:
    """工具规格：元数据 + 审批策略 + 对应执行方法名"""
    name: str                    # 工具名（Agent 调用名）
    description: str             # 能力描述（供 LLM 选择工具）
    risk_level: str              # RiskLevel.READ / WRITE
    requires_approval: bool      # 是否需要人工审批
    handler: str                 # DbToolExecutor 上的执行方法名


# ---------------------------------------------------------------------------
# 工具清单（集中声明）
# ---------------------------------------------------------------------------

def _build_tools() -> dict[str, ToolSpec]:
    """构建工具注册表（审批开关由配置动态决定）。"""
    tools: dict[str, ToolSpec] = {}

    def _register(spec: ToolSpec) -> None:
        tools[spec.name] = spec

    # 只读工具
    _register(ToolSpec(
        name="list_tables",
        description="列出当前授权范围内可访问的数据库表清单",
        risk_level=RiskLevel.READ,
        requires_approval=False,
        handler="list_tables",
    ))
    _register(ToolSpec(
        name="query_data",
        description="按条件查询指定表的数据，仅限授权表白名单，单次最多返回 50 行",
        risk_level=RiskLevel.READ,
        requires_approval=False,
        handler="query_data",
    ))
    _register(ToolSpec(
        name="search_knowledge",
        description=(
            "检索知识库文档，返回相关文档片段（执行任务需要政策/规则/文档依据时调用）。"
            "参数 query 为检索问题，k 为返回文档数（1~10，默认 4）"
        ),
        risk_level=RiskLevel.READ,
        requires_approval=False,
        handler="search_knowledge",
    ))

    # 写工具：insert 审批开关由配置控制，update/delete 强制审批
    # 二次确认（§5.5.2）：全部写工具必填 user_intent_quote（用户原话片段）
    _register(ToolSpec(
        name="insert_data",
        description=(
            "向指定表插入一条数据，仅限授权表白名单。"
            "必须提供 user_intent_quote：用户原话中表达插入/新增意图的原文片段"
        ),
        risk_level=RiskLevel.WRITE,
        requires_approval=settings.agent_insert_requires_approval,
        handler="insert_data",
    ))
    _register(ToolSpec(
        name="update_data",
        description=(
            "按条件更新指定表的数据，仅限授权表白名单，必须携带过滤条件。"
            "必须提供 user_intent_quote：用户原话中表达更新/修改意图的原文片段"
        ),
        risk_level=RiskLevel.WRITE,
        requires_approval=True,
        handler="update_data",
    ))
    _register(ToolSpec(
        name="delete_data",
        description=(
            "按条件删除指定表的数据，仅限授权表白名单，必须携带过滤条件。"
            "必须提供 user_intent_quote：用户原话中表达删除意图的原文片段"
        ),
        risk_level=RiskLevel.WRITE,
        requires_approval=True,
        handler="delete_data",
    ))
    return tools


# 注册表实例（模块加载时构建）
REGISTRY: dict[str, ToolSpec] = _build_tools()


def get_tool(name: str) -> Optional[ToolSpec]:
    """按名称获取工具规格，未注册返回 None。"""
    return REGISTRY.get(name)


def get_all_tools() -> list[ToolSpec]:
    """获取全部工具规格（按注册顺序）。"""
    return list(REGISTRY.values())


def get_read_tools() -> list[ToolSpec]:
    """获取只读工具（任务 Agent 默认绑定）。"""
    return [t for t in REGISTRY.values() if t.risk_level == RiskLevel.READ]


def get_approval_write_tools() -> list[ToolSpec]:
    """获取需要审批的写工具（update/delete 及配置开启审批的 insert）。"""
    return [
        t for t in REGISTRY.values()
        if t.risk_level == RiskLevel.WRITE and t.requires_approval
    ]
