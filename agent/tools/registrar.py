"""
工具注册器（启动时同步 tool_registry 表，设计文档 §7.8 / §8.3）

职责：
1. **扫描 @tool 装饰器工具**：遍历 `agent_tool_scan_packages` 配置的包（默认 `tools`），
   识别其中使用 `@tool` 装饰器定义的工具（langchain BaseTool 实例），
   提取名称 / 描述 / 参数 JSON Schema；
2. **同步 ToolSpec 注册表**：将 `agent/tools/registry.py` 中声明的内置工具
   （db 工具 + search_knowledge）转为记录；
3. **幂等 upsert**：按工具名（唯一键）入库——新增工具 status=生效；
   已存在工具仅刷新元数据，**保留原状态**（人工下线的 0、熔断中的 2 不被覆盖），
   保证"代码热升级不会重置运维决策"。

调用时机：app/main.py lifespan 中 `Base.metadata.create_all` 之后（失败降级不阻塞启动）。
"""
import importlib
import inspect
import logging
import pkgutil
from typing import Any, Dict, List, Optional

from langchain_core.tools import BaseTool

from core.config import settings
from models.tool_model import ToolCircuitState, ToolRegistry, ToolStatus
from models.tool_schema import ToolSyncResult
from agent.tools.registry import REGISTRY, ToolSpec

logger = logging.getLogger(__name__)

# 内置工具分类（按 handler 归属判断）
_CATEGORY_BY_HANDLER = {
    "search_knowledge": "knowledge",
}


# =============================================================================
# 1. 扫描 @tool 装饰器工具
# =============================================================================

def discover_decorated_tools() -> Dict[str, BaseTool]:
    """
    扫描配置的包，发现使用 @tool 装饰器定义的工具。

    判定方式：`@tool` 装饰器返回 langchain `BaseTool`（StructuredTool）实例，
    因此对模块成员做 `isinstance(obj, BaseTool)` 判定即可。

    :return: {工具名: BaseTool}（同名冲突时以先扫描到的为准并告警）
    """
    tools: Dict[str, BaseTool] = {}
    packages = [p for p in settings.agent_tool_scan_packages.split(",") if p.strip()]

    for pkg_name in packages:
        try:
            pkg = importlib.import_module(pkg_name)
        except Exception as e:
            logger.warning(f"[工具注册] 导入工具包 {pkg_name} 失败（跳过扫描）: {e}")
            continue

        # 包自身
        _scan_module_tools(pkg, tools)
        # 子模块（递归）
        prefix = pkg.__name__ + "."
        for mod_info in pkgutil.walk_packages(pkg.__path__, prefix=prefix):
            try:
                mod = importlib.import_module(mod_info.name)
            except Exception as e:
                logger.warning(f"[工具注册] 导入模块 {mod_info.name} 失败（跳过）: {e}")
                continue
            _scan_module_tools(mod, tools)
    return tools


# 进程级扫描结果缓存（运行时绑定复用，避免重复导入工具包）
_decorated_tools_cache: Optional[Dict[str, BaseTool]] = None


def get_decorated_tools() -> Dict[str, BaseTool]:
    """
    获取扫描到的 @tool 装饰器工具（进程级缓存，惰性扫描一次）。

    供注册器与运行时绑定共用：绑定工具时按工具名解析 @tool 对象。
    """
    global _decorated_tools_cache
    if _decorated_tools_cache is None:
        _decorated_tools_cache = discover_decorated_tools()
    return _decorated_tools_cache


def _scan_module_tools(module, tools: Dict[str, BaseTool]) -> None:
    """扫描单个模块内 @tool 装饰器定义的工具。"""
    for name, obj in inspect.getmembers(module, lambda o: isinstance(o, BaseTool)):
        if name in tools:
            logger.warning(
                f"[工具注册] 工具名 {name} 重复定义（{tools[name].__class__.__module__} 与 "
                f"{module.__name__}），保留先注册者"
            )
            continue
        tools[name] = obj
        logger.info(f"[工具注册] 发现 @tool 工具: {name}（来源 {module.__name__}.{name}）")


def _decorated_tool_source(tool_name: str, tools: Dict[str, BaseTool]) -> str:
    """构造 @tool 工具的来源描述 module:attr。"""
    obj = tools[tool_name]
    return f"module:{obj.__class__.__module__}:{tool_name}"


def _decorated_tool_schema(tool: BaseTool) -> Optional[Dict[str, Any]]:
    """提取 @tool 工具的参数 JSON Schema。"""
    try:
        if tool.args_schema is not None:
            return tool.args_schema.model_json_schema()
    except Exception as e:
        logger.debug(f"[工具注册] 提取工具 {tool.name} 参数 Schema 失败: {e}")
    return None


# =============================================================================
# 2. ToolSpec 注册表 → 记录
# =============================================================================

def _signature_to_json_schema(func) -> Optional[Dict[str, Any]]:
    """
    将执行器方法签名转换为 JSON Schema（管理台展示参数结构用）。

    仅做尽力而为的类型映射（支持 Optional / list / dict 与默认值）；
    无法解析时返回 None（不影响注册）。
    """
    from typing import Any as _Any, get_args, get_origin

    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return None

    properties: Dict[str, Any] = {}
    required: List[str] = []
    type_map = {str: "string", int: "integer", float: "number", bool: "boolean"}

    for name, param in sig.parameters.items():
        if name in ("self", "kwargs", "args"):
            continue
        annotation = param.annotation if param.annotation is not inspect.Parameter.empty else _Any
        schema: Dict[str, Any] = {}
        optional = False

        # 处理 Optional[...]（Union[T, None]）
        origin = get_origin(annotation)
        if origin is not None and hasattr(annotation, "__args__"):
            args = get_args(annotation)
            if type(None) in args:
                optional = True
                annotation = next((a for a in args if a is not type(None)), _Any)
        if param.default is not inspect.Parameter.empty:
            optional = True
            schema["default"] = param.default

        ann_origin = get_origin(annotation)
        if annotation in type_map:
            schema["type"] = type_map[annotation]
        elif ann_origin is list:
            schema["type"] = "array"
            inner = get_args(annotation)
            if inner and inner[0] in type_map:
                schema["items"] = {"type": type_map[inner[0]]}
        elif ann_origin is dict or annotation is dict:
            schema["type"] = "object"
        else:
            schema["type"] = "string"  # 未知类型降级为字符串

        properties[name] = schema
        if not optional:
            required.append(name)

    schema: Dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _spec_to_row(spec: ToolSpec) -> ToolRegistry:
    """ToolSpec → ToolRegistry 记录（parameters 由执行方法签名推导）。"""
    handler_func = _resolve_handler_func(spec.handler)
    return ToolRegistry(
        name=spec.name,
        description=spec.description,
        parameters=_signature_to_json_schema(handler_func) if handler_func else None,
        category=_CATEGORY_BY_HANDLER.get(spec.handler, "db"),
        risk_level=spec.risk_level,
        requires_approval=int(spec.requires_approval),
        status=ToolStatus.ACTIVE,
        source=f"executor:{spec.handler}",
        failure_threshold=settings.agent_circuit_failure_threshold,
        failure_ratio=settings.agent_circuit_failure_ratio,
        window_seconds=settings.agent_circuit_window_seconds,
        cooldown_seconds=settings.agent_circuit_cooldown_seconds,
        half_open_max_trials=settings.agent_circuit_half_open_max_trials,
    )


def _get_executor_classes() -> list:
    """惰性获取执行器类列表（db 工具 + 知识库检索工具），避免注册器与执行器循环依赖。"""
    classes = []
    try:
        from agent.tools.db_tools import DbToolExecutor
        classes.append(DbToolExecutor)
    except Exception:
        pass
    try:
        from agent.tools.knowledge_tool import KnowledgeToolExecutor
        classes.append(KnowledgeToolExecutor)
    except Exception:
        pass
    return classes


def _resolve_handler_func(handler: str):
    """按 handler 名在全部执行器类上查找方法（db 工具 / 知识库检索工具）。"""
    for cls in _EXECUTOR_CLASSES:
        fn = getattr(cls, handler, None)
        if fn is not None:
            return fn
    return None


_EXECUTOR_CLASSES = _get_executor_classes()


# =============================================================================
# 3. 幂等 upsert
# =============================================================================

async def sync_tool_registry(db) -> ToolSyncResult:
    """
    启动注册主入口：扫描 + 构建 + 幂等 upsert 工具注册表。

    - 新增工具：插入，status=生效（1），熔断参数取当前配置快照；
    - 已存在工具：仅刷新描述 / 参数 / 风险 / 审批开关 / 来源 / 熔断配置快照，
      **保留 status / circuit_state / 计数**（人工下线与熔断状态不被覆盖）；
    - 仅配置了新增而代码已移除的工具：不自动删除（保留审计，可人工下线）。

    :param db: 数据库会话
    :return: 同步结果统计
    """
    result = ToolSyncResult()
    rows_by_name: Dict[str, ToolRegistry] = {}

    # 2.1 @tool 装饰器工具
    decorated = discover_decorated_tools()
    for name, tool in decorated.items():
        rows_by_name[name] = ToolRegistry(
            name=name,
            description=tool.description,
            parameters=_decorated_tool_schema(tool),
            category="custom",
            risk_level="read",
            requires_approval=0,
            status=ToolStatus.ACTIVE,
            source=_decorated_tool_source(name, decorated),
            failure_threshold=settings.agent_circuit_failure_threshold,
            failure_ratio=settings.agent_circuit_failure_ratio,
            window_seconds=settings.agent_circuit_window_seconds,
            cooldown_seconds=settings.agent_circuit_cooldown_seconds,
            half_open_max_trials=settings.agent_circuit_half_open_max_trials,
        )
    result.decorated_tools = len(decorated)

    # 2.2 ToolSpec 注册表
    specs = list(REGISTRY.values())
    for spec in specs:
        rows_by_name.setdefault(spec.name, _spec_to_row(spec))
    result.registry_tools = len(specs)

    # 2.3 upsert
    try:
        from sqlalchemy import select

        existing_result = await db.execute(select(ToolRegistry))
        existing = {r.name: r for r in existing_result.scalars().all()}
    except Exception as e:
        await db.rollback()
        result.errors.append(f"读取现有工具注册表失败: {e}")
        logger.error(f"[工具注册] 读取现有注册表失败: {e}")
        return result

    for name, new_row in rows_by_name.items():
        old = existing.get(name)
        try:
            if old is None:
                db.add(new_row)
                result.inserted += 1
            else:
                await _refresh_metadata(old, new_row)
                result.updated += 1
        except Exception as e:
            await db.rollback()
            result.errors.append(f"工具 {name} 注册失败: {e}")
            logger.warning(f"[工具注册] 工具 {name} 注册失败: {e}")

    try:
        await db.commit()
    except Exception as e:
        await db.rollback()
        result.errors.append(f"提交失败: {e}")
        logger.error(f"[工具注册] 提交失败: {e}")
        return result

    logger.info(
        f"[工具注册] 同步完成: 扫描模块={result.decorated_tools} 个 @tool 工具, "
        f"ToolSpec={result.registry_tools} 个, 新增={result.inserted}, 更新={result.updated}, "
        f"错误={len(result.errors)}"
    )
    return result


async def _refresh_metadata(old: ToolRegistry, new: ToolRegistry) -> None:
    """
    刷新已有工具的元数据（保留状态与熔断现场）。

    幂等约束：description / parameters / category / risk_level / requires_approval /
    source / 熔断配置快照随代码更新；status / circuit_state / 计数 / 熔断时间戳保持不变。
    """
    old.description = new.description
    old.parameters = new.parameters
    old.category = new.category
    old.risk_level = new.risk_level
    old.requires_approval = new.requires_approval
    old.source = new.source
    # 熔断配置快照随最新配置刷新（若工具当前已熔断，快照用于恢复后参照）
    old.failure_threshold = new.failure_threshold
    old.failure_ratio = new.failure_ratio
    old.window_seconds = new.window_seconds
    old.cooldown_seconds = new.cooldown_seconds
    old.half_open_max_trials = new.half_open_max_trials
