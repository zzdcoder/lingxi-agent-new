"""
主图组装

基于 StateGraph 构建 Agent 主图：
intent_router → 条件路由（knowledge / chat / task_agent）→ merge → finalize → END

要点：
- route_by_intents 返回节点名**列表**时，LangGraph 在同一 superstep 并行执行多个节点（多意图并行 fan-out）；
- merge 节点有多条入边，LangGraph 在所有前驱完成后才执行（barrier 自动汇聚）；
- task_agent 节点采用惰性注册：任务节点模块（agent/nodes/task_node.py）尚不可用时自动跳过，
  保证阶段 1 主图（知识库/聊天）可独立运行。

**观测与加固（阶段 4，设计文档 §15）**：
- 全部节点挂载 `agent.observability.node_trace`：记录调用量 / 耗时 / 异常，消除零日志盲区；
- merge 汇总 LLM 补配超时（agent_merge_timeout_seconds）与有限重试，避免汇聚点挂死拖垮整条链路；
- finalize 节点将 `final_response` / `rag_answer` 快照回写 task_execution（补齐 §8.1 审计字段）。
"""
import logging
from typing import Optional, Union
from urllib.parse import quote_plus

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END

from agent.state import AgentState
from core.config import settings
from agent.observability import (
    K_INTENT_PREFIX,
    get_trace_id,
    metrics,
    node_trace,
)

logger = logging.getLogger(__name__)

# 节点名常量
NODE_INTENT_ROUTER = "intent_router"
NODE_KNOWLEDGE = "knowledge"
NODE_CHAT = "chat"
NODE_TASK_AGENT = "task_agent"
NODE_MERGE = "merge"
NODE_FINALIZE = "finalize"


# =============================================================================
# 节点导入（惰性，避免阶段间强依赖）
# =============================================================================

def _import_node_modules() -> dict:
    """导入已实现的节点模块（缺失时跳过，保证图可逐步扩展）。"""
    nodes = {}
    try:
        from agent.nodes.knowledge_node import knowledge_node
        from agent.nodes.chat_node import chat_node
        nodes[NODE_KNOWLEDGE] = knowledge_node
        nodes[NODE_CHAT] = chat_node
    except ImportError as e:
        logger.warning(f"基础节点导入失败: {e}")
    try:
        from agent.nodes.task_node import task_agent_node
        nodes[NODE_TASK_AGENT] = task_agent_node
    except ImportError:
        logger.info("任务节点未注册（阶段 2 尚未实现），task 意图暂路由到 chat 兜底")
    return nodes


# =============================================================================
# 节点实现
# =============================================================================

@node_trace(NODE_INTENT_ROUTER)
async def intent_router_node(
    state: AgentState, config: RunnableConfig | None = None
) -> dict:
    """
    意图识别节点：分层判定（规则门控 → 决策缓存 → 向量就近 → LLM 兜底）。

    §16 优化：规则层命中时**完全不调用 LLM**（斜杠命令、寒暄、数据库强信号、
    短追问继承），LLM 只负责真正需要语义理解的长尾输入。

    :param state: 图状态
    :param config: 运行配置
    :return: 状态增量（intents / intent_reason / intent_confidence）
    """
    from agent.intent_router import IntentRouter
    from agent.intent_gate import KIND_NONE, KIND_SLASH, gate as _gate

    user_input = state["messages"][-1].content
    model = state.get("model", "qwen-turbo")

    # 斜杠命令短路：不进 LLM、不进知识库/任务分支，直接由 chat 分支输出指令说明。
    # （Hermes 第一层「命令路由」的等价实现；命令名不在内置表内时 gate 返回 NONE。）
    if settings.intent_gate_enabled and settings.intent_slash_enabled:
        decision = _gate(user_input, last_intents=state.get("last_intents"))
        if decision.kind == KIND_SLASH:
            metrics.incr(f"{K_INTENT_PREFIX}slash")
            handoff = _slash_handoff(decision.slash_name or "", decision.slash_args or "")
            logger.info(
                f"[Agent-意图] 斜杠命令短路 /{decision.slash_name} "
                f"trace_id={get_trace_id()}"
            )
            # 把指令说明写入 state.reason，chat 分支据此作答（见 chat_node 的 slash 分支）
            return {
                "intents": ["chat"],
                "intent_reason": decision.reason,
                "intent_confidence": 1.0,
                "slash_command": decision.slash_name,
                "slash_args": decision.slash_args,
                "slash_handoff": handoff,
            }
        if decision.kind != KIND_NONE:
            # 规则层其它命中（寒暄/强信号/追问继承）：直接给确定性意图，省掉 LLM
            metrics.incr(f"{K_INTENT_PREFIX}{decision.rule}")
            result = IntentRouter._to_result(decision)
            logger.info(
                f"[Agent-意图] 规则命中 rule={decision.rule} intents={result.intents} "
                f"trace_id={get_trace_id()}"
            )
            for intent in result.intents:
                metrics.incr(f"{K_INTENT_PREFIX}{intent}")
            return {
                "intents": result.intents,
                "intent_reason": result.reason,
                "intent_confidence": result.confidence,
            }

    router = IntentRouter(model=model)
    result = await router.classify(
        user_input,
        query_embedding=state.get("query_embedding"),
        last_intents=state.get("last_intents"),
    )
    logger.info(
        f"[Agent-意图] intents={result.intents}, "
        f"confidence={result.confidence:.2f}, reason={result.reason[:80]} "
        f"trace_id={get_trace_id()}"
    )
    # 意图分布观测（多为粗粒度标签，避免高基数）
    for intent in result.intents:
        metrics.incr(f"{K_INTENT_PREFIX}{intent}")
    return {
        "intents": result.intents,
        "intent_reason": result.reason,
        "intent_confidence": result.confidence,
    }


# 斜杠命令说明表（与 agent/intent_gate.SLASH_COMMANDS 对齐，仅用于回执文案）
_SLASH_HELP: dict[str, str] = {
    "clear": "「清空上下文」请使用前台按钮或 `/api/conversations` 接口，本服务不在服务端清除会话数据。",
    "new": "「新建会话」请在前台点击「新建对话」，本服务会自动分配新的会话 ID。",
    "stop": "如需中止当前生成，请直接关闭应答卡片或刷新页面；服务端会随连接断开释放资源。",
    "help": "可用指令：`/help`、`/status`、`/clear`、`/new`、`/stop`。直接在输入框用自然语言提问也可以。",
    "status": "服务状态查询请访问 `GET /health?detailed=true`；运行指标请访问 `GET /api/agent/metrics`。",
}


def _slash_handoff(name: str, args: str) -> str:
    """生成斜杠命令的本地回执（免 LLM，保证指令类输入零延迟）。"""
    tip = _SLASH_HELP.get(name)
    if tip:
        return tip
    return f"暂不支持的指令 `/{name}`。{_SLASH_HELP['help']}"



def route_by_intents(state: AgentState) -> Union[str, list[str]]:
    """
    根据多意图返回路由目标。

    - 返回节点名**列表**时，LangGraph 会在同一 superstep 并行执行多个节点；
    - 返回单个节点名则走单分支。

    意图组合规则（设计文档 §5.2）：
    - 仅 ["chat"] → chat
    - 仅 ["knowledge_base"] / 仅 ["task"] → 单分支
    - ["task","knowledge_base"] → 并行 fan-out
    - 异常/低置信度（意图识别已降级为 chat）→ chat
    """
    intents = set(state.get("intents") or ["chat"])
    if intents == {"task", "knowledge_base"}:
        # 并行分支：task_agent 未注册时退化为仅知识库
        if _is_node_registered(NODE_TASK_AGENT):
            target = [NODE_KNOWLEDGE, NODE_TASK_AGENT]
        else:
            target = NODE_KNOWLEDGE
        logger.info(f"[Agent-路由] {sorted(intents)} -> {target} trace_id={get_trace_id()}")
        return target
    if "task" in intents:
        target = NODE_TASK_AGENT if _is_node_registered(NODE_TASK_AGENT) else NODE_CHAT
    elif "knowledge_base" in intents:
        target = NODE_KNOWLEDGE
    else:
        target = NODE_CHAT
    logger.info(f"[Agent-路由] {sorted(intents)} -> {target} trace_id={get_trace_id()}")
    return target


_registered_nodes: set = set()


def _register_node(name: str) -> None:
    """标记某节点已注册到图。"""
    _registered_nodes.add(name)


def _is_node_registered(name: str) -> bool:
    """判断节点是否已注册。"""
    return name in _registered_nodes


@node_trace(NODE_MERGE)
async def merge_node(
    state: AgentState, config: RunnableConfig | None = None
) -> dict:
    """
    汇总节点（barrier）：合并知识库分支（rag_answer）与任务分支（task_answer）。

    策略：
    1. 仅一个分支有结果 → 直接作为 final_response（免 LLM 调用）；
    2. 两个分支均有结果 → LLM 合并润色（知识库结论 + 任务结果 + 综合建议）；
    3. 分支失败/无结果 → 容错输出剩余分支结果或可读错误；
    4. 审批恢复容错：并行（task+knowledge）场景下任务分支先触发审批中断会
       取消仍在执行的知识分支（设计文档 §14.2），检测到知识分支结果缺失时
       重新执行该分支，保证恢复后汇总完整。
    """
    rag = state.get("rag_answer")
    task = state.get("task_answer")
    intents = state.get("intents") or []

    # 审批中断恢复容错：知识库分支被并行中断取消时重新执行以恢复结果
    if not rag and "knowledge_base" in intents:
        rag = await _recover_knowledge_branch(state, config)

    if rag and task:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI
        from prompt.prompt_storage import MERGE_ANSWERS_PROMPT

        question = state["messages"][-1].content
        # 超时 + 有限重试：汇聚点为 DAG 必经之路，挂死会拖垮整条链路，
        # 期间 LLM 抖动（偶发超时/5xx）以重试自愈，重试耗尽才降级为字符串拼接。
        attempts = max(0, settings.agent_merge_llm_retries) + 1
        last_error: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                llm = ChatOpenAI(
                    model=state.get("model", "qwen-turbo"),
                    openai_api_key=settings.api_key,
                    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
                    temperature=0.3,
                    timeout=settings.agent_merge_timeout_seconds,
                )
                prompt = ChatPromptTemplate.from_messages([
                    ("system", MERGE_ANSWERS_PROMPT),
                    ("human", "用户问题：{question}\n知识库结论：{rag}\n任务执行结果：{task}"),
                ])
                resp = await (prompt | llm).ainvoke({
                    "question": question, "rag": rag, "task": task,
                })
                logger.info(
                    f"[Agent-汇总] LLM 合并完成 attempt={attempt} "
                    f"trace_id={get_trace_id()}"
                )
                return {"final_response": resp.content}
            except Exception as e:
                last_error = e
                logger.warning(
                    f"[Agent-汇总] LLM 合并失败 attempt={attempt}/{attempts}: {e} "
                    f"trace_id={get_trace_id()}"
                )
                metrics.incr("agent.merge.llm_retry")
        logger.warning(
            f"[Agent-汇总] LLM 合并最终失败，回退拼接: {last_error} "
            f"trace_id={get_trace_id()}"
        )
        return {"final_response": f"{rag}\n\n{task}"}

    logger.info(
        f"[Agent-汇总] 单分支结果透传: 有知识库={bool(rag)}, 有任务={bool(task)} "
        f"trace_id={get_trace_id()}"
    )
    return {"final_response": rag or task or "暂无可用结果"}


async def _recover_knowledge_branch(
    state: AgentState, config: RunnableConfig | None
) -> Optional[str]:
    """
    重新执行知识库分支以恢复 rag_answer（并行中断取消场景）。

    复用 knowledge_node 节点逻辑，但抑制其 SSE 流式输出——恢复后的合并
    结果由上层（审批服务 resume_graph / 正常流）统一推送，避免重复内容。
    恢复失败时返回 None，merge 继续按容错策略输出剩余分支结果。
    """
    try:
        from agent.nodes.knowledge_node import knowledge_node

        quiet_config = config
        if config is not None and "configurable" in config:
            quiet_config = {
                **config,
                "configurable": {**config["configurable"], "sse_queue": None},
            }
        out = await knowledge_node(state, quiet_config)
        return out.get("rag_answer")
    except Exception as e:
        logger.warning(f"[Agent-汇总] 恢复知识库分支失败: {e}")
        return None


@node_trace(NODE_FINALIZE)
async def finalize_node(
    state: AgentState, config: RunnableConfig | None = None
) -> dict:
    """
    收尾节点：保证 final_response 产出，并将结果快照回写任务审计记录。

    落库内容（设计文档 §8.1）：
    - `result`：final_response（含 merge 合并润色结果）；
    - `rag_answer`：知识库分支产出（并行场景快照，供审计/回溯）。

    说明：知识库/聊天单分支路由时不创建任务记录，此处自然跳过，不报错。
    """
    # 审计快照回写（best-effort：失败仅告警，不影响响应产出）
    await _persist_final_snapshot(config, state)

    final = state.get("final_response")
    if not final:
        error = state.get("error")
        final = f"处理失败：{error}" if error else "暂无可用结果"
        logger.info(
            f"[Agent-收尾] 无 final_response，兜底输出 error={state.get('error')} "
            f"trace_id={get_trace_id()}"
        )
        return {"final_response": final}
    return {}


async def _persist_final_snapshot(
    config: RunnableConfig | None, state: AgentState
) -> None:
    """
    将最终回答与知识库分支快照写入 task_execution（补齐审计字段）。

    定位策略：优先使用状态中的 task_execution_id；缺失时取该会话最近一条
    running 记录（任务分支已在其中写 tool_calls / task_answer）。
    """
    db = (config or {}).get("configurable", {}).get("db") if config else None
    if db is None:
        return
    conversation_id = state.get("conversation_id")
    task_id = state.get("task_execution_id")
    if not conversation_id and not task_id:
        return
    try:
        from sqlalchemy import select

        from models.task_model import TaskExecution

        stmt = select(TaskExecution)
        if task_id:
            stmt = stmt.where(TaskExecution.id == task_id)
        else:
            stmt = stmt.where(
                TaskExecution.conversation_id == conversation_id,
                TaskExecution.status == "running",
            )
        result = await db.execute(
            stmt.order_by(TaskExecution.created_at.desc()).limit(1)
        )
        record = result.scalars().first()
        if record is None:
            return
        if state.get("final_response"):
            record.result = state["final_response"]
        if state.get("rag_answer"):
            record.rag_answer = state["rag_answer"]
        await db.commit()
        logger.debug(f"[Agent-收尾] 审计快照已回写: task_id={record.id}")
    except Exception as e:
        try:
            await db.rollback()
        except Exception:
            pass
        logger.warning(f"[Agent-收尾] 审计快照回写失败（非致命）: {e}")


# =============================================================================
# 主图构建
# =============================================================================

def build_graph(checkpointer=None):
    """
    构建 Agent 主图。

    :param checkpointer: LangGraph 检查点（阶段 3 挂载 MySQLAsyncSaver）
    :return: 编译后的图
    """
    g = StateGraph(AgentState)

    nodes = _import_node_modules()
    _registered_nodes.clear()

    g.add_node(NODE_INTENT_ROUTER, intent_router_node)
    if NODE_KNOWLEDGE in nodes:
        g.add_node(NODE_KNOWLEDGE, nodes[NODE_KNOWLEDGE])
        _register_node(NODE_KNOWLEDGE)
    if NODE_CHAT in nodes:
        g.add_node(NODE_CHAT, nodes[NODE_CHAT])
        _register_node(NODE_CHAT)
    if NODE_TASK_AGENT in nodes:
        g.add_node(NODE_TASK_AGENT, nodes[NODE_TASK_AGENT])
        _register_node(NODE_TASK_AGENT)

    g.add_node(NODE_MERGE, merge_node)
    g.add_node(NODE_FINALIZE, finalize_node)

    g.add_edge(START, NODE_INTENT_ROUTER)

    # 条件路由：映射值为节点名或节点名列表（列表 → 并行 fan-out）
    path_map = {}
    if _is_node_registered(NODE_KNOWLEDGE):
        path_map[NODE_KNOWLEDGE] = NODE_KNOWLEDGE
    if _is_node_registered(NODE_TASK_AGENT):
        path_map[NODE_TASK_AGENT] = NODE_TASK_AGENT
    if _is_node_registered(NODE_CHAT):
        path_map[NODE_CHAT] = NODE_CHAT
    g.add_conditional_edges(NODE_INTENT_ROUTER, route_by_intents, path_map)

    # 三条分支统一汇入 merge（barrier）
    for name in (NODE_KNOWLEDGE, NODE_CHAT, NODE_TASK_AGENT):
        if _is_node_registered(name):
            g.add_edge(name, NODE_MERGE)

    g.add_edge(NODE_MERGE, NODE_FINALIZE)
    g.add_edge(NODE_FINALIZE, END)

    return g.compile(checkpointer=checkpointer)


# =============================================================================
# 全局图单例
# =============================================================================

_graph = None
_checkpointer = None
_saver_ctx = None


def set_checkpointer(checkpointer) -> None:
    """设置全局检查点（阶段 3 由 lifespan 注入 MySQLAsyncSaver）。"""
    global _checkpointer
    _checkpointer = checkpointer


def get_checkpointer():
    """获取全局检查点（None 表示未启用审批模式）。"""
    return _checkpointer


def _mysql_url() -> str:
    """构造 langgraph MySQL 检查点连接串（aiomysql 驱动）。"""
    return (
        f"mysql+aiomysql://{settings.db_user}:{quote_plus(settings.db_password)}"
        f"@{settings.db_host}:{settings.db_port}/{settings.db_name}"
    )


async def init_checkpointer() -> bool:
    """
    初始化 MySQL 检查点（lifespan 启动时调用）。

    失败降级为无审批模式：返回 False，主图不挂检查点、任务 Agent 不绑定写工具。
    """
    global _checkpointer, _saver_ctx
    if _checkpointer is not None:
        return True
    try:
        from langgraph.checkpoint.mysql.aio import AIOMySQLSaver

        _saver_ctx = AIOMySQLSaver.from_conn_string(_mysql_url())
        saver = await _saver_ctx.__aenter__()
        await saver.setup()
        _checkpointer = saver
        try:
            get_graph(rebuild=True)
        except Exception as e:
            logger.warning(f"Agent 主图重建失败（将按需重试）: {e}")
        logger.info("审批检查点初始化完成（MySQL）")
        return True
    except Exception as e:
        logger.warning(f"审批检查点初始化失败（降级为无审批模式）: {e}")
        _saver_ctx = None
        _checkpointer = None
        return False


async def close_checkpointer() -> None:
    """关闭检查点连接（lifespan 退出时调用）。"""
    global _checkpointer, _saver_ctx
    if _saver_ctx is not None:
        try:
            await _saver_ctx.__aexit__(None, None, None)
        except Exception:
            pass
        _saver_ctx = None
    _checkpointer = None


def get_graph(rebuild: bool = False):
    """
    获取编译后的主图（惰性构建 + 缓存）。

    :param rebuild: 强制重建（依赖注册变化时使用）
    :return: 编译后的图
    """
    global _graph
    if _graph is None or rebuild:
        _graph = build_graph(checkpointer=_checkpointer)
        logger.info("Agent 主图编译完成")
    return _graph
