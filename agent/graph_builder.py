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
from agent.streaming import get_sse_queue, put_content
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

# 拼接模式的分段标题（§27.4.1，merge 免 LLM 路径使用）
CONCAT_RAG_HEADER = "【知识依据】"
CONCAT_TASK_HEADER = "【执行结果】"


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
    from agent.intent_router import IntentResult, IntentRouter, get_router
    from agent.intent_gate import KIND_NONE, KIND_SLASH, gate as _gate

    user_input = state["messages"][-1].content
    model = state.get("model", "qwen-turbo")

    # §17.2 跨轮状态隔离：本节点是图入口，**每轮必执行**，因此由它无条件覆盖写入
    # 「本轮计划执行的分支」标记。
    #
    # 为什么必须放在入口节点：LangGraph 检查点按 thread 累积，未执行节点的字段会
    # 保留上一轮旧值。入口节点每轮都跑 → 它的返回值必然覆盖上一轮 → 无需 reducer
    # （单节点写入，不存在并行写冲突），也无需清空任何业务字段。
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
                # 斜杠短路由 chat 节点输出回执（chat 写 rag_answer）
                "executed_branches": ["knowledge"],
                **_cache_fields(["chat"]),
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
                "executed_branches": _planned_branches(result.intents),
                **_cache_fields(result.intents),
            }

    # §18.5：入口预计算 query 双向量（稠密 + 稀疏，一次 API 调用全链路复用）。
    #
    # 原实现里 query_embedding 由**下游**的 knowledge_node 计算并回写 state，
    # 而本节点更早执行，因此传给意图层的向量恒为 None
    # → L2「向量就近原型」判定层从不生效（§16 设计的能力实际是死代码）。
    # 改为入口算一次，供 L2 判定 / 知识检索 / 语义缓存三层复用。
    #
    # 传递方式用 `config["configurable"]` 而**不是** state：1024 维稠密向量写进
    # state 会被检查点（MySQL）逐轮序列化落库，既撑大存储又拖慢 checkpoint 写入；
    # configurable 不进检查点，且同一次 run 内对下游节点可见。
    query_embedding = None
    if settings.cache_embed_prefetch_enabled:
        query_embedding = await _prefetch_query_vectors(user_input, config)

    router = get_router(model)
    try:
        result = await router.classify(
            user_input,
            query_embedding=query_embedding,
            last_intents=state.get("last_intents"),
        )
    except Exception as e:
        # §19.1 最后一道防线：意图判别是「优化项」而非「必经项」，任何漏出来的
        # 异常都不得击穿主链路。线上曾因 max_tokens 被思考 token 吃满导致
        # LengthFinishReasonError 上抛，整条请求 500——这里兜住，降级 chat 继续跑。
        logger.error(
            f"[Agent-意图] 分类异常，降级为 chat: {type(e).__name__}: {e} "
            f"trace_id={get_trace_id()}"
        )
        metrics.incr(f"{K_INTENT_PREFIX}degraded")
        result = IntentResult(
            intents=["chat"], reason=f"意图识别异常降级: {type(e).__name__}", confidence=0
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
        "executed_branches": _planned_branches(result.intents),
        **_cache_fields(result.intents),
    }


def _cache_fields(intents: list) -> dict:
    """
    生成本轮的答案缓存准入字段（§18.2）。

    与 `executed_branches` 同理：由入口节点每轮覆盖写入，避免检查点里
    上一轮的策略残留影响本轮判定。

    容错：本函数在主链路的**必经路径**上（每次 merge 前的路由返回都要走到），
    因此任何异常都不得向上抛——按「保守」原则降级为 DENY（不缓存），
    保证主链路永远能拿到两个字段。
    """
    try:
        from agent.intent_gate import cache_policy_for

        policy, ttl = cache_policy_for(intents)
    except Exception as e:
        # 缓存准入是加速项而非功能项：取不到策略就禁用缓存，不影响作答
        logger.warning(f"[Agent-意图] 缓存策略计算失败，降级为不缓存: {e}")
        policy, ttl = "deny", 0
    return {"cache_policy": policy, "cache_ttl_seconds": ttl}


async def _prefetch_query_vectors(user_input: str, config) -> Optional[list]:
    """
    预计算 query 稠密 + 稀疏向量，写入 configurable 供下游复用（§18.5）。

    三个消费方：
    1. 意图层 L2「向量就近原型」判定（此前恒为 None，从未生效）；
    2. knowledge_node 的混合检索（稠密 + 稀疏）；
    3. 语义缓存的相似查询（复用预计算向量，省一次 embedding）。

    best-effort：失败时返回 None，下游各自按需重算，不影响主链路。

    :return: 稠密向量（失败为 None）
    """
    try:
        import asyncio

        from embeddings.embedding_deal import DashScopeEmbedding

        dense, sparse = await asyncio.to_thread(
            DashScopeEmbedding(api_key=settings.api_key).embed_query_with_sparse,
            user_input,
        )
    except Exception as e:
        logger.warning(f"[Agent-意图] query 向量预计算失败（下游按需重算）: {e}")
        return None

    # 写入 configurable（不进检查点），供 knowledge_node 复用
    if config is not None:
        try:
            cfg = config.setdefault("configurable", {})
            if isinstance(cfg, dict):
                cfg["query_embedding"] = dense
                cfg["query_sparse"] = sparse
        except Exception as e:
            logger.debug(f"[Agent-意图] 向量写入 configurable 失败（下游按需重算）: {e}")
    return dense


def _planned_branches(intents: list) -> list:
    """
    根据本轮意图推导「计划执行的分支名」（§17.2 跨轮状态隔离）。

    映射规则与 `route_by_intents` 保持一致：
    - chat / knowledge_base 单分支 → 由 chat 或 knowledge 节点产出 rag_answer，
      故统一记为 "knowledge"（两者都写 rag_answer 字段）；
    - task 参与时 → 走 task_agent 节点，记为 "task_agent"。

    注意：返回的是**计划**而非**实际**。节点执行异常时会返回 `rag_answer: None`
    （见 knowledge_node 的 except 分支），merge 的 `if rag and task` 仍不成立，
    因此「计划」判据与实际判据在此场景下等价，无需额外区分。
    """
    intents = set(intents or [])
    planned: list = []
    if "chat" in intents or "knowledge_base" in intents:
        planned.append("knowledge")
    if "task" in intents:
        planned.append("task_agent")
    return planned


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

    意图组合规则（设计文档 §5.2 / §27.3）：
    - 仅 ["chat"] → chat
    - 仅 ["knowledge_base"] / 仅 ["task"] → 单分支
    - ["task","knowledge_base"] → **串行**：先 knowledge，再由
      `route_after_knowledge` 接力进 task_agent（§27.3.1）
    - 异常/低置信度（意图识别已降级为 chat）→ chat

    §27.3 为什么由并行改为串行：
    1. 并行时两分支各自检索知识库，任务分支看不到知识分支的产出，只能重复检索；
    2. 并行结果必须靠一次额外的 LLM 合并（`agent_merge_timeout_seconds` 60s
       × 重试 1 次 = 最坏 120s，且位于 DAG 必经汇聚点）；
    3. 并行审批中断会**取消**仍在执行的知识分支，恢复后需 `_recover_knowledge_branch`
       补跑（§14.2）；串行时知识分支早已落检查点，天然可用。
    串行后 `merge` 仍按 `executed_branches` 判定（§17.2），两分支都被记为已执行。

    回滚：`agent_multi_intent_serial=False` 时恢复原并行 fan-out。
    """
    intents = set(state.get("intents") or ["chat"])
    if intents == {"task", "knowledge_base"}:
        if not _is_node_registered(NODE_TASK_AGENT):
            # 任务节点未注册（阶段 1 主图）→ 退化为仅知识库
            logger.info(
                f"[Agent-路由] {sorted(intents)} -> {NODE_KNOWLEDGE}（任务节点未注册） "
                f"trace_id={get_trace_id()}"
            )
            return NODE_KNOWLEDGE
        if getattr(settings, "agent_multi_intent_serial", True):
            logger.info(
                f"[Agent-路由] {sorted(intents)} -> 串行 "
                f"{NODE_KNOWLEDGE} → {NODE_TASK_AGENT} trace_id={get_trace_id()}"
            )
            return NODE_KNOWLEDGE
        logger.info(
            f"[Agent-路由] {sorted(intents)} -> 并行 "
            f"[{NODE_KNOWLEDGE}, {NODE_TASK_AGENT}] trace_id={get_trace_id()}"
        )
        return [NODE_KNOWLEDGE, NODE_TASK_AGENT]
    if "task" in intents:
        target = NODE_TASK_AGENT if _is_node_registered(NODE_TASK_AGENT) else NODE_CHAT
    elif "knowledge_base" in intents:
        target = NODE_KNOWLEDGE
    else:
        target = NODE_CHAT
    logger.info(f"[Agent-路由] {sorted(intents)} -> {target} trace_id={get_trace_id()}")
    return target


def route_after_knowledge(state: AgentState) -> str:
    """
    知识分支执行后的去向（§27.3.2）。

    **判据只看意图与节点可用性，不看知识分支是否成功。** 知识分支失败
    （`rag_answer=None`，见 knowledge_node 的 except 分支）时仍必须进任务分支：
    任务分支自带 `search_knowledge`，可以自行补检索自愈；若因失败改道 merge，
    会**直接丢掉任务意图**（用户明确要求的数据操作不会被执行）。

    非多意图（仅 knowledge_base / chat）→ 返回 merge，与串行改造前一致。

    :return: 下一节点名（"task_agent" 或 "merge"）
    """
    intents = set(state.get("intents") or [])
    if (
        "task" in intents
        and _is_node_registered(NODE_TASK_AGENT)
        and getattr(settings, "agent_multi_intent_serial", True)
    ):
        logger.info(f"[Agent-路由] 知识分支完成 -> {NODE_TASK_AGENT}（串行接力）")
        return NODE_TASK_AGENT
    return NODE_MERGE


_registered_nodes: set = set()


def _register_node(name: str) -> None:
    """标记某节点已注册到图。"""
    _registered_nodes.add(name)


def _is_node_registered(name: str) -> bool:
    """判断节点是否已注册。"""
    return name in _registered_nodes


def _pushed_contents(rag: Optional[str], task: Optional[str]) -> list:
    """
    本轮分支节点**已经流式推送过**的完整内容（§27.6）。

    merge 是 barrier，此处读到的 rag / task 就是各分支实际推给用户的内容
    （task 分支推送时已按 `delivered_chars` 裁切，见 task_node §21.5），
    且已按 `executed_branches` 过滤掉跨轮残留，因此可直接作为去重集合。

    ⚠️ 本字段必须在 `AgentState` 里声明才有效 —— LangGraph 只认识 schema 声明的
    channel，未声明字段的写入会被**静默丢弃**（不报错）。

    :param rag: 知识库/聊天分支产出（本轮实际执行时）
    :param task: 任务分支产出（本轮实际执行时）
    :return: 已推送内容列表（保序：知识段在前）
    """
    return [c for c in (rag, task) if c]


def _concat_answers(rag: str, task: str) -> str:
    """
    拼接两段分支产出（§27.4.1，merge 的免 LLM 路径）。

    与降级裸拼接 `f"{rag}\\n\\n{task}"` 的区别：加固定小标题，让用户分得清
    「依据」与「结果」；任一段缺失时退化为单段输出，不产生空标题。

    :param rag: 知识库分支产出
    :param task: 任务分支产出
    :return: 拼接后的最终回答
    """
    if not rag:
        return task or ""
    if not task:
        return rag
    return f"{CONCAT_RAG_HEADER}\n{rag}\n\n{CONCAT_TASK_HEADER}\n{task}"


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

    # ---------------------------------------------------------------------
    # §17.2 跨轮状态隔离：按「本轮实际计划执行的分支」判定，而非「字段是否非空」。
    #
    # 修复前的问题（实测 17.4s 白付 + 串味幻觉）：检查点按 thread 累积，本轮只路由
    # task 单分支时，上一轮 knowledge/chat 分支写入的 rag_answer 仍留在 channel 里，
    # 于是 `if rag and task` 成立 → 白跑一次合并 LLM，且用的是与本轮问题**完全无关**
    # 的历史内容（本次是上一轮的「退货流程」）。
    #
    # 修复方式：用入口节点每轮覆盖写入的 executed_branches 为准。历史残留被显式忽略，
    # 但**不删除**——并行审批中断场景下 rag_answer 必须保留给 resume 合并（§17.2.4）。
    # ---------------------------------------------------------------------
    executed = set(state.get("executed_branches") or [])
    if settings.agent_merge_cross_turn_guard and executed:
        ran_kb = "knowledge" in executed
        ran_task = "task_agent" in executed
        if not ran_kb and rag:
            # 本轮没跑知识库/聊天分支，却读到了非空 rag_answer —— 必为跨轮残留。
            metrics.incr("agent.merge.stale_rag_ignored")
            logger.warning(
                f"[Agent-汇总] 忽略跨轮残留的 rag_answer（本轮未执行该分支）"
                f" len={len(rag)} trace_id={get_trace_id()}"
            )
        if not ran_task and task:
            metrics.incr("agent.merge.stale_task_ignored")
            logger.warning(
                f"[Agent-汇总] 忽略跨轮残留的 task_answer（本轮未执行该分支）"
                f" len={len(task)} trace_id={get_trace_id()}"
            )
        rag = rag if ran_kb else None
        task = task if ran_task else None

    # 审批中断恢复容错：知识库分支被并行中断取消时重新执行以恢复结果。
    #
    # §27.3.3：**串行模式下必须禁用**。串行时知识分支在任务分支之前已经跑完，
    # 失败就是失败；若在此处补跑，等于把「重复生成」从流程开头搬到了结尾，
    # 而且发生在用户已看到任务结果之后，体验更差。
    # 保留该分支仅为 `agent_multi_intent_serial=False`（并行回滚路径）服务。
    if (
        not rag
        and "knowledge_base" in intents
        and not getattr(settings, "agent_multi_intent_serial", True)
    ):
        rag = await _recover_knowledge_branch(state, config)

    pushed = _pushed_contents(rag, task)

    if rag and task:
        # §27.4：默认走拼接（免 LLM）。
        # 依据：注入改造完成后，知识结论已进入任务分支的上下文，两段内容高度重叠，
        # 此时 LLM 合并做的其实主要是「去重 + 加过渡词」，性价比反转。
        if not getattr(settings, "agent_merge_llm_enabled", False):
            metrics.incr("agent.merge.mode.concat")
            logger.info(
                f"[Agent-汇总] 拼接模式（未调用 LLM）: rag={len(rag)}, task={len(task)} "
                f"trace_id={get_trace_id()}"
            )
            return {
                "final_response": _concat_answers(rag, task),
                "pushed_contents": pushed,
                # 拼接结果 = 两段已推内容的直接组合 → 抑制 finalize 补推
                "final_push_suppressed": True,
            }

        metrics.incr("agent.merge.mode.llm")
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
                return {
                    "final_response": resp.content,
                    "pushed_contents": pushed,
                    # LLM 产出的是**新文本**（含综合建议段），与已推两段不同 → 需补推
                    "final_push_suppressed": False,
                }
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
        # §27.6：降级拼接同样是「两段已推内容的组合」→ 一并抑制补推，
        # 否则 LLM 抖动时会退化成「用户看两遍」。
        return {
            "final_response": _concat_answers(rag, task),
            "pushed_contents": pushed,
            "final_push_suppressed": True,
        }

    metrics.incr("agent.merge.mode.passthrough")
    logger.info(
        f"[Agent-汇总] 单分支结果透传: 有知识库={bool(rag)}, 有任务={bool(task)} "
        f"本轮分支={sorted(executed) if executed else '未知'} "
        f"trace_id={get_trace_id()}"
    )
    # §27.6：单分支时 final_response 来自唯一已推过的分支内容（或兜底文案）：
    # - 命中分支内容 → 已推过，抑制补推；
    # - 落到「暂无可用结果」兜底文案 → 是**新文本**，必须补推（让用户看到失败原因）。
    fallback = rag or task
    return {
        "final_response": fallback or "暂无可用结果",
        "pushed_contents": pushed,
        "final_push_suppressed": bool(fallback),
    }


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

    §20 F2.4：补推 final_response。
    各分支节点都会在自己结束时推一次内容，但**合并后的最终回答只在这里产生**
    （merge 节点产出 final_response，但原实现不推送）。并行双意图场景下，
    用户看到的是 knowledge + task 两段原始内容，缺少「综合建议」这一段；
    某些降级路径（分支静默失败、error 兜底）更是**一条内容都不会推**，
    前端只能等到 done 事件，界面上一片空白。此处按开关统一补推一次。

    §27.6 去重（本次修复，三重失效全部堵上）：
    1. `pushed_contents` 在 `AgentState` 中**声明**（否则写入被静默丢弃）；
    2. 由 `merge_node` 每轮覆盖写入（单点写，避免并行写冲突）；
    3. 本节点据此跳过补推，并在 merge 标记 `final_push_suppressed` 时整体跳过。
    - `final_push_suppressed=True`（拼接/降级拼接）：final_response 就是两段已推内容的
      直接组合，补推必然重复 → 跳过；
    - 兜底文案（无 final_response 的 error 分支）：是用户唯一能看到失败原因的机会
      → **不受抑制标记影响**，照常推送。
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
        # 兜底文案同样需要推送 —— 这是用户唯一能看到失败原因的机会
        await _push_final_response(config, final, state)
        return {"final_response": final}

    if state.get("final_push_suppressed"):
        metrics.incr("agent.finalize.push_suppressed")
        logger.info(
            f"[Agent-收尾] final_response 由已推内容拼接而成，跳过补推 "
            f"len={len(final)} trace_id={get_trace_id()}"
        )
        return {}

    await _push_final_response(config, final, state)
    return {}


def _already_pushed_branches(state: AgentState) -> set[str]:
    """
    收集本轮**已由分支节点推送过**的内容（用于 finalize 去重）。

    §27.6 修复：本函数此前是**死代码**（全仓无调用点），且其数据来源
    `state["pushed_contents"]` 未在 `AgentState` 中声明 —— 而 LangGraph 对未声明
    字段的写入是**静默丢弃**的，因此 `task_node` 写了也没用。两处叠加导致
    §20 F2.4 宣称的去重机制从未生效。现在由 `merge_node` 每轮覆盖写入该字段，
    并在 `_push_final_response` 中真正参与判定。

    按 strip 后的文本比对；字段缺失/类型异常时返回空集，退化为「总是补推」
    —— 宁可多推一次，也不要让用户什么都看不到。
    """
    pushed = state.get("pushed_contents")
    if not isinstance(pushed, (list, tuple, set)):
        return set()
    return {str(c).strip() for c in pushed if c}


async def _push_final_response(
    config: RunnableConfig | None, content: str, state: AgentState | None = None
) -> None:
    """
    补推最终回答（§20 F2.4 / §27.6，best-effort）。

    开关 `agent_push_final_response` 关闭时直接返回（逐字节回到改动前行为）。

    去重判据（§27.6）：`content` 与分支节点已推内容相同时不推，避免用户把
    同一段文字看两遍。`state` 为 None 时跳过去重（老调用点兼容）。

    :param config: 运行配置（取 sse_queue）
    :param content: 待补推内容
    :param state: 图状态（用于读取已推内容集合）
    """
    if not content:
        return
    if not getattr(settings, "agent_push_final_response", True):
        return
    if state is not None and content.strip() in _already_pushed_branches(state):
        metrics.incr("agent.finalize.push_skipped_duplicate")
        logger.info(
            f"[Agent-收尾] final_response 与已推内容重复，跳过补推 "
            f"len={len(content)} trace_id={get_trace_id()}"
        )
        return
    queue = get_sse_queue(config)
    if queue is None:
        return
    try:
        await put_content(queue, content, branch="final")
    except Exception as e:
        # 推送失败不得影响图执行收尾（与分支节点的容错策略一致）
        logger.warning(f"[Agent-收尾] final_response 补推失败（非致命）: {e}")


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

        # ------------------------------------------------------------------
        # §25 P2-2：rag_answer 只在「本轮真的跑过知识库分支」时才回写。
        #
        # merge_node 已按 executed_branches 把跨轮残留的 rag_answer 判为无效并
        # 置空（§17.2），但本函数读的是**未经 merge 清洗的原始 state** ——
        # 于是上一轮知识库分支的答案被写进了本轮 task_execution.rag_answer，
        # 审计里出现「本轮任务是删除用户，rag_answer 却是退货流程」的串味记录，
        # 复盘时极易误判「任务分支也回答了知识库问题」。
        #
        # 判据与 merge_node 完全对齐：`executed_branches` 为空时视为「未知」，
        # 按放行处理（保守，避免拦掉真实结果）。
        # ------------------------------------------------------------------
        executed = set(state.get("executed_branches") or [])
        stale_rag = bool(
            settings.agent_merge_cross_turn_guard
            and executed
            and "knowledge" not in executed
        )
        if stale_rag and state.get("rag_answer"):
            metrics.incr("agent.finalize.stale_rag_skipped")
            logger.warning(
                f"[Agent-收尾] 跳过回写跨轮残留的 rag_answer（本轮未执行知识库分支）"
                f" len={len(state['rag_answer'])} trace_id={get_trace_id()}"
            )
        if state.get("rag_answer") and not stale_rag:
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

    # §27.3.2：知识分支的出边改为**条件边** —— 多意图串行时由它接力进任务分支。
    # ⚠️ 不能再对同一节点同时添加固定边与条件边（LangGraph 视为重复边）。
    # 判据见 `route_after_knowledge`：只看意图，不看知识分支成败（失败也必须进任务自愈）。
    if _is_node_registered(NODE_KNOWLEDGE):
        kb_map = {NODE_MERGE: NODE_MERGE}
        if _is_node_registered(NODE_TASK_AGENT):
            kb_map[NODE_TASK_AGENT] = NODE_TASK_AGENT
        g.add_conditional_edges(NODE_KNOWLEDGE, route_after_knowledge, kb_map)

    # 其余分支统一汇入 merge（barrier）
    for name in (NODE_CHAT, NODE_TASK_AGENT):
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
