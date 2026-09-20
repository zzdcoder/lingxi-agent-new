"""
意图识别器

对用户输入进行多标签意图分类（knowledge_base / task / chat），
使用 ChatOpenAI 的官方结构化输出 API，并内置降级与归一化策略。

**分层判定（2026-09-18 优化，设计文档 §16）**

原实现为「每次请求一次同步 LLM 分类」，存在三个问题：往返延迟全额计入首字
时间、确定性输入（寒暄/斜杠/强信号）也被迫付 LLM 成本、单点误判直接走错分支。
现改为四层递进，**LLM 只做「真正需要语义理解」的那部分**：

| 层 | 机制 | LLM 成本 | 典型命中率 |
| -- | ---- | ------- | --------- |
| L0 | `agent.intent_gate` 规则门控（斜杠/寒暄/强信号/短追问继承） | 0 | 15~25% |
| L1 | 进程内决策缓存（同会话重复输入） | 0 | 5~15% |
| L2 | 语义缓存 + 向量就近原型（复用 query embedding） | 0（含 embedding 复用） | 10~25% |
| L3 | LLM 结构化输出（**返回置信度 + 触发主动采样**） | 1 次 | 剩余全部 |

准确率侧的三个动作：
1. **规则层只收敛确定性，不猜测**（不确定一律回落 LLM），引入不了新的误判源；
2. **强制 reason 写「证据片段」**，并要求自评 confidence（另见 `_assess`）；
3. **低置信度主动升级模型**（`intent_escalation_model`），一次重判，命中后回写
   缓存——把「一次便宜但可能错的判断」换成「便宜优先 + 贵模型兜底」。

降级与归一化（保持不变）：
- LLM 调用失败 / 输出非法 → 降级为 ["chat"]，保证主流程可用；
- 置信度低于阈值 → 优先升级重判，仍低则降级为 ["chat"]；
- 多意图归一化：去重、丢弃 chat（仅作兜底）、异常组合按优先级收敛。
"""
import logging
import time
from typing import Literal, Optional

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from core.config import settings
from prompt.prompt_storage import INTENT_CLASSIFICATION_PROMPT
from agent import intent_gate
from agent.intent_gate import GateDecision

logger = logging.getLogger(__name__)

# 意图集合（结构化输出约束）
INTENT_TYPES = ("knowledge_base", "task", "chat")

# 置信度兜底阈值：低于该值视为低置信度，触发升级重判 / 降级
CONFIDENCE_THRESHOLD = 0.5

# 升级重判的触发阈值：低于该值但高于 CONFIDENCE_THRESHOLD 时换更强模型再判一次
ESCALATION_THRESHOLD = 0.75

# 观测用指标前缀（对应 §15.5 指标目录）
_K_INTENT = "agent.intent."
_K_GATE = "agent.intent.gate."
_K_SOURCE = "agent.intent.source."

# 原型向量缓存（进程级，懒加载一次）
_PROTO_VEC_CACHE: Optional[dict] = None
_PROTO_VEC_TRIED = False


class IntentResult(BaseModel):
    """意图识别结构化输出（多标签：同一输入可命中多个意图）"""
    intents: list[Literal["knowledge_base", "task", "chat"]] = Field(
        description="意图列表"
    )
    reason: str = Field(description="分类依据")
    confidence: float = Field(description="整体置信度 0~1")


def _metrics():
    """惰性获取指标单例（观测不可用时静默返回 None）。"""
    try:
        from agent.observability import metrics

        return metrics
    except Exception:
        return None


def _observe_source(source: str) -> None:
    """记录本次判定的来源层（用于统计 LLM 调用降幅）。"""
    m = _metrics()
    if m is not None:
        m.incr(f"{_K_SOURCE}{source}")


def _proto_vectors() -> Optional[dict]:
    """
    加载并缓存原型语料的稠密向量（懒加载；失败返回 None，静默退化）。

    仅在首次需要 Tier-2 时才构建，不增加启动耗时；构建失败不影响 LLM 兜底。
    """
    global _PROTO_VEC_CACHE, _PROTO_VEC_TRIED
    if _PROTO_VEC_CACHE is not None or _PROTO_VEC_TRIED:
        return _PROTO_VEC_CACHE
    _PROTO_VEC_TRIED = True
    try:
        from embeddings.embedding_deal import DashScopeEmbedding

        emb = DashScopeEmbedding()
        texts = [t for t, _ in intent_gate.PROTOTYPES]
        vecs = emb.embed_documents(texts)
        if vecs and len(vecs) == len(texts):
            _PROTO_VEC_CACHE = dict(zip(texts, vecs))
            logger.info(f"[Agent-意图] 原型向量缓存构建完成: {len(vecs)} 条")
    except Exception as e:
        logger.warning(f"[Agent-意图] 原型向量构建失败（跳过向量判定层）: {e}")
    return _PROTO_VEC_CACHE


class IntentRouter:
    """意图识别器：分层快通道（规则/缓存/向量）+ LLM 结构化输出兜底"""

    def __init__(self, model: str = "qwen-turbo"):
        # 阶段 4 加固：意图识别是主链路首站，挂死会阻塞整条请求，
        # 补配单次推理超时 + 1 次自动重试（偶发网络抖动自愈，重试仍失败则降级 chat）。
        self._model = model
        self._llm = ChatOpenAI(
            model=model,
            openai_api_key=settings.api_key,
            openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0,
            timeout=settings.agent_llm_timeout_seconds,
            max_retries=1,
        ).with_structured_output(IntentResult)
        # 升级重判用的更强模型（懒创建；与主模型相同则跳过，避免无效开销）
        self._escalation_llm = None
        self._escalation_model = (settings.intent_escalation_model or "").strip()

    # ------------------------------------------------------------------
    # 对外入口
    # ------------------------------------------------------------------

    async def classify(
        self,
        user_input: str,
        query_embedding: Optional[list] = None,
        last_intents: Optional[list] = None,
    ) -> IntentResult:
        """
        对用户输入进行多标签意图分类（分层判定，LLM 为兜底而非必经）。

        :param user_input: 用户输入
        :param query_embedding: 已预计算的 query 稠密向量（复用语义缓存，可省一次 embedding）
        :param last_intents: 上一轮意图（用于短追问继承）
        :return: IntentResult（intents 已归一化）
        """
        started = time.perf_counter()

        # ---- L0/L1：规则门控 + 进程内决策缓存 ----
        decision = self._fast_path(user_input, last_intents)
        if decision is not None:
            self._log_decision(decision, started)
            return self._to_result(decision)

        # ---- L2：向量就近原型（复用已算出的 embedding，零额外 API） ----
        decision = self._vector_path(user_input, query_embedding)
        if decision is not None:
            intent_gate.put_cached_decision(user_input, decision)
            self._log_decision(decision, started)
            return self._to_result(decision)

        # ---- L3：LLM 结构化输出 ----
        result = await self._llm_classify(user_input)
        _observe_source("llm")
        self._log_llm(result, started)
        # LLM 高置信结果进入进程内缓存（同会话重复输入零成本）
        if result.confidence >= settings.intent_llm_cache_confidence:
            intent_gate.put_cached_decision(
                user_input,
                GateDecision(
                    kind=intent_gate.KIND_FORCED, rule="llm_cached",
                    intents=list(result.intents), confidence=result.confidence,
                    reason=result.reason or "LLM 高置信结果缓存",
                ),
            )
        return self._normalize(result)

    # ------------------------------------------------------------------
    # 分层实现
    # ------------------------------------------------------------------

    def _fast_path(
        self, user_input: str, last_intents: Optional[list]
    ) -> Optional[GateDecision]:
        """L0 规则门控 + L1 进程内缓存（零成本，先看缓存避免重复跑正则）。"""
        cached = intent_gate.get_cached_decision(user_input)
        if cached is not None:
            _observe_source("rule_cache")
            return cached

        decision = intent_gate.gate(user_input, last_intents=last_intents)
        if decision.hit:
            m = _metrics()
            if m is not None:
                m.incr(f"{_K_GATE}{decision.kind}.{decision.rule}")
            _observe_source(f"rule_{decision.kind}")
            intent_gate.put_cached_decision(user_input, decision)
            return decision
        return None

    def _vector_path(
        self, user_input: str, query_embedding: Optional[list]
    ) -> Optional[GateDecision]:
        """
        L2 向量就近判定。

        `query_embedding` 由呼叫方（知识库分支）预计算后回传时可省一次 API；
        未提供时本层直接跳过（不主动为分类单独付一次 embedding 调用）。
        """
        if not query_embedding:
            return None
        vecs = _proto_vectors()
        if not vecs:
            return None
        decision = intent_gate.match_by_embedding(query_embedding, vecs)
        if decision is None:
            return None
        m = _metrics()
        if m is not None:
            m.incr(f"{_K_GATE}forced.{decision.rule}")
        _observe_source("embed")
        return decision

    async def _llm_classify(self, user_input: str) -> IntentResult:
        """
        L3：LLM 结构化输出。

        低置信度（< ESCALATION_THRESHOLD）时用更强模型重判一次（主动采样），
        命中后取更高置信的那份，降低「便宜模型的偶发误判」。
        """
        result = await self._invoke(self._llm, user_input)
        if result is None:
            logger.warning("意图识别返回空结果，降级为普通聊天")
            return IntentResult(
                intents=["chat"], reason="意图识别返回空结果，降级为普通聊天",
                confidence=0,
            )
        if CONFIDENCE_THRESHOLD <= result.confidence < ESCALATION_THRESHOLD:
            better = await self._escalate(user_input)
            if better is not None and better.confidence > result.confidence:
                m = _metrics()
                if m is not None:
                    m.incr(f"{_K_INTENT}escalated")
                logger.info(
                    f"[Agent-意图] 升级重判生效: {result.confidence:.2f} -> "
                    f"{better.confidence:.2f} (model={self._escalation_model})"
                )
                return better
        if result.confidence < CONFIDENCE_THRESHOLD:
            logger.info(
                f"意图识别置信度 {result.confidence:.2f} 低于阈值"
                f"{CONFIDENCE_THRESHOLD}，降级为普通聊天"
            )
            return IntentResult(
                intents=["chat"], reason="置信度低于阈值",
                confidence=result.confidence,
            )
        return result

    async def _escalate(self, user_input: str) -> Optional[IntentResult]:
        """用更强模型重判（未配置或与主模型相同则跳过）。"""
        if not self._escalation_model or self._escalation_model == self._model:
            return None
        try:
            if self._escalation_llm is None:
                self._escalation_llm = ChatOpenAI(
                    model=self._escalation_model,
                    openai_api_key=settings.api_key,
                    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
                    temperature=0,
                    timeout=settings.agent_llm_timeout_seconds,
                    max_retries=1,
                ).with_structured_output(IntentResult)
            return await self._invoke(self._escalation_llm, user_input)
        except Exception as e:
            logger.warning(f"[Agent-意图] 升级重判失败（沿用原判定）: {e}")
            return None

    @staticmethod
    async def _invoke(llm, user_input: str) -> Optional[IntentResult]:
        """执行一次结构化输出推理（异常上抛由上层统一处理）。"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", INTENT_CLASSIFICATION_PROMPT),
            ("human", "{input}"),
        ])
        return await (prompt | llm).ainvoke({"input": user_input})

    # ------------------------------------------------------------------
    # 结果转换 / 归一化
    # ------------------------------------------------------------------

    @staticmethod
    def _to_result(decision: GateDecision) -> IntentResult:
        """把快通道判定转为 IntentResult（并做同样的归一化）。"""
        return IntentRouter._normalize(
            IntentResult(
                intents=list(decision.intents) or ["chat"],
                reason=decision.reason or decision.rule,
                confidence=decision.confidence,
            )
        )

    @staticmethod
    def _log_decision(decision: GateDecision, started: float) -> None:
        """快通道命中日志（含耗时，便于统计省下的 LLM 往返）。"""
        cost_ms = (time.perf_counter() - started) * 1000
        logger.info(
            f"[Agent-意图] 快通道命中 kind={decision.kind} rule={decision.rule} "
            f"intents={decision.intents} confidence={decision.confidence:.2f} "
            f"cost={cost_ms:.1f}ms reason={decision.reason[:60]}"
        )
        m = _metrics()
        if m is not None:
            m.observe(f"{_K_INTENT}latency_ms", cost_ms)

    @staticmethod
    def _log_llm(result: IntentResult, started: float) -> None:
        """LLM 分类日志。"""
        cost_ms = (time.perf_counter() - started) * 1000
        logger.info(
            f"[Agent-意图] LLM 分类 intents={result.intents} "
            f"confidence={result.confidence:.2f} cost={cost_ms:.1f}ms"
        )
        m = _metrics()
        if m is not None:
            m.observe(f"{_K_INTENT}latency_ms", cost_ms)

    @staticmethod
    def _normalize(result: IntentResult) -> IntentResult:
        """
        意图归一化：去重、丢弃 chat、异常多意图按优先级收敛。

        - ["task","knowledge_base"] → 保留两个，触发并行分支
        - 含 chat 的其它组合 → 丢弃 chat
        - 空列表 / 三个全命中 → 按 task > knowledge_base > chat 收敛
        """
        intents = list(dict.fromkeys(result.intents))          # 去重保序
        if "chat" in intents and len(intents) > 1:             # chat 不参与并行
            intents.remove("chat")
        if not intents or len(intents) > 2:
            # 异常组合收敛：task 优先于 knowledge_base，chat 仅兜底
            intents = [i for i in ("task", "knowledge_base") if i in intents] or ["chat"]
        return IntentResult(
            intents=intents, reason=result.reason, confidence=result.confidence
        )
