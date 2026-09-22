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
  （§19.1：该承诺此前只对「返回空结果」生效，异常仍会击穿主链路；现改为
  捕获 LengthFinishReasonError 等全部异常，先试升级模型兜底，再降级 chat）
- 置信度低于阈值 → 优先升级重判，仍低则降级为 ["chat"]；
- 多意图归一化：去重、丢弃 chat（仅作兜底）、异常组合按优先级收敛。
"""
import asyncio
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
# §17.3：原实现用 bool 标志「已尝试」，失败一次就**永久放弃**该层（且修好代码也必须
# 重启进程才生效）。现改为：仅**成功**时置位；失败累计到 _PROTO_VEC_MAX_FAILS 次后
# 才熔断，避免每次请求都去撞一个确定性失败的构建。
_PROTO_VEC_READY = False
_PROTO_VEC_FAILS = 0
_PROTO_VEC_MAX_FAILS = 3
# 构建并发去重锁（避免启动预热与首个请求同时构建）
_PROTO_VEC_LOCK: Optional[asyncio.Lock] = None


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

    §17.3 修复：
    1. 原代码写 `DashScopeEmbedding()`，但该类 `api_key` 是**必填位置参数**
       （见 embeddings/embedding_deal.py 的 __init__ 签名），导致每次构建都抛
       TypeError、且只在日志里留一条 WARNING —— 向量判定层**从未生效过**。
    2. 原 `_PROTO_VEC_TRIED` 失败即永久置位，改为「仅成功才置位 + 有限次失败熔断」。
    """
    global _PROTO_VEC_CACHE, _PROTO_VEC_READY, _PROTO_VEC_FAILS
    if _PROTO_VEC_READY or _PROTO_VEC_FAILS >= _PROTO_VEC_MAX_FAILS:
        return _PROTO_VEC_CACHE
    try:
        from embeddings.embedding_deal import DashScopeEmbedding

        emb = DashScopeEmbedding(api_key=settings.api_key)
        texts = [t for t, _ in intent_gate.PROTOTYPES]
        vecs = emb.embed_documents(texts)
        if vecs and len(vecs) == len(texts):
            _PROTO_VEC_CACHE = dict(zip(texts, vecs))
            _PROTO_VEC_READY = True
            logger.info(f"[Agent-意图] 原型向量缓存构建完成: {len(vecs)} 条")
        else:
            _PROTO_VEC_FAILS += 1
            logger.warning(
                f"[Agent-意图] 原型向量构建结果不完整（跳过向量判定层）: "
                f"got={len(vecs) if vecs else 0} want={len(texts)} "
                f"fails={_PROTO_VEC_FAILS}/{_PROTO_VEC_MAX_FAILS}"
            )
    except Exception as e:
        _PROTO_VEC_FAILS += 1
        logger.warning(
            f"[Agent-意图] 原型向量构建失败（跳过向量判定层）: {e} "
            f"fails={_PROTO_VEC_FAILS}/{_PROTO_VEC_MAX_FAILS}"
        )
    return _PROTO_VEC_CACHE


async def warmup_proto_vectors() -> bool:
    """
    启动预热：把 20 条原型语料的 embedding 挪到进程启动期（§17.3）。

    目的有二：① 不占首字延迟（首个请求不必等一次 embedding API 往返）；
    ② 早期暴露 api_key / 网络配置问题，而不是等到线上首个请求才静默退化。

    失败不影响启动（返回 False，向量层降级、LLM 兜底照常工作）。
    """
    global _PROTO_VEC_LOCK
    if _PROTO_VEC_READY or _PROTO_VEC_FAILS >= _PROTO_VEC_MAX_FAILS:
        return _PROTO_VEC_READY
    if _PROTO_VEC_LOCK is None:
        _PROTO_VEC_LOCK = asyncio.Lock()
    async with _PROTO_VEC_LOCK:
        # 双重检查：等锁期间可能已被其它协程构建完成
        if _PROTO_VEC_READY:
            return True
        # embed_documents 是同步阻塞调用，放到线程池避免卡住事件循环
        try:
            vecs = await asyncio.to_thread(_proto_vectors)
        except Exception as e:
            logger.warning(f"[Agent-意图] 原型向量预热异常（已降级）: {e}")
            return False
        return bool(vecs)


def _build_intent_llm(model: str):
    """
    构造意图判别用的结构化输出 LLM（§19.1）。

    两处关键参数，都是被线上事故倒逼出来的：

    1. `max_tokens`：原先写死 256，看似「结构化输出只需 ~100 token」很宽裕，
       但 qwen 思考型模型会**先消耗 reasoning_tokens 再产出正文**——256 会被思考
       过程吃满，`finish_reason=length` 导致正文一个 token 都没输出，
       openai SDK 直接抛 `LengthFinishReasonError`，整条请求 500。
       现改为可配（默认 1024），为思考过程留足余量。
    2. `enable_thinking=False`：分类任务本就不需要长链推理，关掉思考既根除
       上面的 token 争用，也顺带把首字延迟砍掉大半（实测 8.3s → 亚秒级）。
       通过 extra_body 下发，不支持该参数的模型会忽略它。

    :param model: 模型名
    :return: 绑定了 IntentResult 结构化输出的 Runnable
    """
    kwargs = {
        "model": model,
        "openai_api_key": settings.api_key,
        "openai_api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "temperature": 0,
        "timeout": settings.intent_llm_timeout_seconds,
        "max_retries": 1,
        "max_tokens": settings.intent_llm_max_tokens,
    }
    if settings.intent_llm_disable_thinking:
        kwargs["extra_body"] = {"enable_thinking": False}
    return ChatOpenAI(**kwargs).with_structured_output(IntentResult)


def _is_length_error(exc: BaseException) -> bool:
    """
    判断异常是否为「输出被 max_tokens 截断」。

    不直接 import `LengthFinishReasonError`：不同 openai SDK 版本的导出位置
    不稳定，硬依赖会引入新的崩溃点。按类名匹配 + 文案兜底，足够稳。
    """
    if type(exc).__name__ == "LengthFinishReasonError":
        return True
    return "length limit was reached" in str(exc)


class IntentRouter:
    """意图识别器：分层快通道（规则/缓存/向量）+ LLM 结构化输出兜底"""

    def __init__(self, model: str = "qwen-turbo"):
        # 阶段 4 加固：意图识别是主链路首站，挂死会阻塞整条请求，
        # 补配单次推理超时 + 1 次自动重试（偶发网络抖动自愈，重试仍失败则降级 chat）。
        #
        # §17.3 调整：超时与 agent_llm_timeout_seconds（60s）解耦，改用
        # intent_llm_timeout_seconds（默认 15s）—— 意图判别只需输出一个极短的
        # 结构化对象，等 60s 毫无意义；超时即降级 chat，比长时间阻塞首字更划算。
        #
        # §19.1：构造逻辑抽到 _build_intent_llm（max_tokens / 思考开关见其文档）。
        self._model = model
        self._llm = _build_intent_llm(model)
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
        try:
            result = await self._invoke(self._llm, user_input)
        except Exception as e:
            # §19.1：意图判别是可选增强，绝不能因为它把整条主链路拖成 500。
            # 原实现让 LengthFinishReasonError 一路上抛到 api 层，一次分类失败
            # 就等于请求彻底失败 —— 与「失败降级 chat」的设计承诺相悖。
            return await self._degrade_on_error(e, user_input)
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

    async def _degrade_on_error(self, exc: BaseException, user_input: str) -> IntentResult:
        """
        LLM 分类失败时的兜底（§19.1）。

        策略：先尝试用升级模型重判一次（换模型常常能绕过同一处的 token 争用），
        仍失败则降级为 chat —— 意图判不准只是走错分支，请求失败是整条链路不可用，
        两者代价差一个数量级。
        """
        m = _metrics()
        if _is_length_error(exc):
            if m is not None:
                m.incr(f"{_K_INTENT}length_truncated")
            logger.warning(
                f"[Agent-意图] LLM 输出被 max_tokens 截断（model={self._model} "
                f"max_tokens={settings.intent_llm_max_tokens}）：{exc}"
            )
        else:
            if m is not None:
                m.incr(f"{_K_INTENT}llm_error")
            logger.warning(f"[Agent-意图] LLM 分类失败（降级处理）: {exc}")

        better = await self._escalate(user_input)
        if better is not None and better.confidence >= CONFIDENCE_THRESHOLD:
            logger.info(
                f"[Agent-意图] 主模型失败后由升级模型兜底成功: "
                f"intents={better.intents} confidence={better.confidence:.2f}"
            )
            return better
        return IntentResult(
            intents=["chat"], reason=f"意图识别失败降级: {type(exc).__name__}",
            confidence=0,
        )

    async def _escalate(self, user_input: str) -> Optional[IntentResult]:
        """用更强模型重判（未配置或与主模型相同则跳过）。"""
        if not self._escalation_model or self._escalation_model == self._model:
            return None
        try:
            if self._escalation_llm is None:
                self._escalation_llm = _build_intent_llm(self._escalation_model)
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


# =============================================================================
# 进程级单例（设计文档 §17.3）
# =============================================================================
# 原实现每请求都 `IntentRouter(model=model)`，等于每次都重建 ChatOpenAI
# （内部各自持有 httpx client / 连接池），连接复用失效、每次都付 TLS 握手，
# 直接抬高首字延迟。意图判别是无状态的，按 model 缓存单例即可。
#
# 按 model 分桶：state["model"] 可由前端指定，不同 model 必须用不同实例。
_ROUTER_CACHE: dict[str, "IntentRouter"] = {}


def get_router(model: str = "qwen-turbo") -> "IntentRouter":
    """
    取进程级 IntentRouter 单例（按 model 分桶）。

    线程/协程安全性：dict 读写与单次构造在 CPython 下无竞态风险；极端并发下
    即便构造出两个实例，也只是多一个客户端，不影响正确性。
    """
    key = model or "qwen-turbo"
    router = _ROUTER_CACHE.get(key)
    if router is None:
        router = IntentRouter(model=key)
        _ROUTER_CACHE[key] = router
    return router


def reset_router_cache() -> None:
    """清空单例缓存（测试用：避免跨用例共享 LLM 客户端）。"""
    _ROUTER_CACHE.clear()
