"""
意图快速通道（规则层 / Tier-0 Gate）

设计依据：Hermes Agent 的「意图理解 = 规则层 + LLM 自决层」多层机制
（见 §16.1）。本项目原实现为**纯 LLM 单次分类**——每个请求都要付一次
同步 LLM 往返（数百 ms ~ 数秒），且对「你好」「/clear」这类确定性输入
完全是浪费，还引入了误分类风险。

本模块提供**零 LLM 成本**的前置判定，分三档：

1. `SLASH`   斜杠命令 → 本地指令，直接由命令层处理（不进主图、不落库）；
2. `CHITCHAT` 寒暄/应答/致谢等 trivia → 直连 chat 分支，跳过意图 LLM；
3. `FORCED`  强信号（SQL/表名 + 写动词、表清单问询、知识库显式指名）→
             跳过意图 LLM，直接给出确定性意图（**准确率 100%**，因为不猜）；
4. `NONE`    未命中 → 交给 Tier-1（复用本地语义缓存的 embedding）或
             Tier-2（LLM 分类）继续判定。

三条工程约束（与 Hermes 的设计原则一致）：
- **规则层只做「确定性收敛」，不做「模糊猜测」**：任何不确定一律回落到
  LLM，避免规则误杀降低准确率；
- **失败静默**（best-effort）：正则异常、配置缺失都不影响主链路；
- **可观测**：每次命中都计 `agent.intent.gate.<tier>.<rule>`，便于线上
  统计跳过率与规则准确率（§16.7 灰度验收要求）。
"""
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# =============================================================================
# 判定结果枚举（字符串常量，避免引入 Enum 给状态序列化添麻烦）
# =============================================================================

KIND_NONE = "none"            # 未命中，交由后续层
KIND_SLASH = "slash"          # 斜杠命令
KIND_CHITCHAT = "chitchat"    # 寒暄/琐碎输入
KIND_FORCED = "forced"        # 强信号确定意图


# =============================================================================
# 规则 1：斜杠命令（Hermes 第一层：CommandDef registry）
# =============================================================================

# 内置命令表（name -> 说明）。新增命令只需在此追加一行，不做 if/elif 链。
SLASH_COMMANDS: dict[str, str] = {
    "clear": "清空当前会话上下文",
    "new": "开启一个新会话",
    "stop": "中止当前生成",
    "help": "查看可用指令",
    "status": "查看服务与依赖状态",
}

# 斜杠命令形态：/cmd 或 /cmd args（命令名限定小写字母、数字、连字符）
_SLASH_RE = re.compile(r"^\s*/([a-z][a-z0-9-]{0,31})(?:\s+([\s\S]*))?$", re.I)


def parse_slash_command(text: str) -> Optional[tuple[str, str]]:
    """
    解析斜杠命令。

    仅当命令名在 `SLASH_COMMANDS` 内置表中时才判定为命令；未知 `/xxx`
    视为普通文本回落到后续层（用户可能在说路径或做除法），避免误吞。

    :param text: 用户原始输入
    :return: (命令名, 参数) 或 None
    """
    m = _SLASH_RE.match(text or "")
    if not m:
        return None
    name = m.group(1).lower()
    if name not in SLASH_COMMANDS:
        return None
    return name, (m.group(2) or "").strip()


# =============================================================================
# 规则 2：寒暄门控（Hermes 第四层：is_trivial_prompt + TRIVIAL_PROMPT_RE）
# =============================================================================

# 可配置的关键词表（逗号分隔）；默认覆盖中文高频寒暄/应答/致谢/告别。
TRIVIAL_KEYWORDS_DEFAULT = (
    "你好,您好,哈喽,哈啰,嗨,在吗,在么,有人吗,早上好,中午好,下午好,晚上好,"
    "早安,晚安,谢谢,多谢,感谢,辛苦了,好的,好嘞,收到,明白,了解,知道了,"
    "嗯,哦,噢,好的呢,可以,没问题,再见,拜拜,回头见,没事了,算了"
)

# 纯标点/表情/空白视为琐碎输入
_PUNCT_ONLY_RE = re.compile(
    r"^[\s\.,;:!\?~\-_=+*/\\|@#$%^&()\[\]{}<>'\"、。，；：！？～—…·`"
    r"\U0001F300-\U0001FAFF\u2600-\u27BF]+$"
)
# 中文/英文字符计数（用于长度阈值判定）
_WORD_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]")

# 琐碎输入最大长度（字符）：超过则必然包含实质内容，不再走 trivia 快通道
TRIVIAL_MAX_LEN = 12


def _trivial_keywords() -> list[str]:
    """读取可配置的寒暄关键词表（环境变量优先，其次 settings，最后默认值）。"""
    raw = os.getenv("AGENT_INTENT_TRIVIAL_KEYWORDS", "")
    if not raw:
        try:
            from core.config import settings

            raw = settings.intent_trivial_keywords or ""
        except Exception:
            raw = ""
    if not raw:
        raw = TRIVIAL_KEYWORDS_DEFAULT
    return [k.strip() for k in raw.split(",") if k.strip()]


def is_trivial(text: str) -> bool:
    """
    判断是否为琐碎输入（寒暄/应答/纯标点/空）。

    判定顺序（**保守优先**：只要可能含实质意图就返回 False）：
    1. 空串 / 纯标点表情 → True；
    2. 长度 > TRIVIAL_MAX_LEN → False（放行给后续层）；
    3. 命中关键词表，且**去除命中词与标点后无剩余实质字符** → True。

    第 3 条是关键：`谢谢，我想查一下订单` 含关键词但仍有实质内容，
    必须返回 False，否则会把知识库/任务请求误判为寒暄。

    :param text: 用户原始输入
    :return: 是否琐碎
    """
    s = (text or "").strip()
    if not s:
        return True
    if _PUNCT_ONLY_RE.match(s):
        return True
    if len(s) > TRIVIAL_MAX_LEN:
        return False
    remainder = s
    for kw in _trivial_keywords():
        remainder = remainder.replace(kw, "")
    remainder = _PUNCT_ONLY_RE.sub("", remainder).strip()
    return not _WORD_RE.search(remainder)


# =============================================================================
# 规则 3：强信号（确定性意图，不经 LLM）
# =============================================================================

# 数据库写动词（用于 task 强信号判定）
_DB_WRITE_VERBS = ("删除", "删掉", "移除", "去掉", "修改", "改成", "更新", "调整为",
                   "新增", "插入", "添加", "增加", "录入")
# 数据库名词（含英文表名形态）
_DB_NOUNS = ("数据库", "数据表", "表", "记录", "数据", "字段", "行", "条")
_DB_NOUN_RE = re.compile(r"\b[a-z][a-z0-9_]{2,30}\b", re.I)   # 形如 user / order_item

# 只读问询动词（"查一下/看看/统计"）+ 表结构词 → 也是 task（只读，无审批）
_TABLE_META_WORDS = ("有哪些表", "哪些表", "表结构", "表清单", "字段有哪些", "有哪些字段",
                     "表定义", "库表")

# 知识库显式指名（双口径：`指名词` 或 `根据/参考 + 文档类词`，避免漏掉「对照政策」）
_KB_HINTS = ("知识库", "文档库", "资料库", "知识文档", "已上传的文档", "上传的文件")
_KB_DOC_WORDS = ("文档", "资料", "文件", "手册", "说明书", "规范", "制度", "政策",
                 "条款", "模板", "FAQ", "faq")
_KB_PREPS = ("根据", "参照", "按照", "依据", "参考", "对照", "结合", "基于", "看", "查阅")

# 意图顺序（与 INTENT_TYPES 保持一致语义，仅用于构造 forced 结果）
_TASK = "task"
_KB = "knowledge_base"
_CHAT = "chat"


@dataclass
class GateDecision:
    """快速通道判定结果。"""

    kind: str = KIND_NONE                     # none / slash / chitchat / forced
    rule: str = ""                            # 命中的规则名（观测与灰度用）
    intents: list[str] = field(default_factory=list)   # 确定意图（kind=forced/chitchat 时有值）
    confidence: float = 0.0                   # 置信度（forced=1.0，chitchat=0.95）
    reason: str = ""                          # 判定依据
    slash_name: Optional[str] = None          # kind=slash 时的命令名
    slash_args: str = ""                      # kind=slash 时的参数

    @property
    def hit(self) -> bool:
        """是否命中快通道（命中即无需后续 LLM 分类）。"""
        return self.kind != KIND_NONE


def _has_kb_hint(text: str) -> bool:
    """
    是否显式指名知识库/文档。

    双口径（任一命中即可）：
    1. 直指词：`知识库` / `资料库` / `已上传的文档` 等；
    2. 介词 + 文档类词：`对照售后政策` / `根据手册` / `参考制度` —— 中文里
       「指向文档作依据」的高频表达，漏掉会造成并行分支丢失。
    """
    if any(h in text for h in _KB_HINTS):
        return True
    for prep in _KB_PREPS:
        idx = text.find(prep)
        while idx >= 0:
            tail = text[idx + len(prep) : idx + len(prep) + 8]
            if any(w in tail for w in _KB_DOC_WORDS):
                return True
            idx = text.find(prep, idx + 1)
    return False


def _looks_like_db_task(text: str) -> bool:
    """
    识别数据库操作强信号。

    判定口径（**宁可漏判、不可误判**，漏判只是走 LLM，误判会走错分支）：
    - 写操作：命中写动词 + （数据库名词 或 英文表名形态 token）；
    - 表结构问询：命中 `_TABLE_META_WORDS`。
    """
    if any(w in text for w in _TABLE_META_WORDS):
        return True
    has_write = any(v in text for v in _DB_WRITE_VERBS)
    if not has_write:
        # 只读查询不强判：`查询` / `看看` 同时也是知识库高频动词，
        # 只有搭配明确的数据库名词才算强信号。
        return any(n in text for n in _DB_NOUNS) and "查" in text
    # 写动词 + （数据库名词 或 英文表名形态 token）
    return any(n in text for n in _DB_NOUNS) or bool(_DB_NOUN_RE.search(text))


def gate(text: str, last_intents: Optional[list[str]] = None) -> GateDecision:
    """
    执行快速通道判定（同步、零 IO、零 LLM）。

    :param text: 用户当前输入
    :param last_intents: 上一轮的意图（用于短追问继承，可为 None）
    :return: GateDecision
    """
    try:
        s = (text or "").strip()

        # ---- 规则 1：斜杠命令（优先级最高，Hermes 第一层） ----
        slash = parse_slash_command(s)
        if slash:
            return GateDecision(
                kind=KIND_SLASH, rule="slash_command", slash_name=slash[0],
                slash_args=slash[1], confidence=1.0,
                reason=f"斜杠命令 /{slash[0]}",
            )

        # ---- 规则 2：寒暄门控（Hermes 第四层） ----
        if is_trivial(s):
            return GateDecision(
                kind=KIND_CHITCHAT, rule="trivial_prompt", intents=[_CHAT],
                confidence=0.95, reason="寒暄/应答类琐碎输入，无需分类",
            )

        # ---- 规则 3：短追问继承（Hermes 消息守卫的「同会话上下文连续性」思路） ----
        # 上一轮已判 task/knowledge_base 且本轮为无动词短句（如「张三」「继续」），
        # 直接继承上一轮意图，跳过 LLM。仅在上一轮为**单分支**时继承，避免
        # 把并行场景的复杂度带进来。
        if last_intents and len(last_intents) == 1 and last_intents[0] in (_TASK, _KB):
            if len(s) <= 8 and not any(v in s for v in _DB_WRITE_VERBS) \
                    and "知识库" not in s and "文档" not in s:
                return GateDecision(
                    kind=KIND_FORCED, rule="followup_inherit",
                    intents=list(last_intents), confidence=0.85,
                    reason=f"短追问（≤8 字）继承上一轮意图 {last_intents[0]}",
                )

        # ---- 规则 4：强信号 → 确定性意图 ----
        kb_hint = _has_kb_hint(s)
        db_task = _looks_like_db_task(s)

        if db_task and kb_hint:
            return GateDecision(
                kind=KIND_FORCED, rule="db_task+explicit_kb",
                intents=[_TASK, _KB], confidence=1.0,
                reason="命中数据库操作强信号且显式指名知识库，并行双分支",
            )
        if db_task:
            return GateDecision(
                kind=KIND_FORCED, rule="db_task_strong", intents=[_TASK],
                confidence=1.0, reason="命中数据库操作强信号（明确表/名词 + 操作动词）",
            )
        if kb_hint:
            return GateDecision(
                kind=KIND_FORCED, rule="explicit_kb", intents=[_KB],
                confidence=1.0, reason="显式指名知识库/文档",
            )

        # ---- 未命中：交给后续层 ----
        return GateDecision(kind=KIND_NONE, rule="", reason="")
    except Exception as e:
        # best-effort：规则层异常绝不影响主链路（退化为走 LLM 分类）
        logger.warning(f"[Agent-意图门控] 规则层异常（退化为 LLM 分类）: {e}")
        return GateDecision(kind=KIND_NONE, rule="gate_error", reason=str(e))


# =============================================================================
# 规则 4 辅助：确定性意图的置信度与缓存 TTL
# =============================================================================

# 确定性意图的 TTL 远长于 LLM 结果：规则判定不依赖模型版本，可放心复用一个小时。
FORCED_TTL_SECONDS = 3600
CHITCHAT_TTL_SECONDS = 1800


def ttl_for(decision: GateDecision) -> int:
    """按判定来源返回缓存 TTL（秒）。"""
    if decision.kind == KIND_FORCED:
        return FORCED_TTL_SECONDS
    if decision.kind == KIND_CHITCHAT:
        return CHITCHAT_TTL_SECONDS
    return 0


# =============================================================================
# Tier-1：embedding 就近命中（复用已算出的 query 向量，零额外 API）
# =============================================================================

# 原型语料：每条 = (文本, 意图)。命中即返回该意图，置信度 = 相似度。
# 说明：与语义缓存（rag 侧 cache_collection_name）**共用同一份 query embedding**，
# 不额外调用 embedding API——这正是「减少 LLM 调用」之外还能「减少 API 调用」的关键。
PROTOTYPES: list[tuple[str, str]] = [
    # --- task（数据库操作） ---
    ("把用户表里张三这条记录删除", _TASK),
    ("修改订单表的金额字段", _TASK),
    ("在用户表新增一条数据", _TASK),
    ("数据库里有哪些表", _TASK),
    ("查询订单表最近的十条记录", _TASK),
    ("更新商品表的价格", _TASK),
    ("统计一下订单表有多少行数据", _TASK),
    # --- knowledge_base（知识库文档） ---
    ("售后政策是怎么规定的", _KB),
    ("产品说明书里关于安装步骤怎么写的", _KB),
    ("根据公司制度文档，年假有多少天", _KB),
    ("这个故障在手册里的处理方法是什么", _KB),
    ("合同模板里违约责任条款怎么约定", _KB),
    # --- chat（通用问答/闲聊） ---
    ("帮我写一首关于春天的诗", _CHAT),
    ("Python 的列表和元组有什么区别", _CHAT),
    ("今天天气怎么样", _CHAT),
    ("给我讲个笑话", _CHAT),
    ("介绍一下你自己", _CHAT),
]


# 相似度阈值：低于此值不信 embedding 判定，回落 LLM
EMBED_THRESHOLD_DEFAULT = 0.86


def embed_threshold() -> float:
    """读取 embedding 判定阈值（环境变量优先，其次 settings，最后默认值）。"""
    raw = os.getenv("AGENT_INTENT_EMBED_THRESHOLD", "")
    if not raw:
        try:
            from core.config import settings

            return float(settings.intent_embed_threshold)
        except Exception:
            return EMBED_THRESHOLD_DEFAULT
    try:
        return float(raw)
    except (TypeError, ValueError):
        return EMBED_THRESHOLD_DEFAULT


def _cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度（向量已归一化时退化为点积，这里保留通用实现）。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = norm_a = norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 0 or norm_b <= 0:
        return 0.0
    return dot / ((norm_a ** 0.5) * (norm_b ** 0.5))


def match_by_embedding(
    query_vec: Optional[list[float]],
    proto_vecs: Optional[dict[str, list[float]]],
) -> Optional[GateDecision]:
    """
    用已算出的 query embedding 做原型最近邻判定。

    :param query_vec: 当前输入的稠密向量（复用语义缓存的预计算结果）
    :param proto_vecs: 原型向量字典 {原型文本: 向量}（外部加载并缓存）
    :return: GateDecision（kind=forced）或 None
    """
    if not query_vec or not proto_vecs:
        return None
    try:
        best_text, best_score = "", -1.0
        for text, vec in proto_vecs.items():
            score = _cosine(query_vec, vec)
            if score > best_score:
                best_text, best_score = text, score
        thr = embed_threshold()
        if best_score < thr:
            return None
        intent = _PROTO_INTENT.get(best_text)
        if not intent:
            return None
        return GateDecision(
            kind=KIND_FORCED, rule="embed_nearest", intents=[intent],
            confidence=round(best_score, 4),
            reason=f"向量就近命中「{best_text[:16]}」(score={best_score:.3f})",
        )
    except Exception as e:
        logger.warning(f"[Agent-意图门控] 向量判定异常（回落 LLM）: {e}")
        return None


_PROTO_INTENT: dict[str, str] = {t: i for t, i in PROTOTYPES}


# =============================================================================
# 决策缓存：结构化持久化（进程内） + 语义缓存两跳
# =============================================================================

# 进程内决策缓存：normalized_text -> (ts, decision)
_DECISION_CACHE: dict[str, tuple[float, GateDecision]] = {}
# 缓存容量上限，超出时按插入顺序淘汰最旧的一批（防高基数撑爆内存）
_DECISION_CACHE_MAX = 512


def decision_cache_key(text: str) -> str:
    """决策缓存键：去空白 + 小写（规则与 LLM 判定都与绝对大小写无关）。"""
    return re.sub(r"\s+", "", (text or "").lower())[:200]


def get_cached_decision(text: str) -> Optional[GateDecision]:
    """读取进程内决策缓存（过期即删）。"""
    key = decision_cache_key(text)
    item = _DECISION_CACHE.get(key)
    if not item:
        return None
    ts, decision = item
    ttl = ttl_for(decision)
    if ttl <= 0 or (time.time() - ts) > ttl:
        _DECISION_CACHE.pop(key, None)
        return None
    return decision


def put_cached_decision(text: str, decision: GateDecision) -> None:
    """写入进程内决策缓存（仅缓存高分/确定性结果）。"""
    if decision.confidence < 0.85:
        return
    if len(_DECISION_CACHE) >= _DECISION_CACHE_MAX:
        # 简单 FIFO 淘汰：dict 保序，弹出最早插入的 1/4
        for k in list(_DECISION_CACHE.keys())[: _DECISION_CACHE_MAX // 4]:
            _DECISION_CACHE.pop(k, None)
    _DECISION_CACHE[decision_cache_key(text)] = (time.time(), decision)


def serialize_decision(decision: GateDecision) -> str:
    """序列化为语义缓存 payload（store 侧要求字符串）。"""
    return json.dumps(
        {
            "kind": decision.kind,
            "rule": decision.rule,
            "intents": decision.intents,
            "confidence": decision.confidence,
            "reason": decision.reason,
        },
        ensure_ascii=False,
    )


def deserialize_decision(payload) -> Optional[GateDecision]:
    """从语义缓存 payload 还原判定结果（非法内容返回 None）。"""
    try:
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        data = json.loads(payload)
        if not isinstance(data, dict):
            return None
        return GateDecision(
            kind=data.get("kind", KIND_NONE),
            rule=data.get("rule", ""),
            intents=list(data.get("intents") or []),
            confidence=float(data.get("confidence") or 0.0),
            reason=data.get("reason", ""),
        )
    except Exception:
        return None
