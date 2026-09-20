"""
Agent 工具统一定义（2026-09-17 收敛）

将 agent 可调用的工具收敛到本文件，按类别以类组织：

- ``DbToolExecutor``（数据库类工具）：list_tables / query_data / insert_data / update_data / delete_data
- ``KnowledgeToolExecutor``（知识库类工具）：search_knowledge
- ``ClarifyTool``（追问类工具，@tool）：ask_user（批量追问 / 取消路径）
- ``ToolSpec`` / ``REGISTRY``（工具元数据注册表）

安全基线（设计文档 §5.5、§11）：
- 数据库工具白名单 + 参数化 + LIMIT + 超时保护，写操作二次确认（user_intent_quote）；
- 知识库检索只读、权限由服务端注入 username、k clamp 与上下文长度截断；
- 追问工具依赖 langgraph interrupt（需 checkpointer），仅审批模式绑定。

原文件 db_tools.py / knowledge_tool.py / registry.py / tools/clarify_tool.py
保留为薄封装（re-export），保证既有引用零改动。
"""
import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from langchain_core.tools import ToolException, tool
from langgraph.types import interrupt

from core.config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# 数据库类工具（原 agent/tools/db_tools.py）
# =============================================================================

# 合法列名（防止注入）：字母/下划线开头，仅含字母数字下划线
_COLUMN_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

# 敏感列：任何情况下禁止查询/写入
SENSITIVE_COLUMNS = {
    "password",
    "password_hash",
    "secret",
    "secret_key",
    "access_token",
    "refresh_token",
    "token",
    "api_key",
    "apikey",
    "private_key",
    "salt",
    "captcha_text",
}

# ---------------------------------------------------------
# 写操作二次确认（设计文档 §5.5.2）
# 写工具必填 user_intent_quote：模型必须从用户原话中引用
# 表达了写意图的片段，命中任一意图关键词才允许进入审批流程，
# 防止模型臆造写操作、误改客户数据。
# ---------------------------------------------------------

# 写意图关键词（命中任一即视为具备用户授权迹象，审查宽容）
WRITE_INTENT_KEYWORDS = (
    "更新", "修改", "改掉", "改动", "改", "删除", "删掉", "删去", "删",
    "新增", "插入", "添加", "增加", "写入", "变更", "替换", "调整",
    "设置", "设为", "改成", "改为", "改为,", "对齐", "纠正", "纠正为",
    "清空", "清除", "移除", "撤销", "重置", "恢复为", "更新为",
)

# 用户原话引用的最大长度（超长截断，防注入异常参数）
_MAX_INTENT_QUOTE_LENGTH = 200


class DbToolError(ToolException):
    """
    数据库工具执行错误（安全拦截 / 参数非法）。

    继承 langchain_core 的 ToolException：工具框架层默认只把 ToolException
    视为"业务可恢复错误"，配合 create_agent 工具绑定的 handle_tool_error
    可将异常转为 ToolMessage 送回 Agent 循环（模型可自我纠正），
    而非直接中断整个 Agent 循环。

    参数/业务可恢复错误**不计入熔断**（设计文档 §7.7.2），重试即可。
    """


class DbToolSystemError(DbToolError):
    """
    数据库工具系统级故障（超时 / 底层执行失败 / 连接异常）。

    与 DbToolError 的区别（熔断触发口径）：
    - DbToolError：参数/业务错误，可重试修正，**不触发熔断**；
    - DbToolSystemError：工具运行环境故障，**触发熔断**（连续失败/失败率超阈值），
      恢复由定时探测任务用失败时沉淀的入参重放验证（设计文档 §7.7）。

    继承 DbToolError 以保持 ToolException 语义（仍可转为 ToolMessage 送回 Agent 循环）。
    """


class DbToolExecutor:
    """数据库工具执行器：白名单 + 参数化 + LIMIT + 超时保护"""

    def __init__(self, session):
        """
        :param session: SQLAlchemy AsyncSession（来自图运行配置 configurable.db）
        """
        self._session = session
        self._allowed_tables = set(settings.agent_db_allowed_table_list)
        self._max_rows = settings.agent_query_max_rows
        self._timeout = settings.agent_task_timeout_seconds
        # 审计记录：每次工具调用的脱敏摘要（供 T2-5 落库）
        self.tool_calls: List[dict] = []

    # =========================================================================
    # 安全校验
    # =========================================================================

    def _assert_table_allowed(self, table: str) -> None:
        """表白名单校验：非法或越权表直接拒绝。"""
        if not table or table not in self._allowed_tables:
            raise DbToolError(f"表 {table!r} 不在授权白名单内，拒绝访问")

    @staticmethod
    def _assert_column_name(column: str) -> None:
        """列名校验：非法格式或敏感列拒绝。"""
        if not _COLUMN_NAME_RE.match(column):
            raise DbToolError(f"列名 {column!r} 非法（仅允许字母/数字/下划线）")
        if column.lower() in SENSITIVE_COLUMNS:
            raise DbToolError(f"列 {column!r} 为敏感字段，禁止访问")

    @classmethod
    def _assert_columns(cls, columns: Optional[List[str]]) -> None:
        """批量列名校验（None 表示全列，仅用于查询）。"""
        if columns:
            for col in columns:
                cls._assert_column_name(col)

    @staticmethod
    def _validate_filters(filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """过滤条件校验：必须是字典，键为合法列名，值不做额外限制（参数化）。"""
        filters = filters or {}
        if not isinstance(filters, dict):
            raise DbToolError("过滤条件必须为对象")
        for key in filters:
            DbToolExecutor._assert_column_name(str(key))
        return filters

    @staticmethod
    def _validate_values(values: Dict[str, Any]) -> Dict[str, Any]:
        """写入值校验：非空字典，键为合法列名。"""
        if not values or not isinstance(values, dict):
            raise DbToolError("写入值必须为非空对象")
        for key in values:
            DbToolExecutor._assert_column_name(str(key))
        return values

    @classmethod
    def _assert_intent_quote(cls, quote: str) -> str:
        """
        写操作二次确认：校验模型从用户原话中引用的写意图片段。

        缺失/为空直接拒绝；超长截断；未命中写意图关键词视为
        缺少用户明确授权迹象，拒绝并引导模型反问用户（设计文档 §5.5.2）。

        :param quote: 用户原话中表达该写操作意图的片段
        :return: 规范化后的 quote（去空白 + 截断）
        :raises DbToolError: 缺少引用或不具备写意图
        """
        if not isinstance(quote, str) or not quote.strip():
            raise DbToolError(
                "写操作缺少 user_intent_quote（用户原话中表达该操作意图的片段）。"
                "请勿直接执行，先向用户确认是否确实要执行此写操作。"
            )
        quote = quote.strip()[:_MAX_INTENT_QUOTE_LENGTH]
        if not any(kw in quote for kw in WRITE_INTENT_KEYWORDS):
            raise DbToolError(
                f"user_intent_quote={quote!r} 未包含明确的写操作意图（如更新/修改/删除/新增）。"
                "请勿直接执行，先与用户核对要执行的具体变更内容。"
            )
        return quote

    # =========================================================================
    # 执行辅助
    # =========================================================================

    async def _run(self, stmt, params: Dict[str, Any]) -> Any:
        """带超时保护的语句执行。"""
        try:
            return await asyncio.wait_for(
                self._session.execute(stmt, params), timeout=self._timeout
            )
        except asyncio.TimeoutError:
            raise DbToolSystemError(f"数据库操作超时（>{self._timeout}s）")

    @staticmethod
    def _elapsed(start: float) -> int:
        return int((time.perf_counter() - start) * 1000)

    @staticmethod
    def _safe_error(msg: str, exc: Exception) -> str:
        """
        构造脱敏的简短错误消息：仅保留错误类别（MySQL 错误码前缀），
        不暴露完整 SQL、参数值或堆栈（设计文档 §11）。
        """
        detail = str(exc).strip()
        # 截取首行，保留形如 (1364, "...") 的错误码与一句话说明
        line = detail.splitlines()[0] if detail else "未知数据库错误"
        return f"{msg}: {line[:160]}"

    def _record_call(self, name: str, params: Dict[str, Any], result: dict,
                     elapsed_ms: int) -> None:
        """
        记录一次工具调用的脱敏摘要（审计用）。

        参数只保留非敏感的结构信息（表名/行数/耗时），不记录完整参数值与结果内容。
        """
        self.tool_calls.append({
            "tool": name,
            "params_summary": {
                k: (f"<{len(v)} 项>" if isinstance(v, (list, dict)) else str(v)[:32])
                for k, v in params.items()
            },
            "result_summary": {
                k: v for k, v in result.items()
                if k in ("ok", "row_count", "affected_rows", "elapsed_ms")
            },
            "elapsed_ms": elapsed_ms,
        })

    # =========================================================================
    # 只读工具
    # =========================================================================

    async def list_tables(self) -> dict:
        """列出授权白名单内的表清单。"""
        start = time.perf_counter()
        result = {
            "ok": True,
            "data": sorted(self._allowed_tables),
            "row_count": len(self._allowed_tables),
            "elapsed_ms": self._elapsed(start),
        }
        self._record_call("list_tables", {}, result, result["elapsed_ms"])
        return result

    async def query_data(
        self,
        table: str,
        columns: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> dict:
        """
        按条件查询数据（参数化 + LIMIT 强制）。

        :param table: 表名（白名单内）
        :param columns: 查询列，None 表示全列（自动排除敏感列）
        :param filters: 等值过滤条件 {col: value}
        :param limit: 最大行数（默认 AGENT_QUERY_MAX_ROWS）
        """
        start = time.perf_counter()
        self._assert_table_allowed(table)
        self._assert_columns(columns)
        filters = self._validate_filters(filters)
        limit = min(int(limit or self._max_rows), self._max_rows)

        # 列选择：显式列 / 全列（排除敏感列）
        if columns:
            select_cols = ", ".join(f"`{c}`" for c in columns)
        else:
            cols = await self._fetch_columns(table)
            safe_cols = [c for c in cols if c.lower() not in SENSITIVE_COLUMNS]
            select_cols = ", ".join(f"`{c}`" for c in safe_cols)

        where_sql, params = self._build_where(filters)
        params["__limit__"] = limit
        stmt = text(f"SELECT {select_cols} FROM `{table}`{where_sql} LIMIT :__limit__")

        try:
            result = await self._run(stmt, params)
            rows = [dict(row) for row in result.mappings().all()]
        except DbToolError:
            raise
        except Exception as e:
            raise DbToolSystemError(self._safe_error("查询失败", e))

        logger.info(
            f"[DbTool-query] table={table} rows={len(rows)} elapsed={self._elapsed(start)}ms"
        )
        result = {
            "ok": True,
            "data": rows,
            "row_count": len(rows),
            "elapsed_ms": self._elapsed(start),
        }
        self._record_call(
            "query_data",
            {"table": table, "columns": columns, "limit": limit},
            result,
            result["elapsed_ms"],
        )
        return result

    async def _fetch_columns(self, table: str) -> List[str]:
        """读取表的真实列名（用于全列查询时排除敏感列）。"""
        try:
            result = await self._run(text("SHOW COLUMNS FROM `%s`" % table), {})
            return [row["Field"] for row in result.mappings().all()]
        except Exception:
            return []

    @staticmethod
    def _build_where(filters: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """构造 WHERE 子句（参数化，防止注入）。"""
        if not filters:
            return "", {}
        clauses = []
        params = {}
        for idx, (col, val) in enumerate(filters.items()):
            key = f"__w{idx}__"
            clauses.append(f"`{col}` = :{key}")
            params[key] = val
        return " WHERE " + " AND ".join(clauses), params

    # =========================================================================
    # 写工具
    # =========================================================================

    async def insert_data(
        self, table: str, values: Dict[str, Any], user_intent_quote: str
    ) -> dict:
        """
        插入单条数据（写操作二次确认：必须引用用户原话中的插入意图）。

        :param table: 表名（白名单内）
        :param values: 写入值 {col: value}
        :param user_intent_quote: 用户原话中表达"插入/新增"意图的片段
        """
        start = time.perf_counter()
        self._assert_table_allowed(table)
        values = self._validate_values(values)
        quote = self._assert_intent_quote(user_intent_quote)

        cols = ", ".join(f"`{c}`" for c in values)
        placeholders = ", ".join(f":__v{i}__" for i in range(len(values)))
        params = {f"__v{i}__": v for i, v in enumerate(values.values())}
        stmt = text(f"INSERT INTO `{table}` ({cols}) VALUES ({placeholders})")

        try:
            result = await self._run(stmt, params)
            await self._session.commit()
        except DbToolError:
            raise
        except Exception as e:
            raise DbToolSystemError(self._safe_error("插入失败", e))

        logger.info(
            f"[DbTool-insert] table={table} elapsed={self._elapsed(start)}ms"
        )
        result = {
            "ok": True,
            "affected_rows": result.rowcount,
            "elapsed_ms": self._elapsed(start),
        }
        self._record_call(
            "insert_data",
            {"table": table, "values": values, "user_intent_quote": quote},
            result,
            result["elapsed_ms"],
        )
        return result

    async def update_data(
        self,
        table: str,
        values: Dict[str, Any],
        user_intent_quote: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """
        按条件更新数据（必须携带过滤条件防止全表更新，且二次确认写意图）。

        :param table: 表名（白名单内）
        :param values: 写入值 {col: value}
        :param user_intent_quote: 用户原话中表达"更新/修改"意图的片段
        :param filters: 等值过滤条件 {col: value}，必填
        """
        start = time.perf_counter()
        self._assert_table_allowed(table)
        values = self._validate_values(values)
        filters = self._validate_filters(filters)
        quote = self._assert_intent_quote(user_intent_quote)
        if not filters:
            raise DbToolError("update_data 必须携带过滤条件，禁止全表更新")

        set_clauses = []
        params = {}
        for idx, (col, val) in enumerate(values.items()):
            key = f"__s{idx}__"
            set_clauses.append(f"`{col}` = :{key}")
            params[key] = val
        where_sql, where_params = self._build_where(filters)
        params.update(where_params)

        stmt = text(
            f"UPDATE `{table}` SET {', '.join(set_clauses)}{where_sql}"
        )
        try:
            result = await self._run(stmt, params)
            await self._session.commit()
        except DbToolError:
            raise
        except Exception as e:
            raise DbToolSystemError(self._safe_error("更新失败", e))

        logger.info(
            f"[DbTool-update] table={table} affected={result.rowcount} "
            f"elapsed={self._elapsed(start)}ms"
        )
        result = {
            "ok": True,
            "affected_rows": result.rowcount,
            "elapsed_ms": self._elapsed(start),
        }
        self._record_call(
            "update_data",
            {"table": table, "values": values, "user_intent_quote": quote},
            result,
            result["elapsed_ms"],
        )
        return result

    async def delete_data(
        self,
        table: str,
        user_intent_quote: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """
        按条件删除数据（必须携带过滤条件防止全表删除，且二次确认写意图）。

        :param table: 表名（白名单内）
        :param user_intent_quote: 用户原话中表达"删除"意图的片段
        :param filters: 等值过滤条件 {col: value}，必填
        """
        start = time.perf_counter()
        self._assert_table_allowed(table)
        filters = self._validate_filters(filters)
        quote = self._assert_intent_quote(user_intent_quote)
        if not filters:
            raise DbToolError("delete_data 必须携带过滤条件，禁止全表删除")

        where_sql, params = self._build_where(filters)
        stmt = text(f"DELETE FROM `{table}`{where_sql}")
        try:
            result = await self._run(stmt, params)
            await self._session.commit()
        except DbToolError:
            raise
        except Exception as e:
            raise DbToolSystemError(self._safe_error("删除失败", e))

        logger.info(
            f"[DbTool-delete] table={table} affected={result.rowcount} "
            f"elapsed={self._elapsed(start)}ms"
        )
        result = {
            "ok": True,
            "affected_rows": result.rowcount,
            "elapsed_ms": self._elapsed(start),
        }
        self._record_call(
            "delete_data",
            {"table": table, "user_intent_quote": quote},
            result,
            result["elapsed_ms"],
        )
        return result


# =============================================================================
# 知识库类工具（原 agent/tools/knowledge_tool.py）
# =============================================================================

# 检索条数上限（模型可传 k，超限 clamp）
MAX_K = 10
# 默认检索条数（与 knowledge 节点 RETRIEVE_TOP_K 一致）
DEFAULT_K = 4
# 单文档最大返回字数
DOC_CONTENT_LIMIT = 800
# 总格式化上下文最大字符数
TOTAL_CONTEXT_LIMIT = 4000


class KnowledgeToolExecutor:
    """知识库检索执行器：为任务 Agent 提供检索文档的工具能力。"""

    def __init__(self, session, username: Optional[str] = None):
        """
        :param session: SQLAlchemy AsyncSession（来自图运行配置 configurable.db）
        :param username: 当前登录用户名（服务端注入，用于知识库权限过滤）
        """
        self._username = username
        self._ks: Optional[Any] = None
        if session is not None:
            from agent.knowledge_service import KnowledgeService
            self._ks = KnowledgeService(db_session=session)
        # 审计记录：结构与 DbToolExecutor 对齐，供任务节点合并落库
        self.tool_calls: List[dict] = []

    # =========================================================================
    # 只读检索工具
    # =========================================================================

    async def search_knowledge(self, query: str, k: int = DEFAULT_K) -> dict:
        """
        检索知识库文档，返回格式化并截断后的上下文文本。

        模型可指定 k（clamp 到 [1, MAX_K]）；username 由构造时注入，
        不随工具参数传递，保证权限边界不可被模型绕过。

        :param query: 检索问题（知识库语义搜索）
        :param k: 返回文档数量（1~10，默认 4）
        :return: {"ok", "context", "docs_count", "truncated", "elapsed_ms"}
        """
        start = time.perf_counter()

        # 参数校验与钳制
        if not isinstance(query, str) or not query.strip():
            raise DbToolError("search_knowledge 的 query 参数必须为非空字符串")
        query = query.strip()
        k = min(max(int(k or DEFAULT_K), 1), MAX_K)

        if self._ks is None:
            return {"ok": False, "context": "", "docs_count": 0,
                    "truncated": False, "elapsed_ms": self._elapsed(start)}

        # 检索（同 knowledge 节点：稠密+稀疏一次生成，权限复用注入的 username）
        embedding, sparse = await self._ks.compute_embedding_with_sparse(query)
        docs, _ = await self._ks.retrieve(
            query, self._username, embedding, sparse=sparse, k=k,
        )

        context, truncated = self._build_truncated_context(docs)
        result = {
            "ok": True,
            "context": context,
            "docs_count": len(docs),
            "truncated": truncated,
            "elapsed_ms": self._elapsed(start),
        }
        self._record_call(query, k, result)
        return result

    # =========================================================================
    # 辅助
    # =========================================================================

    @staticmethod
    def _elapsed(start: float) -> int:
        return int((time.perf_counter() - start) * 1000)

    @classmethod
    def _build_truncated_context(cls, docs: List[Any]) -> tuple[str, bool]:
        """
        将检索文档格式化为上下文文本，并做双重长度截断。

        :param docs: 检索到的文档列表
        :return: (格式化上下文, 是否发生截断)
        """
        parts: List[str] = []
        total = 0
        truncated = False
        for idx, doc in enumerate(docs, 1):
            content = (doc.page_content or "") if hasattr(doc, "page_content") else str(doc)
            if len(content) > DOC_CONTENT_LIMIT:
                content = content[:DOC_CONTENT_LIMIT]
                truncated = True
            block = f"[文档{idx}]\n{content}"
            remain = TOTAL_CONTEXT_LIMIT - total
            if len(block) > remain:
                if remain > 0:
                    parts.append(block[:remain])
                truncated = True
                break
            parts.append(block)
            total += len(block)
        return "\n\n".join(parts), truncated

    def _record_call(self, query: str, k: int, result: dict) -> None:
        """
        记录一次检索调用的脱敏摘要（审计用，结构对齐 DbToolExecutor）。

        只记录查询意图前 64 字符与结果统计，不落完整上下文字段。
        """
        self.tool_calls.append({
            "tool": "search_knowledge",
            "params_summary": {
                "query": query[:64],
                "k": k,
            },
            "result_summary": {
                kk: v for kk, v in result.items()
                if kk in ("ok", "docs_count", "truncated", "elapsed_ms")
            },
            "elapsed_ms": result["elapsed_ms"],
        })


# =============================================================================
# 追问类工具（原 tools/clarify_tool.py）
# =============================================================================

# 中断载荷业务类型标识（与 api/routes/agent.py 的追问识别保持一致）
CLARIFY_TYPE = "clarification"

# 工具名常量（任务节点按此过滤非审批模式下的绑定）
ASK_USER_TOOL_NAME = "ask_user"

# 单批追问问题数量上限（提示词与前台向导按此约束渲染）
_MAX_QUESTIONS = 5


class ClarifyTool:
    """
    追问类工具（Human-in-the-loop clarification）。

    当任务 Agent 掌握的信息不足以完成任务（如缺少必要参数、条件不明确）时，
    调用 ask_user 向用户中断追问：任务暂停并推送追问卡片，用户在前台批量
    回答或取消后以同一 thread 恢复执行，工具返回对应的处理上下文供 Agent
    继续处理。仅 ask_user 一个工具（经 @tool 声明为模块级 BaseTool，供扫描注册）。
    """


@tool
async def ask_user(questions: List[str]) -> str:
    """当执行任务所需的关键信息不足时，向用户批量追问以获取补充信息。

    适用场景：用户请求缺少必要参数（如查询条件不明确、数据主键缺失、
    执行范围不清晰、多义表述需确认等），继续执行会猜测或出错时。
    调用后任务会暂停，等待用户在前台批量回答。

    使用要求：
    - 可一次合并追问 1 到 5 个问题，每个问题必须**具体、独立、可单独回答**；
    - 不要问"是否足够/是否继续"等模糊问题，不要重复追问已明确的信息；
    - 尽量精简：只有确实缺失且相互独立的关键信息才列入，能不问就不问。

    工具返回值：用户填写的问题-答案列表；若用户取消追问，则返回取消说明，
    此时请如实告知用户"其已取消提供相关信息"，基于现有信息尽力处理，
    并给出补充哪些关键信息后可以重试的修复建议，不要再追问。
    """
    payload: Any = interrupt(
        {"type": CLARIFY_TYPE, "questions": _clamp_questions(questions)}
    )
    return _format_resume(payload, _clamp_questions(questions))


def _clamp_questions(questions: Any) -> list:
    """
    追问问题列表的长度与格式兜底（阶段 4 加固）。

    背景：单批 ≤5 个问题此前**只写在提示词里**，代码未兜底——模型偶尔会
    生成 10+ 条问题，导致前台向导卡片超长、审批/追问单存储膨胀。
    这里做服务端硬约束：过滤空值 → 截断到 `_MAX_QUESTIONS` → 记录告警。

    :param questions: LLM 生成的问题列表（可能为任意结构）
    :return: 归一化后的问题列表（1.._MAX_QUESTIONS 条）
    """
    items = [str(q).strip() for q in (questions or []) if str(q).strip()]
    if len(items) > _MAX_QUESTIONS:
        logger.warning(
            f"[Agent-追问] 问题数 {len(items)} 超过上限 {_MAX_QUESTIONS}，已截断"
        )
        from agent.observability import metrics

        metrics.incr("agent.clarify.truncated")
        items = items[:_MAX_QUESTIONS]
    return items or ["请补充完成任务所需的关键信息"]


def _format_resume(payload: Any, questions: List[str]) -> str:
    """将恢复载荷（批量答案 / 取消）格式化为给 LLM 的处理上下文。"""
    if isinstance(payload, dict) and payload.get("canceled"):
        return (
            "用户取消了本次追问，未提供任何补充信息。请据此回复用户："
            "1) 明确告知用户其已取消提供相关信息；"
            "2) 基于现有信息给出尽可能的处理结果或说明；"
            "3) 给出修复建议（如需用户补充哪些关键信息后可重新发起）。"
            "不要再次追问，也不要要求用户立即重试。"
        )
    answers = _extract_answers(payload, len(questions))
    lines = [
        f"{i + 1}. 问题：{q}\n   回答：{a or '（未填写）'}"
        for i, (q, a) in enumerate(zip(questions, answers))
    ]
    return "用户对追问的回答：\n" + "\n".join(lines)


def _extract_answers(payload: Any, expected: int) -> List[str]:
    """从恢复载荷中提取答案列表（兼容批量 answers / 旧单 answer / 裸列表）。"""
    if isinstance(payload, dict):
        if isinstance(payload.get("answers"), list):
            answers = [str(a) for a in payload["answers"]]
        elif "answer" in payload:
            answers = [str(payload["answer"])]
        else:
            answers = []
    elif isinstance(payload, list):
        answers = [str(a) for a in payload]
    elif payload is not None:
        answers = [str(payload)]
    else:
        answers = []
    # 回答数不足时补空位（对应前台跳过未填的场景），保持与问题一一对应
    if len(answers) < expected:
        answers += [""] * (expected - len(answers))
    return answers[:expected]


# =============================================================================
# 工具元数据注册表（原 agent/tools/registry.py）
# =============================================================================


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
    handler: str                 # 执行器上的执行方法名（DbToolExecutor / KnowledgeToolExecutor）


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
