"""
数据库工具执行器

为任务 Agent 提供安全、参数化的数据库访问能力，覆盖设计文档 §5.5.2 / §7.4 / §11 安全基线：

1. 白名单：仅允许操作 AGENT_DB_ALLOWED_TABLES 内的表；
2. 列名校验：正则校验 + 排除敏感列；
3. 参数化：全部语句使用 text() + 绑定参数，禁止拼接 SQL；
4. LIMIT 强制：查询最多返回 AGENT_QUERY_MAX_ROWS 行；
5. 超时保护：单次执行受 AGENT_TASK_TIMEOUT_SECONDS 约束。

每个方法返回结构化结果：{"ok": True, "data": [...], "row_count": N, "elapsed_ms": M}
写操作返回 {"ok": True, "affected_rows": N, ...}
"""
import asyncio
import logging
import re
import time
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from langchain_core.tools import ToolException

from core.config import settings

logger = logging.getLogger(__name__)

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
    数据库工具执行错误（安全拦截 / 参数非法 / 执行失败）。

    继承 langchain_core 的 ToolException：工具框架层默认只把 ToolException
    视为"业务可恢复错误"，配合 create_agent 工具绑定的 handle_tool_error
    可将异常转为 ToolMessage 送回 Agent 循环（模型可自我纠正），
    而非直接中断整个 Agent 循环。
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
            raise DbToolError(f"数据库操作超时（>{self._timeout}s）")

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
            raise DbToolError(self._safe_error("查询失败", e))

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
            raise DbToolError(self._safe_error("插入失败", e))

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
            raise DbToolError(self._safe_error("更新失败", e))

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
            raise DbToolError(self._safe_error("删除失败", e))

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
