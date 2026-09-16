"""
知识库检索工具执行器

为任务 Agent 提供"执行任务时检索相关文档/政策"的能力（RAG-as-tool），
覆盖设计文档 §5.5.2 写操作与检索双保险中的只读检索链路：

1. 只读：风险等级 READ，不触发人工审批；
2. 权限注入：username 由服务端从会话上下文注入，不暴露为工具参数，
   防止模型伪造身份越权检索私有文档（对齐 knowledge 节点的权限过滤）；
3. 成本护栏：k 参数 clamp 到 [1, MAX_K]，防模型自主放大检索开销；
4. 长度截断：单文档与总上下文均设上限，避免长文档撑爆 Agent 上下文。

返回结构化结果：{"ok": True, "context": "...", "docs_count": N,
"truncated": bool, "elapsed_ms": M}
"""
import logging
import time
from typing import Any, List, Optional

from agent.tools.db_tools import DbToolError

logger = logging.getLogger(__name__)

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