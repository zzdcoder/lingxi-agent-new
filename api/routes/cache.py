"""
缓存管理路由

提供缓存的管理接口：清除缓存、查看统计。

**§18.13**：系统现在有**两层**缓存 —— 答案缓存（SemanticCache）与检索缓存
（RetrievalCache）。`/clear` 必须**同时清两层**：只清答案缓存会把「检索缓存」
留成唯一的陈旧数据源（知识库更新后仍返回旧文档片段），比全清更危险。
"""

import logging
from fastapi import APIRouter, Depends

from rag.semantic_cache import get_semantic_cache
from rag.retrieval_cache import get_retrieval_cache
from utils.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cache", tags=["缓存管理"])


@router.post("/clear")
async def clear_cache(current_user=Depends(get_current_user)):
    """
    手动清除全部缓存（答案缓存 + 检索缓存）。

    需要登录权限。适用于知识库手动更新后强制刷新缓存。
    """
    cleared = 0
    detail: dict = {}
    try:
        cache = get_semantic_cache()
        if cache:
            n = await cache.invalidate_all()
            detail["answer"] = n
            cleared += n

        rcache = get_retrieval_cache()
        if rcache:
            n = await rcache.invalidate_all()
            detail["retrieve"] = n
            cleared += n

        logger.info(
            f"手动清除缓存: 共 {cleared} 条（{detail}）(by {current_user.username})"
        )
        return {"message": "缓存已清除", "cleared": cleared, "detail": detail}

    except Exception as e:
        logger.error(f"清除缓存失败: {e}")
        return {"message": f"清除缓存失败: {str(e)}", "cleared": 0, "detail": detail}


@router.get("/stats")
async def cache_stats(current_user=Depends(get_current_user)):
    """
    查看两层缓存的统计信息。

    返回：答案缓存与检索缓存各自的条目数、总命中次数、配置参数等。
    """
    answer_cache = get_semantic_cache()
    retrieve_cache = get_retrieval_cache()

    result: dict = {
        "answer": await _safe_stats(answer_cache),
        "retrieve": await _safe_stats(retrieve_cache),
    }
    result["enabled"] = bool(
        result["answer"].get("enabled") or result["retrieve"].get("enabled")
    )
    return result


async def _safe_stats(cache) -> dict:
    """取单层统计，异常时不影响另一层的返回。"""
    if not cache:
        return {"enabled": False, "message": "未启用"}
    try:
        return await cache.get_stats()
    except Exception as e:
        logger.error(f"获取缓存统计失败: {e}")
        return {"enabled": False, "error": str(e)}
