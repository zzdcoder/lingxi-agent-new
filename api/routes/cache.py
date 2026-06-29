"""
缓存管理路由

提供语义缓存的管理接口：清除缓存、查看统计
"""

import logging
from fastapi import APIRouter, Depends

from rag.semantic_cache import get_semantic_cache
from utils.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cache", tags=["缓存管理"])


@router.post("/clear")
async def clear_cache(current_user=Depends(get_current_user)):
    """
    手动清除全部语义缓存。

    需要登录权限。适用于知识库手动更新后强制刷新缓存。
    """
    try:
        cache = get_semantic_cache()
        if not cache:
            return {"message": "语义缓存未启用", "cleared": 0}

        cleared = await cache.invalidate_all()
        logger.info(f"手动清除语义缓存: {cleared} 条 (by {current_user.username})")
        return {"message": f"缓存已清除", "cleared": cleared}

    except Exception as e:
        logger.error(f"清除缓存失败: {e}")
        return {"message": f"清除缓存失败: {str(e)}", "cleared": 0}


@router.get("/stats")
async def cache_stats(current_user=Depends(get_current_user)):
    """
    查看语义缓存统计信息。

    返回：条目数、总命中次数、配置参数等。
    """
    try:
        cache = get_semantic_cache()
        if not cache:
            return {"enabled": False, "message": "语义缓存未启用"}

        stats = await cache.get_stats()
        return stats

    except Exception as e:
        logger.error(f"获取缓存统计失败: {e}")
        return {"enabled": False, "error": str(e)}
