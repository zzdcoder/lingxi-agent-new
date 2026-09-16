"""
工具管理路由

提供工具注册表（tool_registry）的管理能力：
- 查询工具列表（按状态/分类/关键字过滤）；
- 调整工具状态（启用/下线/手动熔断/恢复），支撑运维治理（设计文档 §7.7.3 / §9）。

状态语义：1=生效，0=失效，2=熔断。
"""
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from models.tool_model import ToolRegistry
from models.tool_schema import ToolOut, ToolStatusUpdate
from agent.tools.circuit_breaker import breaker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tools", tags=["工具管理"])


@router.get(
    "",
    response_model=list[ToolOut],
    summary="工具列表",
    description="查询工具注册表中的工具，可按状态 / 分类 / 名称关键字过滤",
)
async def list_tools(
    status: Optional[int] = Query(default=None, description="状态过滤: 1=生效 0=失效 2=熔断"),
    category: Optional[str] = Query(default=None, description="分类过滤: db/knowledge/custom"),
    keyword: Optional[str] = Query(default=None, description="名称或描述关键字"),
    db: AsyncSession = Depends(get_db),
) -> list[ToolRegistry]:
    query = select(ToolRegistry)
    if status is not None:
        query = query.where(ToolRegistry.status == status)
    if category:
        query = query.where(ToolRegistry.category == category)
    if keyword:
        like = f"%{keyword}%"
        query = query.where(
            (ToolRegistry.name.like(like)) | (ToolRegistry.description.like(like))
        )
    query = query.order_by(ToolRegistry.status.desc(), ToolRegistry.category, ToolRegistry.name)
    result = await db.execute(query)
    return list(result.scalars().all())


@router.get(
    "/{name}",
    response_model=ToolOut,
    summary="工具详情",
    description="按工具名查询单个工具的完整信息（含熔断状态与调用统计）",
)
async def get_tool_detail(
    name: str,
    db: AsyncSession = Depends(get_db),
) -> ToolRegistry:
    result = await db.execute(select(ToolRegistry).where(ToolRegistry.name == name))
    tool = result.scalars().first()
    if tool is None:
        raise HTTPException(status_code=404, detail=f"工具 {name} 不存在")
    return tool


@router.patch(
    "/{name}/status",
    response_model=ToolOut,
    summary="调整工具状态",
    description=(
        "人工调整工具状态（运维操作）：1↔0（生效/下线）、1→2（手动熔断）、"
        "2→1（恢复）、2→0（熔断后下线）。熔断/下线后，任务节点将不再绑定该工具。"
    ),
)
async def update_tool_status(
    name: str,
    body: ToolStatusUpdate,
    db: AsyncSession = Depends(get_db),
) -> ToolRegistry:
    try:
        tool = await breaker.manual_transition(db, name, body.status, body.remark)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if tool is None:
        raise HTTPException(status_code=404, detail=f"工具 {name} 不存在")
    return tool
