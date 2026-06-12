"""
元数据定义路由

提供元数据字段的增删改查接口
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete

from core.database import get_db
from models.metadata_model import MetadataDefinition
from models.metadata_schema import MetadataCreate, MetadataUpdate, MetadataOut

router = APIRouter(prefix="/metadata", tags=["元数据管理"])


@router.get(
    "",
    response_model=list[MetadataOut],
    summary="列出所有元数据定义",
    description="返回系统中所有元数据字段定义，包含内置与自定义字段",
)
async def list_metadata(db: AsyncSession = Depends(get_db)) -> list[MetadataDefinition]:
    result = await db.execute(select(MetadataDefinition).order_by(MetadataDefinition.created_at))
    return list(result.scalars().all())


@router.post(
    "",
    response_model=MetadataOut,
    status_code=status.HTTP_201_CREATED,
    summary="创建元数据定义",
    description="创建一个新的自定义元数据字段",
)
async def create_metadata(
    data: MetadataCreate,
    db: AsyncSession = Depends(get_db),
) -> MetadataDefinition:
    # 检查名称是否已存在
    existing = await db.execute(
        select(MetadataDefinition).where(MetadataDefinition.name == data.name)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"元数据字段 '{data.name}' 已存在",
        )

    definition = MetadataDefinition(
        name=data.name,
        display_name=data.display_name,
        field_type=data.field_type,
        is_builtin=False,
        is_enabled=True,
    )
    db.add(definition)
    await db.commit()
    await db.refresh(definition)
    return definition


@router.put(
    "/{metadata_id}",
    response_model=MetadataOut,
    summary="更新元数据定义",
    description="更新指定元数据字段的信息",
)
async def update_metadata(
    metadata_id: str,
    data: MetadataUpdate,
    db: AsyncSession = Depends(get_db),
) -> MetadataDefinition:
    result = await db.execute(
        select(MetadataDefinition).where(MetadataDefinition.id == metadata_id)
    )
    definition = result.scalar_one_or_none()
    if not definition:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="元数据字段不存在",
        )

    if definition.is_builtin:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="内置元数据字段不可修改",
        )

    update_data = data.model_dump(exclude_unset=True)
    if update_data:
        await db.execute(
            update(MetadataDefinition)
            .where(MetadataDefinition.id == metadata_id)
            .values(**update_data)
        )
        await db.commit()
        await db.refresh(definition)

    return definition


@router.delete(
    "/{metadata_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="删除元数据定义",
    description="删除指定的自定义元数据字段",
)
async def delete_metadata(
    metadata_id: str,
    db: AsyncSession = Depends(get_db),
) -> None:
    result = await db.execute(
        select(MetadataDefinition).where(MetadataDefinition.id == metadata_id)
    )
    definition = result.scalar_one_or_none()
    if not definition:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="元数据字段不存在",
        )

    if definition.is_builtin:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="内置元数据字段不可删除",
        )

    await db.execute(
        delete(MetadataDefinition).where(MetadataDefinition.id == metadata_id)
    )
    await db.commit()
