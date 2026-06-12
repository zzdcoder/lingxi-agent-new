"""
数据库层

提供异步 SQLAlchemy 引擎、会话工厂与 ORM 基类
"""
from dotenv import load_dotenv
import os
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from core.config import settings
import logging

load_dotenv(".env.dev")
logger=logging.getLogger(__name__)
def get_database_url():
    return f"mysql+aiomysql://{os.getenv('db_user')}:{os.getenv('db_password')}@{os.getenv('db_host')}/{os.getenv('db_name')}"

# 异步引擎
async_engine = create_async_engine(
    get_database_url(),
    pool_pre_ping=False,
    echo=True,
)

# 异步会话工厂
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


class Base(DeclarativeBase):
    """所有 ORM 模型的基类"""
    pass


async def get_db() -> AsyncSession:
    """FastAPI Dependency：为每个请求创建一个异步数据库会话"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

if __name__ == '__main__':
    logger.debug("数据库连接url: %s", get_database_url())
