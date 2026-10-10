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

def _resolve_pool_pre_ping() -> bool:
    """
    决定是否启用 `pool_pre_ping`（§25.11）。

    ⚠️ **生产事故（2026-09-23）**：直接把 `pool_pre_ping=True` 交给
    `create_async_engine` 会让**所有复用连接的接口返回 400**。根因链：

    1. SQLAlchemy 的 `MySQLDialect_pymysql._send_false_to_ping` 用
       **pymysql** 的 `Connection.ping` 签名做探测（见 `dialects/mysql/pymysql.py`）；
    2. pymysql >= 2.2 把签名改成了 `ping(self, reconnect=False)`，
       探测逻辑判定「默认值是 False」→ 返回 **False**；
    3. 于是 `do_ping` 走 `dbapi_connection.ping()` —— **不传参**；
    4. 但 aiomysql 下这个对象是 SQLAlchemy 自己的
       `AsyncAdapt_aiomysql_connection.ping(self, reconnect)` —— **没有默认值**
       → `TypeError: ... missing 1 required positional argument: 'reconnect'`；
    5. 只有**复用**连接才做 pre_ping（新建连接不探测），所以表现为
       「第一条请求正常，之后全部失败」（2026-09-23 的登录成功、会话列表/新建会话 400）。

    即：**SQLAlchemy 用 pymysql 的签名去决定怎么调 aiomysql 的适配器** ——
    上游的签名探测与实际调用对象不一致。

    处置（fail-safe 优先）：
    - 默认关闭（`db_pool_pre_ping=False`），与事故前行为一致；
    - 显式开启但**未**开兼容补丁 → 打 ERROR 日志并**强制关闭**（宁可不探活，
      也不能让整个 API 挂掉）；
    - 开启兼容补丁（`db_pool_pre_ping_shim=True`）→ 覆盖方言的
      `_send_false_to_ping` 为 True，使适配器恒定收到 `reconnect` 参数，
      此时 pre_ping 才真正可用。
    """
    if not settings.db_pool_pre_ping:
        return False

    if not settings.db_pool_pre_ping_shim:
        logger.error(
            "[DB] pool_pre_ping 已配置开启，但当前依赖组合（aiomysql + "
            "pymysql>=2.2）下会导致「复用连接时 TypeError: ping() missing "
            "1 required positional argument: 'reconnect'」，已**强制关闭**以避免"
            "接口大面积 400。若确需探活，请同时设置 DB_POOL_PRE_PING_SHIM=true。"
        )
        return False

    try:
        from sqlalchemy.dialects.mysql.aiomysql import MySQLDialect_aiomysql

        # 覆盖 memoized_property：aiomysql 适配器的 ping(self, reconnect)
        # **恒定**接受 reconnect 参数，因此无条件传 False 才是正确调用形态。
        MySQLDialect_aiomysql._send_false_to_ping = True
    except Exception as e:  # 防御：补丁失败就退回不探活，绝不影响建引擎
        logger.error(f"[DB] pool_pre_ping 兼容补丁应用失败，已强制关闭: {e}")
        return False

    logger.warning(
        "[DB] pool_pre_ping 已开启（附兼容补丁）。注意：每次取出连接都会多一次"
        "ping 往返；若上游修复了该签名探测问题，可移除补丁。"
    )
    return True


# 异步引擎
#
# §25.11：连接池健壮性。
# 本系统在多个层级用 `asyncio.wait_for` **取消在途的数据库操作**
# （单次工具超时 `with_tool_timeout`、语句级 `_run`、任务墙钟超时），
# 被取消时连接可能来不及正常归还 —— 这正是生产日志里
# "The garbage collector is trying to clean up non-checked-in connection"
# 的来源。回收与探活不能消除取消本身（根治需要把超时下沉到服务端），
# 但能保证**损坏的连接不会被发给下一个请求**。
async_engine = create_async_engine(
    get_database_url(),
    pool_pre_ping=_resolve_pool_pre_ping(),
    pool_recycle=settings.db_pool_recycle_seconds,
    echo=settings.db_echo,
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
            # 仅在会话仍处于活动事务中时才提交，避免重复 commit
            if session.is_active:
                await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

if __name__ == '__main__':
    logger.debug("数据库连接url: %s", get_database_url())
