"""
FastAPI 应用入口

组装应用、注册路由、挂载全局异常处理与生命周期事件
"""

import asyncio
import logging
import os
import socket
import sys
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from sqlalchemy import select

from core.config import settings
from core.database import async_engine, Base, AsyncSessionLocal
from core.exceptions import RAGException
from api.routes import attachments
from api.routes import auth
from api.routes import metadata
from api.routes import file_process
from api.routes import conversation
from api.routes import cache as cache_routes
from api.routes import agent as agent_routes
from api.routes import approval as approval_routes
from api.routes import tools as tools_routes
from models.metadata_model import MetadataDefinition
from utils.captcha import cleanup_expired_captchas

# 配置全局日志：输出到控制台，级别 INFO，带时间戳和模块名。
# 注意：trace_id 字段**不由此处声明**，而是由 observability.install_trace_logging()
# 在过滤器装配成功后注入（避免观测初始化失败导致全量日志崩溃）。
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)
load_dotenv(".env.dev")

# 观测初始化（阶段 4）：
# ① trace_id 注入所有既有 logger；② LangSmith 追踪开关。
# 必须在任何可能造成请求处理前完成，且不因失败阻塞启动。
try:
    from agent.observability import init_tracing, install_trace_logging

    install_trace_logging(level=logging.INFO)
    init_tracing(
        settings.langsmith_tracing,
        project=settings.langsmith_project,
        endpoint=settings.langsmith_endpoint,
    )
except Exception as e:  # 观测自身异常不得阻塞启动
    print(f"[观测] 初始化失败（降级为无观测模式）: {e}")

# 配置 ChromaDB 日志级别
logging.getLogger("chromadb").setLevel(logging.INFO)

# 服务端口配置（集中管理，避免冲突）
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8878"))  # ChromaDB 端口


def _check_port_available(port: int) -> bool:
    """检测端口是否可用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", port)) != 0

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理：启动时建表并初始化内置元数据"""
    try:
        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except Exception as exc:
        logger.warning(f"数据库初始化失败，应用将以降级模式运行: {exc}")

    # 同步工具注册表（启动注册）：扫描 @tool 装饰器工具 + ToolSpec 注册表 → tool_registry 表
    # 失败不阻塞启动（任务节点绑定工具时降级为无工具可用，见 agent/tools/manager.py）
    if settings.tool_registry_sync_on_start:
        try:
            from agent.tools.registrar import sync_tool_registry
            async with AsyncSessionLocal() as session:
                await sync_tool_registry(session)
            logger.info("工具注册表同步完成")
        except Exception as e:
            logger.warning(f"工具注册表同步失败（不阻塞启动）: {e}")

    # 初始化混合检索器（启动时预热，含 Cross-Encoder 重排序模型）
    try:
        from rag.rag_conversation_service import init_hybrid_retriever
        init_hybrid_retriever()
        logger.info("混合检索器启动预热完成")
    except Exception as e:
        logger.error(f"混合检索器启动预热失败: {e}")

    # 初始化语义缓存（基于 Qdrant，降级为无缓存模式）
    try:
        from rag.semantic_cache import init_semantic_cache
        init_semantic_cache()
        logger.info("语义缓存初始化完成")
    except Exception as e:
        logger.warning(f"语义缓存初始化失败（降级为无缓存模式）: {e}")

    # 启动验证码过期清理后台任务
    async def captcha_cleanup_loop():
        while True:
            await asyncio.sleep(60)
            removed = cleanup_expired_captchas()
            if removed > 0:
                logger.info(f"清理了 {removed} 条过期验证码记录")

    cleanup_task = asyncio.create_task(captcha_cleanup_loop())

    # 启动语义缓存过期清理后台任务（每小时执行一次）
    async def cache_cleanup_loop():
        while True:
            await asyncio.sleep(3600)
            try:
                from rag.semantic_cache import get_semantic_cache
                cache = get_semantic_cache()
                if cache:
                    removed = await cache.cleanup_expired()
                    if removed > 0:
                        logger.info(f"清理了 {removed} 条过期语义缓存")
            except Exception as e:
                logger.debug(f"语义缓存清理任务异常: {e}")

    cache_task = asyncio.create_task(cache_cleanup_loop())

    # 启动熔断器计数批量落库后台任务（内存计数合并写回 tool_registry）
    async def circuit_flush_runner():
        try:
            from agent.tools.circuit_breaker import flush_loop
            await flush_loop(AsyncSessionLocal)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"[熔断器] 后台批量落库任务退出异常: {e}")

    circuit_task = asyncio.create_task(circuit_flush_runner())

    # 启动熔断恢复定时探测（APScheduler）：每 N 秒扫描熔断工具并重放探测恢复。
    # 独立调度器对象，随 lifespan 启动/关闭；多 worker 部署需额外保证单点（当前单进程）。
    scheduler = None
    if settings.agent_probe_enabled:
        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler
            from apscheduler.triggers.interval import IntervalTrigger

            from agent.tools.probe import scan_and_probe

            scheduler = AsyncIOScheduler(timezone="Asia/Shanghai")
            scheduler.add_job(
                scan_and_probe,
                trigger=IntervalTrigger(seconds=settings.agent_probe_interval_seconds),
                id="tool_circuit_probe",
                max_instances=1,
                coalesce=True,
                misfire_grace_time=30,
            )
            scheduler.start()
            logger.info(
                f"熔断恢复定时探测已启动（间隔 {settings.agent_probe_interval_seconds}s）"
            )
        except Exception as e:
            scheduler = None
            logger.warning(f"熔断恢复定时探测启动失败（熔断工具将只能人工恢复）: {e}")

    # 初始化审批检查点（MySQL，失败降级为无审批模式）
    # 必须早于主图预热：主图编译需要携带 checkpointer
    try:
        from agent.graph_builder import init_checkpointer
        await init_checkpointer()
    except Exception as e:
        logger.warning(f"审批检查点初始化异常（降级为无审批模式）: {e}")

    # 预热 Agent 主图（失败不阻塞启动，/api/agent/chat 首次调用时自动编译）
    try:
        from agent.graph_builder import get_graph
        get_graph()
        logger.info("Agent 主图编译预热完成")
    except Exception as e:
        logger.warning(f"Agent 主图预热失败（首次调用时重试）: {e}")

    yield

    # 熔断器退出前冲刷待落库计数，避免最后窗口内计数丢失
    try:
        from agent.tools.circuit_breaker import breaker
        async with AsyncSessionLocal() as session:
            await breaker.flush(session)
    except Exception as e:
        logger.warning(f"[熔断器] 退出冲刷异常: {e}")

    # 取消后台任务
    cleanup_task.cancel()
    cache_task.cancel()
    circuit_task.cancel()

    # 关闭熔断恢复定时探测调度器
    if scheduler is not None:
        try:
            scheduler.shutdown(wait=False)
        except Exception as e:
            logger.warning(f"熔断恢复定时探测调度器关闭异常: {e}")
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

    # 关闭审批检查点连接
    try:
        from agent.graph_builder import close_checkpointer
        await close_checkpointer()
    except Exception as e:
        logger.warning(f"审批检查点关闭异常: {e}")


app = FastAPI(
    title="lingxi-agent",
    debug=os.getenv("DEBUG", "").lower() in ("true", "1", "yes"),
    lifespan=lifespan
)

# CORS 中间件（允许前端开发环境跨域）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 全局异常处理器
@app.exception_handler(RAGException)
async def rag_exception_handler(request: Request, exc: RAGException):
    return JSONResponse(
        status_code=exc.code,
        content={"detail": exc.message},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("未捕获的异常")
    return JSONResponse(
        status_code=500,
        content={"detail": "服务器内部错误"},
    )


# 挂载本地上传文件静态资源
os.makedirs("uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# 注册路由
app.include_router(attachments.router, prefix="/api")
app.include_router(auth.router, prefix="/api")
app.include_router(metadata.router, prefix="/api")
app.include_router(file_process.router, prefix="/api")
app.include_router(conversation.router, prefix="/api")
app.include_router(cache_routes.router, prefix="/api")
app.include_router(agent_routes.router, prefix="/api")
app.include_router(approval_routes.router, prefix="/api")
app.include_router(tools_routes.router, prefix="/api")


@app.get("/health", tags=["健康检查"])
async def health_check(detailed: bool = False):
    """
    健康检查。

    :param detailed: 为 true 时返回**依赖级别的详细体检**（阶段 4 观测）：
        业务库连通性、LangGraph 检查点（决定审批模式是否生效）、主图编译状态、
        工具注册/熔断分布、混合检索器与语义缓存初始化状态。
        生产建议由监控系统周期性拉取 `?detailed=true`，对 `status != ok` 告警。

    :return: 健康快照；精简模式保持向后兼容（{"status": "ok"}）
    """
    if not detailed:
        return {"status": "ok"}
    try:
        from agent.health import collect_health

        from agent.observability import get_trace_id

        snapshot = await collect_health(AsyncSessionLocal)
        snapshot["trace_id"] = get_trace_id()
        return snapshot
    except Exception as e:
        logger.warning(f"健康巡检失败: {e}")
        return {"status": "degraded", "detail": f"健康巡检失败: {str(e)[:200]}"}