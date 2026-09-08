"""
灵犀智能助手 - 统一启动入口

企业级启动脚本，支持开发和生产两种模式。

用法:
    # 开发模式（热重载）
    python run.py --dev

    # 生产模式
    python run.py

    # 自定义参数
    python run.py --host 127.0.0.1 --port 8080
"""

import argparse
import logging
import os
import sys

from dotenv import load_dotenv
import uvicorn

# 加载环境变量
ENV_FILE = ".env.dev" if os.path.exists(".env.dev") else ".env"
load_dotenv(ENV_FILE)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="灵犀智能助手 API 服务",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --dev                  # 开发模式（热重载）
  %(prog)s --dev --port 8080      # 开发模式，自定义端口
  %(prog)s --host 0.0.0.0         # 生产模式，监听所有网卡
        """
    )

    parser.add_argument(
        "--dev",
        action="store_true",
        help="启用开发模式（热重载、详细日志）",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("HOST", "0.0.0.0"),
        help="绑定主机地址 (默认: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "8000")),
        help="绑定端口 (默认: 8000)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("WORKERS", "1")),
        help="工作进程数，生产模式建议设置为 CPU 核心数 (默认: 1)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # 开发模式配置
    if args.dev:
        logger.info("=" * 50)
        logger.info("🚀 开发模式启动")
        logger.info("=" * 50)
        logger.info(f"📁 热重载已启用")
        logger.info(f"🔧 日志级别: INFO")
        logger.info(f"🌐 服务地址: http://{args.host}:{args.port}")
        logger.info(f"📚 API 文档: http://{args.host}:{args.port}/docs")
        logger.info("=" * 50)

        uvicorn.run(
            "app.main:app",
            host=args.host,
            port=args.port,
            reload=True,
            reload_dirs=["app", "api", "rag", "models", "core", "utils", "embeddings", "ingestion", "prompt", "agent"],
            reload_excludes=[
                "qdrant_data",
                "qdrant_data_backup",
                "uploads",
                "temp",
                ".venv",
                ".idea",
                "*.pyc",
                "__pycache__",
                "*.log",
                "*.sqlite3",
                "*.db",
                "*.tmp",
                "*.bak",
            ],
            log_level="debug",
            access_log=True,
        )
    else:
        # 生产模式配置
        logger.info("=" * 50)
        logger.info("🚀 生产模式启动")
        logger.info("=" * 50)
        logger.info(f"🌐 服务地址: http://{args.host}:{args.port}")
        logger.info(f"👷 工作进程: {args.workers}")
        logger.info("=" * 50)

        uvicorn.run(
            "app.main:app",
            host=args.host,
            port=args.port,
            workers=args.workers,
            reload=False,
            log_level="info",
            access_log=True,
        )


if __name__ == "__main__":
    main()
