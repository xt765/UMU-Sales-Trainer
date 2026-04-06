"""UMU Sales Trainer 主入口模块。

提供 FastAPI 应用实例和生命周期管理，包括中间件注册、
路由初始化、数据库初始化和事件处理。
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from umu_sales_trainer.api.middleware import (
    LoggingMiddleware,
    RateLimitMiddleware,
    TokenCountMiddleware,
)
from umu_sales_trainer.api.router import api_router
from umu_sales_trainer.services.database import get_db_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8080",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8080",
]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """应用生命周期上下文管理器。

    处理启动和关闭事件：
    - 启动时初始化数据库
    - 关闭时执行清理工作

    Args:
        app: FastAPI 应用实例

    Yields:
        控制权返回给应用
    """
    logger.info("Starting UMU Sales Trainer...")
    db_service = get_db_service()
    db_service.init_db()
    logger.info("Database initialized successfully")
    yield
    logger.info("Shutting down UMU Sales Trainer...")


def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用实例。

    注册所有中间件、路由和 CORS 配置。

    Returns:
        配置完成的 FastAPI 应用实例
    """
    app = FastAPI(
        title="UMU Sales Trainer API",
        description="AI-powered sales training system API",
        version="0.1.0",
        lifespan=lifespan,
    )

    static_dir = Path(__file__).parent.parent.parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(TokenCountMiddleware)

    app.include_router(api_router)

    return app


app = create_app()


@app.get("/")
async def root():
    """根路由，重定向到前端页面。"""
    from fastapi.responses import RedirectResponse

    return RedirectResponse(url="/static/index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "umu_sales_trainer.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
