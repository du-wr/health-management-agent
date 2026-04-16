import logging
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session

from app.api.routes import router
from app.core.config import get_settings
from app.core.database import engine, init_db
from app.services.knowledge_service import knowledge_service
from app.worker import run_worker


logger = logging.getLogger(__name__)
settings = get_settings()
_embedded_worker_lock = threading.Lock()
_embedded_worker_started = False


def _start_embedded_worker_if_needed() -> None:
    """按配置启动内嵌报告 worker，避免本地只开 API 时队列无人消费。"""
    global _embedded_worker_started
    if not settings.report_queue_run_embedded_worker:
        return

    with _embedded_worker_lock:
        if _embedded_worker_started:
            return
        worker_thread = threading.Thread(
            target=run_worker,
            name="embedded-report-worker",
            daemon=True,
        )
        worker_thread.start()
        _embedded_worker_started = True
        logger.info("Embedded report worker started.")


app = FastAPI(title=settings.app_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router, prefix="/api")


@app.on_event("startup")
def on_startup() -> None:
    """应用启动时初始化数据库、知识库，并按需启动内嵌解析 worker。"""
    init_db()
    with Session(engine) as session:
        knowledge_service.ensure_initialized(session)
    _start_embedded_worker_if_needed()


@app.get("/healthz")
def healthcheck() -> dict[str, str]:
    """最简单的健康检查接口，用来确认服务本身是否正常启动。"""
    return {"status": "ok"}
