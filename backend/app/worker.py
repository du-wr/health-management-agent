from __future__ import annotations

import logging
import threading
import time

from sqlmodel import Session

from app.core.config import get_settings
from app.core.database import engine, ensure_database_ready
from app.services.knowledge_service import knowledge_service
from app.services.report_queue_service import report_queue_service
from app.services.report_service import report_service


logger = logging.getLogger(__name__)
_embedded_worker_lock = threading.Lock()
_embedded_worker_thread: threading.Thread | None = None


def run_worker() -> None:
    """持续消费报告解析队列。"""
    settings = get_settings()
    ensure_database_ready()
    with Session(engine) as session:
      knowledge_service.ensure_initialized(session)

    logger.info("Report worker started.")
    while True:
        task_id: str | None = None
        report_id: str | None = None
        try:
            with Session(engine) as session:
                task = report_queue_service.claim_next_task(session)
                if task is None:
                    time.sleep(max(settings.report_queue_poll_interval_seconds, 0.2))
                    continue
                task_id = task.id
                report_id = task.report_id

            if report_id is None:
                continue

            report_service.process_report(report_id)

            with Session(engine) as session:
                report_queue_service.mark_succeeded(session, task_id)
            logger.info("Report task succeeded: report_id=%s task_id=%s", report_id, task_id)
        except Exception as exc:
            logger.exception("Report worker failed: report_id=%s task_id=%s", report_id, task_id)
            if task_id:
                with Session(engine) as session:
                    report_queue_service.mark_failed(session, task_id, str(exc))
            time.sleep(max(settings.report_queue_poll_interval_seconds, 0.2))


def start_embedded_worker() -> threading.Thread:
    """在 API 进程内启动一个守护线程 worker，避免开发环境只开后端时队列无人消费。"""
    global _embedded_worker_thread
    with _embedded_worker_lock:
        if _embedded_worker_thread and _embedded_worker_thread.is_alive():
            return _embedded_worker_thread

        thread = threading.Thread(
            target=run_worker,
            name="report-queue-worker",
            daemon=True,
        )
        thread.start()
        _embedded_worker_thread = thread
        logger.info("Embedded report worker started.")
        return thread


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_worker()
