from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlmodel import Session, select

from app.core.config import get_settings
from app.models.entities import Report, ReportParseTask


class ReportQueueService:
    """报告解析任务队列服务。

    当前实现是“数据库持久化队列 + 独立 worker 轮询消费”：
    - API 负责把任务写入表中
    - worker 负责抢占并执行任务
    - 任务状态和重试信息都落库
    """

    def __init__(self) -> None:
        self.settings = get_settings()

    def enqueue_report(self, session: Session, report_id: str) -> ReportParseTask:
        """为报告创建解析任务。

        如果这份报告已经存在未完成任务，就直接复用，避免重复入队。
        """
        existing = session.exec(
            select(ReportParseTask)
            .where(ReportParseTask.report_id == report_id)
            .where(ReportParseTask.status.in_(("queued", "running")))
            .order_by(ReportParseTask.created_at.desc())
        ).first()
        if existing:
            return existing

        task = ReportParseTask(report_id=report_id)
        session.add(task)

        report = session.get(Report, report_id)
        if report:
            report.parse_status = "queued"
            session.add(report)

        session.commit()
        session.refresh(task)
        return task

    def claim_next_task(self, session: Session) -> ReportParseTask | None:
        """抢占下一条可执行任务。"""
        now = datetime.now(timezone.utc)
        candidates = session.exec(
            select(ReportParseTask)
            .where(ReportParseTask.task_type == "report_parse")
            .where(
                (ReportParseTask.status == "queued")
                | (
                    (ReportParseTask.status == "running")
                    & (ReportParseTask.leased_until.is_not(None))
                    & (ReportParseTask.leased_until < now)
                )
            )
            .order_by(ReportParseTask.created_at.asc())
        ).all()
        for task in candidates:
            if task.attempts >= task.max_attempts:
                task.status = "failed"
                task.last_error = "Max attempts reached."
                task.updated_at = now
                session.add(task)
                continue

            task.status = "running"
            task.attempts += 1
            task.leased_until = now + timedelta(seconds=max(self.settings.report_queue_lease_seconds, 30))
            task.updated_at = now
            session.add(task)
            session.commit()
            session.refresh(task)
            return task

        session.commit()
        return None

    def mark_succeeded(self, session: Session, task_id: str) -> None:
        """标记任务执行成功。"""
        task = session.get(ReportParseTask, task_id)
        if not task:
            return
        task.status = "succeeded"
        task.leased_until = None
        task.last_error = None
        task.updated_at = datetime.now(timezone.utc)
        session.add(task)
        session.commit()

    def mark_failed(self, session: Session, task_id: str, error: str) -> None:
        """标记任务失败。

        失败后如果没有超过最大重试次数，就重新回到 queued；
        超过后才最终落到 failed。
        """
        task = session.get(ReportParseTask, task_id)
        if not task:
            return
        task.last_error = error
        task.leased_until = None
        task.updated_at = datetime.now(timezone.utc)
        task.status = "failed" if task.attempts >= task.max_attempts else "queued"
        session.add(task)
        session.commit()


report_queue_service = ReportQueueService()
