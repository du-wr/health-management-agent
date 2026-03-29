from __future__ import annotations

from dataclasses import asdict, dataclass
from threading import Lock
from typing import Any


@dataclass
class ReportProgressState:
    """一份报告当前的解析进度快照。"""
    report_id: str
    stage: str
    label: str
    progress: int
    parse_status: str
    done: bool = False
    error: str | None = None
    version: int = 0

    def to_payload(self) -> dict[str, Any]:
        """转换成适合 SSE 发送的字典结构。"""
        payload = asdict(self)
        payload.pop("version", None)
        return payload


class ReportProgressService:
    """在内存里维护报告解析进度。

    这是 Demo 级实现，优点是简单直接；如果以后做正式产品，
    这部分通常会迁移到 Redis 或任务系统里。
    """

    def __init__(self) -> None:
        self._states: dict[str, ReportProgressState] = {}
        self._lock = Lock()

    def initialize(self, report_id: str, *, parse_status: str = "uploaded") -> None:
        """上传完成后，为这份报告创建初始进度状态。"""
        with self._lock:
            self._states[report_id] = ReportProgressState(
                report_id=report_id,
                stage="queued",
                label="上传完成，等待开始解析",
                progress=5,
                parse_status=parse_status,
                done=False,
                version=1,
            )

    def update(
        self,
        report_id: str,
        *,
        stage: str,
        label: str,
        progress: int,
        parse_status: str | None = None,
        done: bool = False,
        error: str | None = None,
    ) -> None:
        """更新进度状态。每次更新都会递增 version，便于 SSE 去重。"""
        with self._lock:
            current = self._states.get(report_id)
            version = 1 if current is None else current.version + 1
            self._states[report_id] = ReportProgressState(
                report_id=report_id,
                stage=stage,
                label=label,
                progress=max(0, min(progress, 100)),
                parse_status=parse_status or (current.parse_status if current else "processing"),
                done=done,
                error=error,
                version=version,
            )

    def mark_complete(self, report_id: str, *, parse_status: str) -> None:
        """标记报告解析完成。"""
        self.update(
            report_id,
            stage="completed",
            label="报告解析完成",
            progress=100,
            parse_status=parse_status,
            done=True,
        )

    def mark_failed(self, report_id: str, *, error: str) -> None:
        """标记报告解析失败。"""
        self.update(
            report_id,
            stage="failed",
            label="报告解析失败",
            progress=100,
            parse_status="error",
            done=True,
            error=error,
        )

    def get_state(self, report_id: str) -> ReportProgressState | None:
        """读取当前快照，并返回一份副本，避免外部误改内部状态。"""
        with self._lock:
            state = self._states.get(report_id)
            if state is None:
                return None
            return ReportProgressState(**asdict(state))


report_progress_service = ReportProgressService()
