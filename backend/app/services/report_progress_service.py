from __future__ import annotations

from dataclasses import asdict, dataclass
from threading import Lock
from typing import Any


@dataclass
class ReportProgressState:
    report_id: str
    stage: str
    label: str
    progress: int
    parse_status: str
    done: bool = False
    error: str | None = None
    version: int = 0

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("version", None)
        return payload


class ReportProgressService:
    def __init__(self) -> None:
        self._states: dict[str, ReportProgressState] = {}
        self._lock = Lock()

    def initialize(self, report_id: str, *, parse_status: str = "uploaded") -> None:
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
        self.update(
            report_id,
            stage="completed",
            label="报告解析完成",
            progress=100,
            parse_status=parse_status,
            done=True,
        )

    def mark_failed(self, report_id: str, *, error: str) -> None:
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
        with self._lock:
            state = self._states.get(report_id)
            if state is None:
                return None
            return ReportProgressState(**asdict(state))


report_progress_service = ReportProgressService()
