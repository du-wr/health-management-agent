import asyncio
from io import BytesIO
from pathlib import Path

from fastapi import UploadFile
from sqlmodel import Session, SQLModel, create_engine, select

from app.models.entities import ChatMessage, ChatSession, Report, ReportParseTask
from app.services.knowledge_service import knowledge_service
from app.services.react_agent import react_agent_service
from app.services.report_progress_service import report_progress_service
from app.services.report_queue_service import report_queue_service
from app.services.report_service import report_service


def make_session() -> Session:
    """为报告主链评测创建独立数据库。"""

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    return Session(engine)


def test_report_pipeline_success_can_auto_generate_analysis(tmp_path: Path, monkeypatch) -> None:
    """覆盖主成功链：上传 -> 入队 -> claim -> 解析完成 -> 自动报告解读。"""

    with make_session() as session:
        knowledge_service.seed_local_knowledge(session)

        upload = UploadFile(filename="report.pdf", file=BytesIO(b"dummy"))
        created = asyncio.run(report_service.create_upload(session, upload, tmp_path))
        queued_task = report_queue_service.enqueue_report(session, created.report_id)
        claimed_task = report_queue_service.claim_next_task(session)

        assert claimed_task is not None
        assert claimed_task.id == queued_task.id
        assert claimed_task.status == "running"

        monkeypatch.setattr(
            report_service,
            "extract_text",
            lambda path, report_id=None: ("总胆固醇 6.00 mmol/L 3.1-5.2", []),
        )

        report_service.process_report_with_session(session, created.report_id)
        report_queue_service.mark_succeeded(session, claimed_task.id)

        report = session.get(Report, created.report_id)
        task = session.get(ReportParseTask, claimed_task.id)
        progress = report_progress_service.get_state(created.report_id)

        assert report is not None
        assert report.parse_status == "parsed"
        assert task is not None
        assert task.status == "succeeded"
        assert progress is not None
        assert progress.done is True
        assert progress.parse_status == "parsed"

        chat_session = ChatSession(title="auto-analysis", report_id=created.report_id)
        session.add(chat_session)
        session.commit()
        session.refresh(chat_session)

        events = list(
            react_agent_service.stream_report_auto_analysis(
                session=session,
                session_id=chat_session.id,
                report_id=created.report_id,
                output_dir=Path("."),
            )
        )
        final_event = next(event for event in events if event["event"] == "final")
        messages = session.exec(select(ChatMessage).where(ChatMessage.session_id == chat_session.id)).all()

    assert final_event["data"]["intent"] == "report_follow_up"
    assert "总胆固醇" in final_event["data"]["answer"]
    assert [message.role for message in messages] == ["assistant"]


def test_report_pipeline_failure_marks_error_and_requeues(tmp_path: Path, monkeypatch) -> None:
    """覆盖失败边界：解析出错后，报告状态应失败且任务重新回到队列。"""

    with make_session() as session:
        upload = UploadFile(filename="report.pdf", file=BytesIO(b"dummy"))
        created = asyncio.run(report_service.create_upload(session, upload, tmp_path))
        queued_task = report_queue_service.enqueue_report(session, created.report_id)
        claimed_task = report_queue_service.claim_next_task(session)

        assert claimed_task is not None
        assert claimed_task.id == queued_task.id

        def fake_extract_text(path: Path, report_id: str | None = None):
            raise RuntimeError("mock parse failed")

        monkeypatch.setattr(report_service, "extract_text", fake_extract_text)

        report_service.process_report_with_session(session, created.report_id)
        report_queue_service.mark_failed(session, claimed_task.id, "mock parse failed")

        report = session.get(Report, created.report_id)
        task = session.get(ReportParseTask, claimed_task.id)
        progress = report_progress_service.get_state(created.report_id)

        assert report is not None
        assert report.parse_status == "error"
        assert task is not None
        assert task.status == "queued"
        assert task.last_error == "mock parse failed"
        assert progress is not None
        assert progress.done is True
        assert progress.parse_status == "error"
        assert progress.error == "mock parse failed"
