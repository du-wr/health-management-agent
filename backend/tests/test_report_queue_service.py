from sqlmodel import Session, SQLModel, create_engine

from app.models.entities import Report
from app.services.report_queue_service import report_queue_service


def make_session() -> Session:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    return Session(engine)


def test_enqueue_report_creates_single_active_task() -> None:
    with make_session() as session:
        report = Report(file_name="report.pdf", file_path="report.pdf", parse_status="queued")
        session.add(report)
        session.commit()
        session.refresh(report)

        first = report_queue_service.enqueue_report(session, report.id)
        second = report_queue_service.enqueue_report(session, report.id)

        assert first.id == second.id
        assert second.status == "queued"


def test_claim_and_finish_report_task() -> None:
    with make_session() as session:
        report = Report(file_name="report.pdf", file_path="report.pdf", parse_status="queued")
        session.add(report)
        session.commit()
        session.refresh(report)

        queued = report_queue_service.enqueue_report(session, report.id)
        claimed = report_queue_service.claim_next_task(session)

        assert claimed is not None
        assert claimed.id == queued.id
        assert claimed.status == "running"
        assert claimed.attempts == 1

        report_queue_service.mark_succeeded(session, claimed.id)
        finished = session.get(type(claimed), claimed.id)

        assert finished is not None
        assert finished.status == "succeeded"
