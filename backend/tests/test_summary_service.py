from datetime import datetime, timedelta, timezone

from sqlmodel import Session, SQLModel, create_engine

from app.models.entities import SummaryArtifact
from app.services.summary_service import summary_service


def make_session() -> Session:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    return Session(engine)


def test_list_for_session_returns_newest_first() -> None:
    with make_session() as session:
        now = datetime.now(timezone.utc)
        older = SummaryArtifact(
            report_id="report-1",
            session_id="session-1",
            markdown="# older",
            pdf_path="older.pdf",
            created_at=now - timedelta(minutes=10),
        )
        newer = SummaryArtifact(
            report_id="report-1",
            session_id="session-1",
            markdown="# newer",
            pdf_path="newer.pdf",
            created_at=now,
        )
        other_session = SummaryArtifact(
            report_id="report-2",
            session_id="session-2",
            markdown="# other",
            pdf_path="other.pdf",
            created_at=now + timedelta(minutes=5),
        )
        session.add(older)
        session.add(newer)
        session.add(other_session)
        session.commit()

        summaries = summary_service.list_for_session(session, "session-1")

        assert [item.markdown for item in summaries] == ["# newer", "# older"]
