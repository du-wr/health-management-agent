from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine, select

from app.models.entities import ChatMessage, ChatSession, SummaryArtifact
from app.services.session_service import session_service


def make_session() -> Session:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    return Session(engine)


def test_delete_session_removes_messages_and_summaries(tmp_path: Path) -> None:
    with make_session() as session:
        target_session = ChatSession(title="目标会话")
        preserved_session = ChatSession(title="保留会话")
        session.add(target_session)
        session.add(preserved_session)
        session.commit()
        session.refresh(target_session)
        session.refresh(preserved_session)

        session.add(ChatMessage(session_id=target_session.id, role="user", content="请解释这份报告"))
        session.add(ChatMessage(session_id=preserved_session.id, role="user", content="这是另一段对话"))
        pdf_path = tmp_path / "summary.pdf"
        pdf_path.write_text("# 健康小结", encoding="utf-8")
        session.add(
            SummaryArtifact(
                report_id="report-1",
                session_id=target_session.id,
                markdown="# 健康小结",
                pdf_path=str(pdf_path),
            )
        )
        session.commit()

        session_service.delete_session(session, target_session.id)

        assert session.get(ChatSession, target_session.id) is None
        assert session.get(ChatSession, preserved_session.id) is not None
        assert session.exec(select(ChatMessage).where(ChatMessage.session_id == target_session.id)).all() == []
        assert len(session.exec(select(ChatMessage).where(ChatMessage.session_id == preserved_session.id)).all()) == 1
        assert session.exec(select(SummaryArtifact).where(SummaryArtifact.session_id == target_session.id)).all() == []
        assert pdf_path.exists() is False
