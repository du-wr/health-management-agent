from sqlmodel import Session, SQLModel, create_engine

from app.models.entities import ChatMessage, ChatSession, LabItem, Report
from app.services.agent_memory_service import agent_memory_service


def make_session() -> Session:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    return Session(engine)


def test_refresh_session_memory_builds_summary_from_messages() -> None:
    with make_session() as session:
        chat_session = ChatSession(title="memory")
        session.add(chat_session)
        session.commit()
        session.refresh(chat_session)
        session_id = chat_session.id

        session.add(ChatMessage(session_id=chat_session.id, role="user", content="我最近最担心血脂"))
        session.add(ChatMessage(session_id=chat_session.id, role="assistant", content="建议先关注 LDL 和总胆固醇。", intent="report_follow_up"))
        session.add(ChatMessage(session_id=chat_session.id, role="user", content="那我还需要复查吗"))
        session.commit()

        memory = agent_memory_service.refresh_session_memory(session, chat_session.id, latest_run_id="run-1")

    assert memory.session_id == session_id
    assert memory.latest_run_id == "run-1"
    assert memory.message_count == 3
    assert memory.latest_intent == "report_follow_up"
    assert any("血脂" in item for item in memory.focus_points)
    assert "近期用户重点关注" in memory.summary_text


def test_refresh_report_insight_builds_monitoring_summary() -> None:
    with make_session() as session:
        report = Report(
            file_name="report.pdf",
            file_path="report.pdf",
            raw_text="LDL 4.2 mmol/L 0-3.4",
            parse_status="parsed",
        )
        session.add(report)
        session.commit()
        session.refresh(report)
        report_id = report.id

        session.add(
            LabItem(
                report_id=report.id,
                name="低密度脂蛋白",
                value_raw="4.2",
                value_num=4.2,
                unit="mmol/L",
                reference_range="0-3.4",
                status="high",
            )
        )
        session.commit()

        insight = agent_memory_service.refresh_report_insight(session, report.id)

    assert insight.report_id == report_id
    assert insight.parse_status == "parsed"
    assert "低密度脂蛋白" in insight.abnormal_item_names
    assert "重点跟踪" in insight.monitoring_summary
