from sqlmodel import Session, SQLModel, create_engine

from app.models.entities import ChatSession, LabItem, Report
from app.services.report_tool_service import report_tool_service
from app.services.session_service import session_service


def make_session() -> Session:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    return Session(engine)


def test_compare_report_trends_uses_previous_session_report() -> None:
    with make_session() as session:
        chat_session = ChatSession(title="trend")
        previous_report = Report(file_name="previous.pdf", file_path="previous.pdf", parse_status="parsed")
        current_report = Report(file_name="current.pdf", file_path="current.pdf", parse_status="parsed")
        session.add(chat_session)
        session.add(previous_report)
        session.add(current_report)
        session.commit()
        session.refresh(chat_session)
        session.refresh(previous_report)
        session.refresh(current_report)

        session.add(
            LabItem(
                report_id=previous_report.id,
                name="低密度脂蛋白",
                value_raw="3.8",
                value_num=3.8,
                unit="mmol/L",
                reference_range="0-3.4",
                status="high",
            )
        )
        session.add(
            LabItem(
                report_id=current_report.id,
                name="LDL-C",
                value_raw="4.5",
                value_num=4.5,
                unit="mmol/L",
                reference_range="0-3.4",
                status="high",
            )
        )
        session.commit()

        session_service.bind_report(session, chat_session.id, previous_report.id)
        session_service.bind_report(session, chat_session.id, current_report.id)

        result = report_tool_service.compare_report_trends(
            session,
            chat_session.id,
            current_report.id,
            focus_item_names=["低密度脂蛋白"],
        )

    assert result.previous_report_id == previous_report.id
    assert len(result.comparisons) == 1
    assert result.comparisons[0].direction == "up"
    assert "上升" in result.comparisons[0].summary


def test_build_report_risk_flags_marks_high_priority_items() -> None:
    flags = report_tool_service.build_report_risk_flags(
        [
            {
                "name": "空腹血糖",
                "value_raw": "7.4",
                "value_num": 7.4,
                "unit": "mmol/L",
                "reference_range": "3.9-6.1",
                "status": "high",
            }
        ]
    )

    assert len(flags) == 1
    assert flags[0].level == "high_priority"
    assert "空腹血糖" in flags[0].reason
