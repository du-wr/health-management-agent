from pathlib import Path

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from app.core.schemas import AgentResponse
from app.models.entities import ChatMessage, ChatSession, LabItem, Report
from app.services.agent_eval_service import AgentEvalCase, agent_eval_service
from app.services.agent_memory_service import agent_memory_service
from app.services.knowledge_service import knowledge_service
from app.services.react_agent import react_agent_service


CASES_PATH = Path(__file__).parent / "evals" / "agent_eval_cases.json"
ALL_EVAL_CASES = agent_eval_service.load_cases(CASES_PATH)


def make_session() -> Session:
    """为评测集提供独立数据库，避免不同 case 之间互相污染。"""

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    return Session(engine)


def _seed_report_context(session: Session, case: AgentEvalCase) -> tuple[ChatSession | None, Report | None]:
    """按 case 内容补齐报告、会话和历史消息。"""

    report: Report | None = None
    if case.has_report:
        report = Report(
            file_name=case.report_file_name,
            file_path=case.report_file_name,
            raw_text=case.report_raw_text,
            parse_status=case.report_parse_status,
        )
        session.add(report)
        session.commit()
        session.refresh(report)

        for item in case.report_items:
            session.add(
                LabItem(
                    report_id=report.id,
                    name=item.name,
                    value_raw=item.value_raw,
                    value_num=item.value_num,
                    unit=item.unit,
                    reference_range=item.reference_range,
                    status=item.status,
                )
            )
        session.commit()
        agent_memory_service.refresh_report_insight(session, report.id)

    chat_session: ChatSession | None = None
    if case.category in {"auto_report_analysis", "memory"} or case.history_messages:
        chat_session = ChatSession(title=case.name, report_id=report.id if report else None)
        session.add(chat_session)
        session.commit()
        session.refresh(chat_session)

    if chat_session and case.history_messages:
        for message in case.history_messages:
            session.add(
                ChatMessage(
                    session_id=chat_session.id,
                    role=message.role,
                    content=message.content,
                    intent=message.intent,
                    safety_level=message.safety_level,
                )
            )
        session.commit()
        agent_memory_service.refresh_session_memory(session, chat_session.id, latest_run_id="seed-memory-run")

    return chat_session, report


def _run_response_case(case: AgentEvalCase):
    """运行需要经过完整 Agent 主链的评测 case。"""

    react_agent_service.answer_cache.clear()
    with make_session() as session:
        knowledge_service.seed_local_knowledge(session)
        chat_session, report = _seed_report_context(session, case)

        if case.category == "auto_report_analysis":
            assert chat_session is not None
            assert report is not None
            events = list(
                react_agent_service.stream_report_auto_analysis(
                    session=session,
                    session_id=chat_session.id,
                    report_id=report.id,
                    output_dir=Path("."),
                )
            )
            final_event = next(event for event in events if event["event"] == "final")
            response = AgentResponse.model_validate(final_event["data"])

            # 自动报告解读不应额外伪造一条用户消息。
            messages = session.exec(select(ChatMessage).where(ChatMessage.session_id == chat_session.id)).all()
            assert [message.role for message in messages] == ["assistant"]
            return agent_eval_service.evaluate_response_case(case, response)

        response = react_agent_service.respond(
            session=session,
            session_id=chat_session.id if chat_session else None,
            report_id=report.id if report else None,
            message=case.message,
            output_dir=Path("."),
        )
        return agent_eval_service.evaluate_response_case(case, response)


@pytest.mark.parametrize(
    "case",
    [item for item in ALL_EVAL_CASES if item.category == "routing"],
    ids=lambda item: item.case_id,
)
def test_agent_eval_routing_cases(case: AgentEvalCase) -> None:
    result = agent_eval_service.evaluate_routing_case(case)
    assert result.passed, result.failures


@pytest.mark.parametrize(
    "case",
    [item for item in ALL_EVAL_CASES if item.category == "safety"],
    ids=lambda item: item.case_id,
)
def test_agent_eval_safety_cases(case: AgentEvalCase) -> None:
    result = agent_eval_service.evaluate_safety_case(case)
    assert result.passed, result.failures


@pytest.mark.parametrize(
    "case",
    [item for item in ALL_EVAL_CASES if item.category in {"report_follow_up", "auto_report_analysis", "memory"}],
    ids=lambda item: item.case_id,
)
def test_agent_eval_response_cases(case: AgentEvalCase) -> None:
    result = _run_response_case(case)
    assert result.passed, result.failures


def test_agent_eval_suite_summary_is_all_green() -> None:
    """汇总检查，确保当前最小回归集整体通过。"""

    results = []
    for case in ALL_EVAL_CASES:
        if case.category == "routing":
            results.append(agent_eval_service.evaluate_routing_case(case))
            continue
        if case.category == "safety":
            results.append(agent_eval_service.evaluate_safety_case(case))
            continue
        results.append(_run_response_case(case))

    summary = agent_eval_service.summarize(results)
    assert summary.failed == 0, [result.model_dump(mode="json") for result in summary.results if not result.passed]
    assert summary.category_summary["routing"]["passed"] >= 2
    assert summary.category_summary["safety"]["passed"] >= 4
    assert summary.category_summary["report_follow_up"]["passed"] >= 3
