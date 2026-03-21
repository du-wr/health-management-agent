from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine

from app.models.entities import ChatMessage, ChatSession, LabItem, Report
from app.services.knowledge_service import knowledge_service
from app.services.llm import llm_service
from app.services.react_agent import react_agent_service
from app.services.routing_service import routing_service


def make_session() -> Session:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    return Session(engine)


def test_react_agent_report_follow_up_uses_fast_path_tools() -> None:
    with make_session() as session:
        knowledge_service.seed_local_knowledge(session)
        report = Report(
            file_name="report.pdf",
            file_path="report.pdf",
            raw_text="总胆固醇 6.00 mmol/L 3.1-5.2",
            parse_status="parsed",
        )
        session.add(report)
        session.commit()
        session.refresh(report)
        session.add(
            LabItem(
                report_id=report.id,
                name="总胆固醇",
                value_raw="6.00",
                value_num=6.0,
                unit="mmol/L",
                reference_range="3.1-5.2",
                status="high",
            )
        )
        session.commit()

        response = react_agent_service.respond(
            session=session,
            session_id=None,
            report_id=report.id,
            message="这些异常指标代表什么？",
            output_dir=Path("."),
        )

        assert response.intent == "report_follow_up"
        assert response.used_tools == ["search_report_items", "interpret_lab"]
        assert "总胆固醇" in response.answer


def test_react_agent_processing_report_returns_wait_message() -> None:
    with make_session() as session:
        knowledge_service.seed_local_knowledge(session)
        report = Report(file_name="report.pdf", file_path="report.pdf", parse_status="processing")
        session.add(report)
        session.commit()
        session.refresh(report)

        response = react_agent_service.respond(
            session=session,
            session_id=None,
            report_id=report.id,
            message="这些异常指标代表什么？",
            output_dir=Path("."),
        )

        assert response.intent == "collect_more_info"
        assert "后台解析中" in response.answer


def test_recent_conversation_context_keeps_latest_n_turns() -> None:
    with make_session() as session:
        chat_session = ChatSession(title="context")
        session.add(chat_session)
        session.commit()
        session.refresh(chat_session)

        messages = [
            ("user", "第一轮用户"),
            ("assistant", "第一轮助手"),
            ("user", "第二轮用户"),
            ("assistant", "第二轮助手"),
            ("user", "第三轮用户"),
            ("assistant", "第三轮助手"),
            ("user", "第四轮用户"),
        ]
        base_time = datetime.now(timezone.utc)
        for index, (role, content) in enumerate(messages):
            session.add(
                ChatMessage(
                    session_id=chat_session.id,
                    role=role,
                    content=content,
                    created_at=base_time + timedelta(seconds=index),
                )
            )
        session.commit()

        history = react_agent_service._recent_conversation_context(session, chat_session.id)

        assert history
        assert history[-1]["content"] == "第四轮用户"
        assert len(history) <= react_agent_service.settings.short_term_context_turns * 2 + 1


def test_react_agent_uses_fast_and_max_models(monkeypatch) -> None:
    calls: list[str] = []
    original_client = llm_service.client
    llm_service.client = object()

    def fake_route(message: str, has_report: bool) -> str:
        return "collect_more_info"

    def fake_chat_json_fast(system_prompt: str, user_prompt: str) -> dict[str, str]:
        calls.append("fast_json")
        return {"intent": "term_explanation", "reason": "mock"}

    def fake_chat_text_max(system_prompt: str, user_prompt: str) -> str:
        calls.append("max_text")
        return "这是由 max 模型生成的回答。"

    monkeypatch.setattr(routing_service, "route", fake_route)
    monkeypatch.setattr(llm_service, "chat_json_fast", fake_chat_json_fast)
    monkeypatch.setattr(llm_service, "chat_text_max", fake_chat_text_max)

    try:
        with make_session() as session:
            knowledge_service.seed_local_knowledge(session)
            response = react_agent_service.respond(
                session=session,
                session_id=None,
                report_id=None,
                message="解释一下低密度脂蛋白是什么意思",
                output_dir=Path("."),
            )

        assert response.intent == "term_explanation"
        assert "max 模型" in response.answer
        assert calls == ["fast_json", "max_text"]
    finally:
        llm_service.client = original_client


def test_term_explanation_incomplete_answer_falls_back(monkeypatch) -> None:
    original_client = llm_service.client
    llm_service.client = object()

    monkeypatch.setattr(routing_service, "route", lambda message, has_report: "term_explanation")
    monkeypatch.setattr(
        llm_service,
        "chat_text_max",
        lambda system_prompt, user_prompt: "灰指甲，医学上称为甲真菌病。\n\n常见致病真菌类型包括：\n易感因素包括：",
    )

    try:
        with make_session() as session:
            knowledge_service.seed_local_knowledge(session)
            response = react_agent_service.respond(
                session=session,
                session_id=None,
                report_id=None,
                message="灰指甲是什么病",
                output_dir=Path("."),
            )

        assert response.intent == "term_explanation"
        assert "常见致病真菌类型包括：" not in response.answer
        assert "### 它是什么" in response.answer
        assert "### 温馨提示" in response.answer
    finally:
        llm_service.client = original_client
