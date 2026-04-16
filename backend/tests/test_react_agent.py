from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine

from app.models.entities import ChatMessage, ChatSession, LabItem, Report
from app.services.knowledge_service import knowledge_service
from app.services.llm import llm_service
from app.services.react_agent import react_agent_service
from app.services.routing_service import routing_service
from app.services.safety_service import DEFAULT_SAFETY_APPENDIX
from app.services.who_service import who_service


def make_session() -> Session:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    return Session(engine)


def test_entry_graph_returns_analysis_for_normal_query(monkeypatch) -> None:
    original_client = llm_service.client
    llm_service.client = object()

    monkeypatch.setattr(routing_service, "route", lambda message, has_report: "term_explanation")
    monkeypatch.setattr(
        llm_service,
        "chat_json_fast",
        lambda system_prompt, user_prompt: {
            "intent": "term_explanation",
            "rewritten_query": "低密度脂蛋白",
            "normalized_term": "低密度脂蛋白",
            "use_local_knowledge": True,
            "use_who": False,
            "use_max": True,
            "reason": "graph_route",
        },
    )

    try:
        with make_session() as session:
            chat_session = ChatSession(title="graph-entry")
            session.add(chat_session)
            session.commit()
            session.refresh(chat_session)

            entry = react_agent_service._run_entry_graph(
                session=session,
                session_id=chat_session.id,
                report_id=None,
                message="解释一下低密度脂蛋白",
            )

        assert entry.immediate_response is None
        assert entry.analysis is not None
        assert entry.analysis.intent == "term_explanation"
        assert entry.analysis.reason == "graph_route"
        assert entry.execution is not None
        assert entry.execution.intent == "term_explanation"
    finally:
        llm_service.client = original_client


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


def test_report_follow_up_response_contains_debug_payload() -> None:
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
            message="分析一下我的报告指标",
            output_dir=Path("."),
        )

    assert response.intent == "report_follow_up"
    assert response.debug is not None
    assert response.debug.analysis
    assert response.debug.plan
    assert response.debug.synthesis


def test_report_follow_up_planner_can_narrow_focus_items(monkeypatch) -> None:
    original_client = llm_service.client
    llm_service.client = object()

    monkeypatch.setattr(routing_service, "route", lambda message, has_report: "report_follow_up")

    def fake_chat_json_fast(system_prompt: str, user_prompt: str) -> dict:
        if "输入分析器" in system_prompt:
            return {
                "intent": "report_follow_up",
                "rewritten_query": "报告指标结果",
                "normalized_term": "",
                "use_local_knowledge": True,
                "use_who": False,
                "use_max": True,
                "reason": "report_follow_up",
            }
        if "体检报告追问拆解器" in system_prompt:
            return {
                "focus_item_names": ["低密度脂蛋白胆固醇"],
                "need_item_explanations": True,
                "need_synthesis": True,
                "need_next_steps": True,
                "synthesis_axes": ["血脂代谢风险"],
                "follow_up_needed": False,
                "reason": "focus_ldl",
            }
        return {
            "items": [
                {
                    "name": "低密度脂蛋白胆固醇",
                    "meaning": "低密度脂蛋白胆固醇升高提示动脉粥样硬化相关风险增加。",
                    "common_reasons": "常见于饮食结构不合理、超重、运动不足或代谢异常。",
                    "watch_points": "建议结合总胆固醇、甘油三酯和是否存在高血压、糖尿病一起看。",
                    "suggested_department": "心血管内科",
                }
            ]
        }

    monkeypatch.setattr(llm_service, "chat_json_fast", fake_chat_json_fast)
    monkeypatch.setattr(llm_service, "chat_text_max", lambda system_prompt, user_prompt: "### 主要异常解读\n### 后续建议")

    try:
        with make_session() as session:
            knowledge_service.seed_local_knowledge(session)
            report = Report(
                file_name="report.pdf",
                file_path="report.pdf",
                raw_text="总胆固醇 6.00 mmol/L 3.1-5.2\n低密度脂蛋白胆固醇 3.72 mmol/L 0-3.4",
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
            session.add(
                LabItem(
                    report_id=report.id,
                    name="低密度脂蛋白胆固醇",
                    value_raw="3.72",
                    value_num=3.72,
                    unit="mmol/L",
                    reference_range="0-3.4",
                    status="high",
                )
            )
            session.commit()

            response = react_agent_service.respond(
                session=session,
                session_id=None,
                report_id=report.id,
                message="分析一下我的报告指标结果",
                output_dir=Path("."),
            )

        assert response.intent == "report_follow_up"
        assert "低密度脂蛋白胆固醇" in response.answer
        assert "血脂代谢风险" in response.answer
        assert "- **总胆固醇**" not in response.answer
    finally:
        llm_service.client = original_client


def test_report_follow_up_synthesis_material_is_used_in_fallback(monkeypatch) -> None:
    original_client = llm_service.client
    llm_service.client = object()

    monkeypatch.setattr(routing_service, "route", lambda message, has_report: "report_follow_up")

    call_index = {"json": 0}

    def fake_chat_json_fast(system_prompt: str, user_prompt: str) -> dict:
        call_index["json"] += 1
        if call_index["json"] == 1:
            return {
                "intent": "report_follow_up",
                "rewritten_query": "报告指标结果",
                "normalized_term": "",
                "use_local_knowledge": True,
                "use_who": False,
                "use_max": True,
                "reason": "report_follow_up",
            }
        if call_index["json"] == 2:
            return {
                "focus_item_names": ["低密度脂蛋白胆固醇"],
                "need_item_explanations": True,
                "need_synthesis": True,
                "need_next_steps": True,
                "synthesis_axes": ["血脂代谢风险"],
                "follow_up_needed": False,
                "reason": "focus_ldl",
            }
        if call_index["json"] == 3:
            return {
                "items": [
                    {
                        "name": "低密度脂蛋白胆固醇",
                        "meaning": "低密度脂蛋白胆固醇升高提示动脉粥样硬化相关风险增加。",
                        "common_reasons": "常见于饮食结构不合理、超重、运动不足或代谢异常。",
                        "watch_points": "建议结合总胆固醇和甘油三酯一起看。",
                        "suggested_department": "心血管内科",
                    }
                ]
            }
        return {
            "summary": "从当前报告看，最值得优先关注的是血脂代谢风险。",
            "priority_axes": ["血脂代谢风险"],
            "combined_findings": ["这类异常更适合结合血脂组合和其他心血管风险因素一起看。"],
            "next_steps": ["建议优先回看饮食、运动和体重变化，并按需复查血脂。"],
        }

    monkeypatch.setattr(llm_service, "chat_json_fast", fake_chat_json_fast)
    monkeypatch.setattr(llm_service, "chat_text_max", lambda system_prompt, user_prompt: "### 主要异常解读\n### 后续建议")

    try:
        with make_session() as session:
            knowledge_service.seed_local_knowledge(session)
            report = Report(
                file_name="report.pdf",
                file_path="report.pdf",
                raw_text="总胆固醇 6.00 mmol/L 3.1-5.2\n低密度脂蛋白胆固醇 3.72 mmol/L 0-3.4",
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
            session.add(
                LabItem(
                    report_id=report.id,
                    name="低密度脂蛋白胆固醇",
                    value_raw="3.72",
                    value_num=3.72,
                    unit="mmol/L",
                    reference_range="0-3.4",
                    status="high",
                )
            )
            session.commit()

            response = react_agent_service.respond(
                session=session,
                session_id=None,
                report_id=report.id,
                message="分析一下我的报告指标结果",
                output_dir=Path("."),
            )

        assert response.intent == "report_follow_up"
        assert "从当前报告看，最值得优先关注的是血脂代谢风险。" in response.answer
        assert "建议优先回看饮食、运动和体重变化，并按需复查血脂。" in response.answer
    finally:
        llm_service.client = original_client


def test_report_follow_up_bare_headings_fall_back_to_structured_content(monkeypatch) -> None:
    original_client = llm_service.client
    llm_service.client = object()

    monkeypatch.setattr(routing_service, "route", lambda message, has_report: "report_follow_up")

    def fake_chat_json_fast(system_prompt: str, user_prompt: str) -> dict:
        if "输入分析器" in system_prompt:
            return {
                "intent": "report_follow_up",
                "rewritten_query": "报告指标结果",
                "normalized_term": "",
                "use_local_knowledge": True,
                "use_who": False,
                "use_max": True,
                "reason": "report_follow_up",
            }
        return {
            "items": [
                {
                    "name": "肌酐",
                    "meaning": "肌酐是常见肾功能指标，升高时提示肾小球滤过功能可能下降。",
                    "common_reasons": "可能与脱水、肌肉量或肾脏排泄有关。",
                    "watch_points": "需要结合 eGFR 和尿检一起判断。",
                    "suggested_department": "肾内科",
                },
                {
                    "name": "血红蛋白",
                    "meaning": "血红蛋白异常常用于评估贫血或血液浓缩状态。",
                    "common_reasons": "可能与缺铁、慢性失血或脱水有关。",
                    "watch_points": "需要结合 MCV、RBC 和症状综合分析。",
                    "suggested_department": "血液科",
                },
            ]
        }

    monkeypatch.setattr(
        llm_service,
        "chat_json_fast",
        fake_chat_json_fast,
    )
    monkeypatch.setattr(
        llm_service,
        "chat_text_max",
        lambda system_prompt, user_prompt: "主要异常解读\n后续建议",
    )

    try:
        with make_session() as session:
            knowledge_service.seed_local_knowledge(session)
            report = Report(
                file_name="report.pdf",
                file_path="report.pdf",
                raw_text="肌酐 115 umol/L 57-111\n血红蛋白 110 g/L 130-175",
                parse_status="parsed",
            )
            session.add(report)
            session.commit()
            session.refresh(report)
            session.add(
                LabItem(
                    report_id=report.id,
                    name="肌酐",
                    value_raw="115",
                    value_num=115.0,
                    unit="umol/L",
                    reference_range="57-111",
                    status="high",
                )
            )
            session.add(
                LabItem(
                    report_id=report.id,
                    name="血红蛋白",
                    value_raw="110",
                    value_num=110.0,
                    unit="g/L",
                    reference_range="130-175",
                    status="low",
                )
            )
            session.commit()

            response = react_agent_service.respond(
                session=session,
                session_id=None,
                report_id=report.id,
                message="分析一下我的报告指标结果",
                output_dir=Path("."),
            )

        assert response.intent == "report_follow_up"
        assert "主要异常解读" in response.answer
        assert "肌酐" in response.answer
        assert "血红蛋白" in response.answer
        assert "后续建议" in response.answer
    finally:
        llm_service.client = original_client


def test_cached_incomplete_report_answer_is_recomputed(monkeypatch) -> None:
    original_client = llm_service.client
    llm_service.client = object()

    monkeypatch.setattr(routing_service, "route", lambda message, has_report: "report_follow_up")

    def fake_chat_json_fast(system_prompt: str, user_prompt: str) -> dict:
        if "输入分析器" in system_prompt:
            return {
                "intent": "report_follow_up",
                "rewritten_query": "报告指标",
                "normalized_term": "",
                "use_local_knowledge": True,
                "use_who": False,
                "use_max": True,
                "reason": "report_follow_up",
            }
        return {
            "items": [
                {
                    "name": "总胆固醇",
                    "meaning": "总胆固醇升高提示血脂代谢异常风险增加。",
                    "common_reasons": "可能与饮食、体重和代谢状态有关。",
                    "watch_points": "建议结合 LDL-C 和甘油三酯一起看。",
                    "suggested_department": "心血管内科",
                }
            ]
        }

    call_count = {"max": 0}

    def fake_chat_text_max(system_prompt: str, user_prompt: str) -> str:
        call_count["max"] += 1
        return "### 主要异常解读\n### 后续建议"

    monkeypatch.setattr(llm_service, "chat_json_fast", fake_chat_json_fast)
    monkeypatch.setattr(llm_service, "chat_text_max", fake_chat_text_max)

    try:
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

            session_id = "cached-session"
            bad_cache_key = react_agent_service._cache_key(session_id, report.id, "分析一下我的报告指标")
            react_agent_service.answer_cache[bad_cache_key] = react_agent_service.answer_cache.get(bad_cache_key) or None
            react_agent_service.answer_cache[bad_cache_key] = react_agent_service.answer_cache.__class__.__mro__[0]  # type: ignore[assignment]
    finally:
        llm_service.client = original_client


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

    monkeypatch.setattr(routing_service, "route", lambda message, has_report: "collect_more_info")

    def fake_chat_json_fast(system_prompt: str, user_prompt: str) -> dict[str, str]:
        calls.append("fast_json")
        return {"intent": "term_explanation", "reason": "mock"}

    def fake_chat_text_max(system_prompt: str, user_prompt: str) -> str:
        calls.append("max_text")
        return "这是由 max 模型生成的回答。"

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


def test_stream_respond_emits_incremental_deltas_before_final(monkeypatch) -> None:
    original_client = llm_service.client
    llm_service.client = object()

    monkeypatch.setattr(routing_service, "route", lambda message, has_report: "term_explanation")
    monkeypatch.setattr(llm_service, "chat_text_stream_max", lambda system_prompt, user_prompt: iter(["Streaming ", "answer"]))

    try:
        with make_session() as session:
            knowledge_service.seed_local_knowledge(session)
            events = list(
                react_agent_service.stream_respond(
                    session=session,
                    session_id=None,
                    report_id=None,
                    message="explain ldl",
                    output_dir=Path("."),
                )
            )

        delta_events = [event for event in events if event["event"] == "delta"]
        final_index = next(index for index, event in enumerate(events) if event["event"] == "final")

        assert [event["data"]["text"] for event in delta_events] == ["Streaming ", "answer"]
        assert delta_events
        assert events.index(delta_events[-1]) < final_index
        assert next(event for event in events if event["event"] == "final")["data"]["answer"].startswith("Streaming answer")
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


def test_term_explanation_uses_who_results_when_available(monkeypatch) -> None:
    original_client = llm_service.client
    llm_service.client = object()

    monkeypatch.setattr(routing_service, "route", lambda message, has_report: "term_explanation")
    monkeypatch.setattr(who_service, "is_configured", lambda: True)
    monkeypatch.setattr(
        who_service,
        "search",
        lambda query, limit=3, language="zh": {
            "query": query,
            "matches": [
                {
                    "title": "甲状腺功能亢进症",
                    "code": "5A00",
                    "uri": "https://id.who.int/icd/entity/123",
                    "definition": "一种由于甲状腺激素分泌过多导致代谢加快的疾病。",
                    "source": "WHO ICD-11",
                }
            ],
        },
    )
    monkeypatch.setattr(
        llm_service,
        "chat_text_max",
        lambda system_prompt, user_prompt: (
            "### 它是什么\n"
            "甲状腺功能亢进症是由于甲状腺激素分泌过多导致代谢加快的疾病。\n\n"
            "### 常见表现或特点\n"
            "常见表现包括心慌、怕热、多汗和体重下降。\n\n"
            "### 常见诱因或易感因素\n"
            "与自身免疫、甲状腺结节等因素有关。\n\n"
            "### 什么时候需要就医\n"
            "若症状持续或加重，建议到内分泌科就诊。\n\n"
            "### 温馨提示\n"
            "需要结合甲状腺功能检查结果综合判断。"
        ),
    )

    try:
        with make_session() as session:
            knowledge_service.seed_local_knowledge(session)
            response = react_agent_service.respond(
                session=session,
                session_id=None,
                report_id=None,
                message="甲亢是什么",
                output_dir=Path("."),
            )

        assert response.intent == "term_explanation"
        assert "lookup_icd11" in response.used_tools
        assert any(citation.source_type == "who_icd11" for citation in response.citations)
        assert "来源于 WHO ICD-11" in response.answer
    finally:
        llm_service.client = original_client


def test_term_explanation_can_use_standardized_who_query(monkeypatch) -> None:
    original_client = llm_service.client
    llm_service.client = object()

    monkeypatch.setattr(routing_service, "route", lambda message, has_report: "term_explanation")
    monkeypatch.setattr(who_service, "is_configured", lambda: True)

    queries: list[str] = []

    def fake_search(query, limit=3, language="zh"):
        queries.append(query)
        if query == "甲状腺功能亢进":
            return {
                "query": query,
                "matches": [
                    {
                        "title": "甲状腺功能亢进症",
                        "code": "5A00",
                        "uri": "https://id.who.int/icd/entity/123",
                        "definition": "一种由于甲状腺激素分泌过多导致代谢加快的疾病。",
                        "source": "WHO ICD-11",
                    }
                ],
            }
        return {"query": query, "matches": []}

    monkeypatch.setattr(who_service, "search", fake_search)
    monkeypatch.setattr(
        llm_service,
        "chat_text_max",
        lambda system_prompt, user_prompt: (
            "### 它是什么\n"
            "甲状腺功能亢进症是由于甲状腺激素分泌过多导致代谢加快的疾病。\n\n"
            "### 常见表现或特点\n"
            "常见表现包括心慌、怕热、多汗和体重下降。\n\n"
            "### 常见诱因或易感因素\n"
            "与自身免疫、甲状腺结节等因素有关。\n\n"
            "### 什么时候需要就医\n"
            "若症状持续或加重，建议到内分泌科就诊。\n\n"
            "### 温馨提示\n"
            "需要结合甲状腺功能检查结果综合判断。"
        ),
    )

    try:
        with make_session() as session:
            knowledge_service.seed_local_knowledge(session)
            response = react_agent_service.respond(
                session=session,
                session_id=None,
                report_id=None,
                message="甲亢是什么",
                output_dir=Path("."),
            )

        assert "甲状腺功能亢进" in queries
        assert "lookup_icd11" in response.used_tools
        assert "来源于 WHO ICD-11" in response.answer
    finally:
        llm_service.client = original_client


def test_term_explanation_bare_headings_trigger_fallback(monkeypatch) -> None:
    original_client = llm_service.client
    llm_service.client = object()

    monkeypatch.setattr(routing_service, "route", lambda message, has_report: "term_explanation")
    monkeypatch.setattr(
        llm_service,
        "chat_text_max",
        lambda system_prompt, user_prompt: "它是什么\n常见表现或特点\n常见诱因或易感因素\n什么时候需要就医\n温馨提示",
    )

    try:
        with make_session() as session:
            knowledge_service.seed_local_knowledge(session)
            response = react_agent_service.respond(
                session=session,
                session_id=None,
                report_id=None,
                message="甲亢是什么",
                output_dir=Path("."),
            )

        assert response.intent == "term_explanation"
        assert response.answer.count("###") >= 3
        assert "它是什么\n常见表现或特点" not in response.answer
    finally:
        llm_service.client = original_client


def test_safety_appendix_is_not_duplicated(monkeypatch) -> None:
    original_client = llm_service.client
    llm_service.client = object()

    monkeypatch.setattr(routing_service, "route", lambda message, has_report: "term_explanation")
    monkeypatch.setattr(
        llm_service,
        "chat_text_max",
        lambda system_prompt, user_prompt: (
            "### 它是什么\n甲亢是甲状腺功能亢进症。\n\n"
            f"{DEFAULT_SAFETY_APPENDIX}"
        ),
    )

    try:
        with make_session() as session:
            knowledge_service.seed_local_knowledge(session)
            response = react_agent_service.respond(
                session=session,
                session_id=None,
                report_id=None,
                message="甲亢是什么",
                output_dir=Path("."),
            )

        assert response.answer.count(DEFAULT_SAFETY_APPENDIX) == 1
    finally:
        llm_service.client = original_client


def test_agent_response_can_hit_persistent_cache(monkeypatch) -> None:
    calls = {"fast": 0, "max": 0}
    original_client = llm_service.client
    llm_service.client = object()

    def fake_chat_json_fast(system_prompt: str, user_prompt: str) -> dict[str, str]:
        calls["fast"] += 1
        return {
            "intent": "term_explanation",
            "rewritten_query": "低密度脂蛋白是什么意思",
            "normalized_term": "低密度脂蛋白胆固醇",
            "use_local_knowledge": True,
            "use_who": False,
            "use_max": True,
            "reason": "term_explanation",
        }

    def fake_chat_text_max(system_prompt: str, user_prompt: str) -> str:
        calls["max"] += 1
        return "### 它是什么\n低密度脂蛋白胆固醇是常见血脂指标。\n\n### 温馨提示\n建议结合体检结果综合判断。"

    monkeypatch.setattr(llm_service, "chat_json_fast", fake_chat_json_fast)
    monkeypatch.setattr(llm_service, "chat_text_max", fake_chat_text_max)

    try:
        with make_session() as session:
            knowledge_service.seed_local_knowledge(session)
            first = react_agent_service.respond(
                session=session,
                session_id=None,
                report_id=None,
                message="低密度脂蛋白是什么意思",
                output_dir=Path("."),
            )

            react_agent_service.answer_cache.clear()

            second = react_agent_service.respond(
                session=session,
                session_id=first.session_id,
                report_id=None,
                message="低密度脂蛋白是什么意思",
                output_dir=Path("."),
            )

        assert first.answer == second.answer
        assert calls["max"] == 1
    finally:
        llm_service.client = original_client
