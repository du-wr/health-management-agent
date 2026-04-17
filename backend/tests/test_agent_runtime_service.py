from sqlmodel import Session, SQLModel, create_engine

from app.services.agent_runtime_service import agent_runtime_service


def make_session() -> Session:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    return Session(engine)


def test_agent_runtime_service_records_goal_run_and_trace() -> None:
    with make_session() as session:
        runtime = agent_runtime_service.start_run(
            session,
            session_id="session-1",
            report_id="report-1",
            message="帮我看看这份报告接下来怎么跟踪",
            response_mode="stream",
        )
        goal = agent_runtime_service.attach_goal(
            session,
            runtime,
            intent="report_follow_up",
            message="帮我看看这份报告接下来怎么跟踪",
            report_id="report-1",
        )
        assert goal is not None

        agent_runtime_service.append_trace(
            session,
            runtime,
            phase="entry",
            step_name="entry_graph_completed",
            payload={"intent": "report_follow_up"},
        )
        agent_runtime_service.complete_run(
            session,
            runtime,
            intent="report_follow_up",
            answer="先关注血脂和尿酸，再安排 3 个月复查。",
            used_tools=["search_report_items", "interpret_lab"],
            debug={"plan": {"focus": ["血脂", "尿酸"]}},
        )

        detail = agent_runtime_service.get_run_detail(session, runtime.run_id)

    assert detail.task_run.run_id == runtime.run_id
    assert detail.task_run.goal_id == goal.id
    assert detail.task_run.intent == "report_follow_up"
    assert detail.goal is not None
    assert detail.goal.goal_type == "report_monitoring"
    assert len(detail.trace_events) >= 1
    assert detail.trace_events[0].phase == "entry"
