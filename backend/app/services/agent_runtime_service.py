from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlmodel import Session, select

from app.core.schemas import AgentGoalSummary, AgentRunDetail, AgentTaskRunSummary, AgentTraceEventRecord
from app.models.entities import AgentGoal, AgentTaskRun, AgentTraceEvent


GOAL_TYPE_BY_INTENT = {
    "report_follow_up": "report_monitoring",
    "term_explanation": "knowledge_learning",
    "symptom_rag_advice": "health_guidance",
    "collect_more_info": "clarification",
    "safety_handoff": "safety_escalation",
}

GOAL_TITLE_BY_TYPE = {
    "report_monitoring": "当前报告持续跟踪",
    "knowledge_learning": "健康知识理解",
    "health_guidance": "健康方向建议",
    "clarification": "信息补充与澄清",
    "safety_escalation": "安全风险处理",
}


@dataclass
class AgentRuntimeContext:
    """一次 Agent 运行在服务层的轻量上下文。"""

    run_id: str
    session_id: str
    report_id: str | None
    response_mode: str
    goal_id: str | None = None
    trace_sequence: int = 0
    cache_status: str = "miss"


class AgentRuntimeService:
    """负责记录 Agent 的目标、任务运行和轨迹。"""

    def start_run(
        self,
        session: Session,
        *,
        session_id: str,
        report_id: str | None,
        message: str,
        response_mode: str,
    ) -> AgentRuntimeContext:
        """创建一次新的运行记录。"""

        normalized_message = re.sub(r"\s+", " ", message.strip())
        run = AgentTaskRun(
            session_id=session_id,
            report_id=report_id,
            user_message=message,
            normalized_message=normalized_message,
            response_mode=response_mode,
        )
        session.add(run)
        session.commit()
        session.refresh(run)
        return AgentRuntimeContext(
            run_id=run.id,
            session_id=session_id,
            report_id=report_id,
            response_mode=response_mode,
        )

    def attach_goal(
        self,
        session: Session,
        runtime: AgentRuntimeContext,
        *,
        intent: str | None,
        message: str,
        report_id: str | None,
    ) -> AgentGoal | None:
        """根据当前 intent 把运行挂到一个长期目标上。"""

        if not intent:
            return None

        goal_type = GOAL_TYPE_BY_INTENT.get(intent, "general_tracking")
        goal = session.exec(
            select(AgentGoal)
            .where(AgentGoal.session_id == runtime.session_id)
            .where(AgentGoal.goal_type == goal_type)
            .order_by(AgentGoal.updated_at.desc())
        ).first()
        if goal is None:
            goal = AgentGoal(
                session_id=runtime.session_id,
                report_id=report_id,
                goal_type=goal_type,
                title=GOAL_TITLE_BY_TYPE.get(goal_type, "健康管理任务"),
                source_intent=intent,
                latest_user_message=message,
                summary_json=json.dumps(
                    {
                        "goal_type": goal_type,
                        "source_intent": intent,
                        "report_id": report_id,
                    },
                    ensure_ascii=False,
                ),
            )
        else:
            goal.report_id = report_id
            goal.source_intent = intent
            goal.latest_user_message = message
            goal.updated_at = self._utc_now()

        goal.last_run_id = runtime.run_id
        session.add(goal)
        session.commit()
        session.refresh(goal)

        run = session.get(AgentTaskRun, runtime.run_id)
        if run is not None:
            run.goal_id = goal.id
            run.intent = intent
            session.add(run)
            session.commit()

        runtime.goal_id = goal.id
        return goal

    def append_trace(
        self,
        session: Session,
        runtime: AgentRuntimeContext,
        *,
        phase: str,
        step_name: str,
        status: str = "completed",
        payload: dict[str, Any] | None = None,
    ) -> AgentTraceEventRecord:
        """追加一条运行轨迹事件。"""

        runtime.trace_sequence += 1
        event = AgentTraceEvent(
            run_id=runtime.run_id,
            sequence_no=runtime.trace_sequence,
            phase=phase,
            step_name=step_name,
            status=status,
            payload_json=json.dumps(self._jsonable(payload or {}), ensure_ascii=False),
        )
        session.add(event)
        session.commit()
        session.refresh(event)
        return self._build_trace_event_record(event)

    def update_cache_status(self, session: Session, runtime: AgentRuntimeContext, cache_status: str) -> None:
        """更新运行的缓存命中状态。"""

        runtime.cache_status = cache_status
        run = session.get(AgentTaskRun, runtime.run_id)
        if run is None:
            return
        run.cache_status = cache_status
        session.add(run)
        session.commit()

    def complete_run(
        self,
        session: Session,
        runtime: AgentRuntimeContext,
        *,
        intent: str | None,
        answer: str,
        used_tools: list[str],
        debug: dict[str, Any] | None = None,
        handoff_required: bool = False,
        status: str = "completed",
    ) -> None:
        """把一次运行标记为完成。"""

        run = session.get(AgentTaskRun, runtime.run_id)
        if run is None:
            return
        run.intent = intent
        run.status = status
        run.cache_status = runtime.cache_status
        run.handoff_required = handoff_required
        run.answer_excerpt = answer.strip()[:400]
        run.used_tools_json = json.dumps(used_tools, ensure_ascii=False)
        run.debug_json = json.dumps(self._jsonable(debug or {}), ensure_ascii=False)
        run.finished_at = self._utc_now()
        session.add(run)

        if runtime.goal_id:
            goal = session.get(AgentGoal, runtime.goal_id)
            if goal is not None:
                goal.status = "active" if status == "completed" else status
                goal.last_run_id = runtime.run_id
                goal.updated_at = self._utc_now()
                session.add(goal)

        session.commit()

    def fail_run(self, session: Session, runtime: AgentRuntimeContext, error_message: str) -> None:
        """记录失败状态。"""

        run = session.get(AgentTaskRun, runtime.run_id)
        if run is None:
            return
        run.status = "failed"
        run.error_message = error_message
        run.finished_at = self._utc_now()
        run.cache_status = runtime.cache_status
        session.add(run)
        session.commit()

    def build_debug_attachment(self, session: Session, runtime: AgentRuntimeContext, *, limit: int = 8) -> dict[str, Any]:
        """生成可直接挂到 AgentResponse.debug 的运行摘要。"""

        run = session.get(AgentTaskRun, runtime.run_id)
        if run is None:
            return {}
        goal = session.get(AgentGoal, runtime.goal_id) if runtime.goal_id else None
        trace_events = session.exec(
            select(AgentTraceEvent)
            .where(AgentTraceEvent.run_id == runtime.run_id)
            .order_by(AgentTraceEvent.sequence_no.asc())
        ).all()
        if limit > 0:
            trace_events = trace_events[-limit:]
        return {
            "goal": self._build_goal_summary(goal).model_dump(mode="json") if goal else {},
            "task_run": self._build_task_run_summary(run).model_dump(mode="json"),
            "trace_summary": [self._build_trace_event_record(item).model_dump(mode="json") for item in trace_events],
        }

    def list_session_runs(self, session: Session, session_id: str, *, limit: int = 20) -> list[AgentTaskRunSummary]:
        """列出某个会话最近的运行记录。"""

        runs = session.exec(
            select(AgentTaskRun)
            .where(AgentTaskRun.session_id == session_id)
            .order_by(AgentTaskRun.started_at.desc())
            .limit(limit)
        ).all()
        return [self._build_task_run_summary(run) for run in runs]

    def get_run_detail(self, session: Session, run_id: str) -> AgentRunDetail:
        """读取一次运行的完整详情。"""

        run = session.get(AgentTaskRun, run_id)
        if run is None:
            raise ValueError("Agent run not found.")
        goal = session.get(AgentGoal, run.goal_id) if run.goal_id else None
        trace_events = session.exec(
            select(AgentTraceEvent)
            .where(AgentTraceEvent.run_id == run_id)
            .order_by(AgentTraceEvent.sequence_no.asc())
        ).all()
        return AgentRunDetail(
            task_run=self._build_task_run_summary(run),
            goal=self._build_goal_summary(goal) if goal else None,
            trace_events=[self._build_trace_event_record(item) for item in trace_events],
        )

    def _build_goal_summary(self, goal: AgentGoal) -> AgentGoalSummary:
        return AgentGoalSummary(
            goal_id=goal.id,
            session_id=goal.session_id,
            report_id=goal.report_id,
            goal_type=goal.goal_type,
            title=goal.title,
            status=goal.status,
            source_intent=goal.source_intent,
            latest_user_message=goal.latest_user_message,
            last_run_id=goal.last_run_id,
            created_at=goal.created_at,
            updated_at=goal.updated_at,
        )

    def _build_task_run_summary(self, run: AgentTaskRun) -> AgentTaskRunSummary:
        try:
            used_tools = json.loads(run.used_tools_json or "[]")
        except Exception:
            used_tools = []
        return AgentTaskRunSummary(
            run_id=run.id,
            session_id=run.session_id,
            goal_id=run.goal_id,
            report_id=run.report_id,
            user_message=run.user_message,
            status=run.status,
            intent=run.intent,
            response_mode=run.response_mode,
            cache_status=run.cache_status,
            handoff_required=run.handoff_required,
            answer_excerpt=run.answer_excerpt,
            used_tools=used_tools if isinstance(used_tools, list) else [],
            started_at=run.started_at,
            finished_at=run.finished_at,
        )

    def _build_trace_event_record(self, event: AgentTraceEvent) -> AgentTraceEventRecord:
        try:
            payload = json.loads(event.payload_json or "{}")
        except Exception:
            payload = {}
        if not isinstance(payload, dict):
            payload = {"value": payload}
        return AgentTraceEventRecord(
            event_id=event.id,
            run_id=event.run_id,
            sequence_no=event.sequence_no,
            phase=event.phase,
            step_name=event.step_name,
            status=event.status,
            payload=payload,
            created_at=event.created_at,
        )

    def _jsonable(self, value: Any) -> Any:
        """把运行时对象尽量收敛成可序列化结构。"""

        if hasattr(value, "model_dump"):
            return self._jsonable(value.model_dump(mode="json"))
        if isinstance(value, dict):
            return {str(key): self._jsonable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._jsonable(item) for item in value]
        if isinstance(value, tuple):
            return [self._jsonable(item) for item in value]
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _utc_now(self) -> datetime:
        return datetime.now(timezone.utc)


agent_runtime_service = AgentRuntimeService()
