from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlmodel import Session, select

from app.core.schemas import ReportInsightRecord, SessionMemoryRecord
from app.models.entities import ChatMessage, ReportInsight, SessionMemory
from app.services.report_tool_service import report_tool_service


class AgentMemoryService:
    """维护会话摘要记忆和报告长期洞察。"""

    def refresh_session_memory(self, session: Session, session_id: str, *, latest_run_id: str | None = None) -> SessionMemoryRecord:
        """根据最近消息刷新会话摘要记忆。"""

        messages = session.exec(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.asc())
        ).all()
        existing = session.get(SessionMemory, session_id)
        report_id = existing.report_id if existing else None
        if messages:
            latest_intent = next((item.intent for item in reversed(messages) if item.intent), None)
            user_messages = [item.content.strip() for item in messages if item.role == "user" and item.content.strip()]
            assistant_messages = [item.content.strip() for item in messages if item.role == "assistant" and item.content.strip()]
            focus_points = [item[:60] for item in user_messages[-3:]]
            summary_parts: list[str] = []
            if focus_points:
                summary_parts.append("近期用户重点关注：" + "；".join(focus_points))
            if assistant_messages:
                summary_parts.append("最近一次系统结论摘要：" + assistant_messages[-1][:120])
            summary_text = "\n".join(summary_parts).strip()
            record = existing or SessionMemory(session_id=session_id, report_id=report_id)
            record.latest_run_id = latest_run_id or record.latest_run_id
            record.summary_text = summary_text
            record.focus_points_json = json.dumps(focus_points, ensure_ascii=False)
            record.latest_intent = latest_intent
            record.message_count = len(messages)
            record.updated_at = self._utc_now()
            session.add(record)
            session.commit()
            session.refresh(record)
            return self._to_session_memory_record(record)

        record = existing or SessionMemory(session_id=session_id, report_id=report_id)
        record.latest_run_id = latest_run_id or record.latest_run_id
        record.summary_text = ""
        record.focus_points_json = "[]"
        record.latest_intent = None
        record.message_count = 0
        record.updated_at = self._utc_now()
        session.add(record)
        session.commit()
        session.refresh(record)
        return self._to_session_memory_record(record)

    def bind_session_report(self, session: Session, session_id: str, report_id: str | None) -> None:
        """把会话记忆和当前报告绑定起来。"""

        record = session.get(SessionMemory, session_id)
        if record is None:
            record = SessionMemory(session_id=session_id, report_id=report_id)
        else:
            record.report_id = report_id
            record.updated_at = self._utc_now()
        session.add(record)
        session.commit()

    def refresh_report_insight(self, session: Session, report_id: str) -> ReportInsightRecord:
        """根据当前报告结构化结果刷新长期洞察。"""

        from app.services.report_service import report_service

        report = report_service.get_report(session, report_id)
        abnormal_items = report.abnormal_items[:8]
        abnormal_payload = [item.model_dump(mode="json") for item in abnormal_items]
        normalized_items = report_tool_service.normalize_lab_items(abnormal_payload)
        risk_flags = report_tool_service.build_report_risk_flags(
            abnormal_payload,
            normalized_items=normalized_items,
        )

        abnormal_names = [item.display_name for item in normalized_items if item.display_name]
        key_findings = [
            f"{item.display_name}: {item.value_raw}{item.unit or ''}（参考范围：{item.reference_range or '未提供'}）"
            for item in normalized_items[:5]
        ]
        key_findings.extend(flag.reason for flag in risk_flags[:3])

        if abnormal_names:
            monitoring_summary = "当前需要重点跟踪的异常指标有：" + "、".join(abnormal_names[:5]) + "。"
            if risk_flags:
                monitoring_summary += " " + "；".join(flag.reason for flag in risk_flags[:2])
        elif report.parse_status == "needs_review":
            monitoring_summary = "报告可用信息有限，建议结合原始报告和人工复核继续判断。"
        else:
            monitoring_summary = "当前报告未识别出明确异常指标，建议结合历史复查继续跟踪。"

        record = session.get(ReportInsight, report_id) or ReportInsight(report_id=report_id)
        record.parse_status = report.parse_status
        record.abnormal_item_names_json = json.dumps(abnormal_names, ensure_ascii=False)
        record.key_findings_json = json.dumps(key_findings, ensure_ascii=False)
        record.monitoring_summary = monitoring_summary
        record.updated_at = self._utc_now()
        session.add(record)
        session.commit()
        session.refresh(record)
        return self._to_report_insight_record(record)

    def load_memories(self, session: Session, session_id: str, report_id: str | None) -> dict[str, Any]:
        """加载入口图和 planner 需要的长期记忆。"""

        session_memory = session.get(SessionMemory, session_id)
        report_insight = session.get(ReportInsight, report_id) if report_id else None
        return {
            "session_memory": self._to_session_memory_record(session_memory).model_dump(mode="json") if session_memory else {},
            "report_insight": self._to_report_insight_record(report_insight).model_dump(mode="json") if report_insight else {},
        }

    def _to_session_memory_record(self, record: SessionMemory | None) -> SessionMemoryRecord:
        if record is None:
            return SessionMemoryRecord(
                session_id="",
                report_id=None,
                latest_run_id=None,
                summary_text="",
                focus_points=[],
                latest_intent=None,
                message_count=0,
                updated_at=self._utc_now(),
            )
        try:
            focus_points = json.loads(record.focus_points_json or "[]")
        except Exception:
            focus_points = []
        if not isinstance(focus_points, list):
            focus_points = []
        return SessionMemoryRecord(
            session_id=record.session_id,
            report_id=record.report_id,
            latest_run_id=record.latest_run_id,
            summary_text=record.summary_text,
            focus_points=[str(item) for item in focus_points],
            latest_intent=record.latest_intent,
            message_count=record.message_count,
            updated_at=record.updated_at,
        )

    def _to_report_insight_record(self, record: ReportInsight | None) -> ReportInsightRecord:
        if record is None:
            return ReportInsightRecord(
                report_id="",
                parse_status="uploaded",
                abnormal_item_names=[],
                key_findings=[],
                monitoring_summary="",
                updated_at=self._utc_now(),
            )
        try:
            abnormal_item_names = json.loads(record.abnormal_item_names_json or "[]")
        except Exception:
            abnormal_item_names = []
        try:
            key_findings = json.loads(record.key_findings_json or "[]")
        except Exception:
            key_findings = []
        if not isinstance(abnormal_item_names, list):
            abnormal_item_names = []
        if not isinstance(key_findings, list):
            key_findings = []
        return ReportInsightRecord(
            report_id=record.report_id,
            parse_status=record.parse_status,
            abnormal_item_names=[str(item) for item in abnormal_item_names],
            key_findings=[str(item) for item in key_findings],
            monitoring_summary=record.monitoring_summary,
            updated_at=record.updated_at,
        )

    def _utc_now(self) -> datetime:
        return datetime.now(timezone.utc)


agent_memory_service = AgentMemoryService()
