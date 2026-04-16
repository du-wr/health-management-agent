from __future__ import annotations

import json
from pathlib import Path

from sqlmodel import Session, select

from app.core.schemas import Citation, SessionDetail, SessionMessage, SessionReportInfo, SessionSummary
from app.models.entities import ChatMessage, ChatSession, Report, SummaryArtifact


class SessionService:
    """会话管理服务。

    这一层专门负责“多会话工作台”需要的能力：
    - 新建会话
    - 会话列表
    - 会话详情
    - 历史消息
    - 绑定报告
    - 删除会话
    - 自动标题
    """

    def create_session(self, session: Session, title: str | None = None) -> SessionSummary:
        """创建一个新的空会话。"""
        chat_session = ChatSession(title=(title or "新对话").strip() or "新对话")
        session.add(chat_session)
        session.commit()
        session.refresh(chat_session)
        return self._build_session_summary(session, chat_session)

    def list_sessions(self, session: Session) -> list[SessionSummary]:
        """返回左侧边栏需要的会话摘要列表。"""
        chat_sessions = session.exec(select(ChatSession)).all()
        summaries = [self._build_session_summary(session, chat_session) for chat_session in chat_sessions]
        summaries.sort(key=lambda item: item.last_message_at, reverse=True)
        return summaries

    def get_session_detail(self, session: Session, session_id: str) -> SessionDetail:
        """返回单个会话的概览信息。"""
        chat_session = self.get_session_entity(session, session_id)
        return self._build_session_detail(session, chat_session)

    def rename_session(self, session: Session, session_id: str, title: str) -> SessionDetail:
        """重命名会话标题。"""
        normalized = title.strip()
        if not normalized:
            raise ValueError("Session title cannot be empty.")
        chat_session = self.get_session_entity(session, session_id)
        chat_session.title = normalized
        session.add(chat_session)
        session.commit()
        session.refresh(chat_session)
        return self._build_session_detail(session, chat_session)

    def bind_report(self, session: Session, session_id: str, report_id: str) -> SessionDetail:
        """把一份报告绑定到指定会话。"""
        chat_session = self.get_session_entity(session, session_id)
        chat_session.report_id = report_id
        session.add(chat_session)
        session.commit()
        session.refresh(chat_session)
        return self._build_session_detail(session, chat_session)

    def delete_session(self, session: Session, session_id: str) -> None:
        """删除指定会话，并清理会话下的消息和健康小结文件。"""
        chat_session = self.get_session_entity(session, session_id)
        messages = session.exec(select(ChatMessage).where(ChatMessage.session_id == session_id)).all()
        artifacts = session.exec(select(SummaryArtifact).where(SummaryArtifact.session_id == session_id)).all()
        pdf_paths = [Path(item.pdf_path) for item in artifacts if item.pdf_path]

        # 先清数据库记录，避免旧消息和小结继续出现在历史视图里。
        for message in messages:
            session.delete(message)
        for artifact in artifacts:
            session.delete(artifact)
        session.delete(chat_session)
        session.commit()

        # PDF 属于会话派生文件，数据库删除后再尽力删除磁盘文件即可。
        for pdf_path in pdf_paths:
            try:
                pdf_path.unlink(missing_ok=True)
            except Exception:
                continue

    def list_messages(self, session: Session, session_id: str) -> list[SessionMessage]:
        """返回某个会话下的全部历史消息。"""
        self.get_session_entity(session, session_id)
        messages = session.exec(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.asc())
        ).all()
        return [self._build_message(message) for message in messages]

    def get_recent_messages(self, session: Session, session_id: str, limit: int = 8) -> list[SessionMessage]:
        """为健康小结提取最近几轮对话上下文。"""
        self.get_session_entity(session, session_id)
        messages = session.exec(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.desc())
            .limit(limit)
        ).all()
        messages.reverse()
        return [self._build_message(message) for message in messages]

    def get_session_entity(self, session: Session, session_id: str) -> ChatSession:
        """读取会话实体，不存在时抛出明确错误。"""
        chat_session = session.get(ChatSession, session_id)
        if not chat_session:
            raise ValueError("Session not found.")
        return chat_session

    def auto_title_if_needed(self, session: Session, session_id: str, message: str) -> None:
        """在首轮提问时，把默认标题自动改成问题摘要。"""
        chat_session = self.get_session_entity(session, session_id)
        if chat_session.title not in {"健康咨询", "新对话"}:
            return
        existing_count = len(session.exec(select(ChatMessage.id).where(ChatMessage.session_id == session_id)).all())
        if existing_count > 0:
            return
        normalized = message.strip()
        if not normalized:
            return
        chat_session.title = normalized[:20]
        session.add(chat_session)
        session.commit()

    def _build_session_summary(self, session: Session, chat_session: ChatSession) -> SessionSummary:
        """组装左侧边栏摘要。"""
        latest_message = self._latest_message(session, chat_session.id)
        message_count = self._message_count(session, chat_session.id)
        return SessionSummary(
            session_id=chat_session.id,
            title=chat_session.title,
            created_at=chat_session.created_at,
            last_message_at=latest_message.created_at if latest_message else chat_session.created_at,
            message_count=message_count,
            last_message_preview=latest_message.content[:48] if latest_message else "开始新的健康咨询",
            report=self._build_report_info(session, chat_session.report_id),
        )

    def _build_session_detail(self, session: Session, chat_session: ChatSession) -> SessionDetail:
        """组装会话详情。"""
        latest_message = self._latest_message(session, chat_session.id)
        return SessionDetail(
            session_id=chat_session.id,
            title=chat_session.title,
            created_at=chat_session.created_at,
            last_message_at=latest_message.created_at if latest_message else chat_session.created_at,
            message_count=self._message_count(session, chat_session.id),
            report=self._build_report_info(session, chat_session.report_id),
        )

    def _build_report_info(self, session: Session, report_id: str | None) -> SessionReportInfo | None:
        """为会话详情补充当前绑定报告的摘要信息。"""
        if not report_id:
            return None
        report = session.get(Report, report_id)
        if not report:
            return None
        return SessionReportInfo(
            report_id=report.id,
            file_name=report.file_name,
            parse_status=report.parse_status,
        )

    def _build_message(self, message: ChatMessage) -> SessionMessage:
        """把数据库消息转换成前端可直接消费的结构。"""
        try:
            raw_citations = json.loads(message.citations_json or "[]")
        except Exception:
            raw_citations = []
        citations: list[Citation] = []
        for item in raw_citations:
            try:
                citations.append(Citation.model_validate(item))
            except Exception:
                continue
        return SessionMessage(
            message_id=message.id,
            role=message.role,
            content=message.content,
            intent=message.intent,
            safety_level=message.safety_level,
            citations=citations,
            created_at=message.created_at,
        )

    def _latest_message(self, session: Session, session_id: str) -> ChatMessage | None:
        """读取某段会话的最新一条消息。"""
        return session.exec(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.desc())
            .limit(1)
        ).first()

    def _message_count(self, session: Session, session_id: str) -> int:
        """统计消息条数，用于边栏展示。"""
        return len(session.exec(select(ChatMessage.id).where(ChatMessage.session_id == session_id)).all())


session_service = SessionService()
