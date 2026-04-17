from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import Column, Text
from sqlalchemy.dialects import mysql
from sqlmodel import Field, SQLModel


LONG_TEXT_TYPE = Text().with_variant(mysql.LONGTEXT(), "mysql")


def utc_now() -> datetime:
    """统一生成 UTC 时间，避免各表时间来源不一致。"""
    return datetime.now(timezone.utc)


class Report(SQLModel, table=True):
    """一份上传报告的主表。"""

    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    file_name: str
    file_path: str
    raw_text: str = Field(default="", sa_column=Column(LONG_TEXT_TYPE, nullable=False, default=""))
    parse_status: str = "uploaded"
    parse_warnings_json: str = Field(default="[]", sa_column=Column(LONG_TEXT_TYPE, nullable=False, default="[]"))
    created_at: datetime = Field(default_factory=utc_now)


class ReportParseTask(SQLModel, table=True):
    """报告解析任务表。"""

    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    report_id: str = Field(index=True)
    task_type: str = "report_parse"
    status: str = Field(default="queued", index=True)
    attempts: int = 0
    max_attempts: int = 3
    leased_until: datetime | None = Field(default=None, index=True)
    last_error: str | None = Field(default=None, sa_column=Column(LONG_TEXT_TYPE, nullable=True))
    created_at: datetime = Field(default_factory=utc_now, index=True)
    updated_at: datetime = Field(default_factory=utc_now, index=True)


class AgentAnswerCache(SQLModel, table=True):
    """Agent 最终回答的持久化缓存。"""

    cache_key: str = Field(primary_key=True)
    report_id: str | None = Field(default=None, index=True)
    normalized_message: str = Field(default="", sa_column=Column(LONG_TEXT_TYPE, nullable=False, default=""))
    response_json: str = Field(sa_column=Column(LONG_TEXT_TYPE, nullable=False))
    answer_text: str = Field(sa_column=Column(LONG_TEXT_TYPE, nullable=False))
    created_at: datetime = Field(default_factory=utc_now, index=True)
    expires_at: datetime = Field(index=True)


class LabItem(SQLModel, table=True):
    """报告里的单个结构化指标。"""

    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    report_id: str = Field(index=True)
    name: str
    value_raw: str
    value_num: float | None = None
    unit: str = ""
    reference_range: str = ""
    status: str = "unknown"
    clinical_note: str | None = Field(default=None, sa_column=Column(LONG_TEXT_TYPE, nullable=True))


class ChatSession(SQLModel, table=True):
    """一段连续对话的会话对象。"""

    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    report_id: str | None = Field(default=None, index=True)
    title: str = "健康咨询"
    created_at: datetime = Field(default_factory=utc_now)


class ChatMessage(SQLModel, table=True):
    """会话中的一条消息。

    这里既保存用户消息，也保存助手消息。
    """

    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    session_id: str = Field(index=True)
    role: str
    content: str = Field(sa_column=Column(LONG_TEXT_TYPE, nullable=False))
    intent: str | None = None
    safety_level: str = "safe"
    citations_json: str = Field(default="[]", sa_column=Column(LONG_TEXT_TYPE, nullable=False, default="[]"))
    created_at: datetime = Field(default_factory=utc_now)


class KnowledgeDoc(SQLModel, table=True):
    """知识库文档表。

    即使当前知识来自本地种子，也统一保存成文档记录，
    这样检索、引用和统计逻辑都能复用。
    """

    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    url: str = Field(unique=True, index=True)
    title: str
    source_domain: str
    source_org: str = ""
    trust_tier: str = "C"
    content_type: str = "article"
    published_at: datetime | None = None
    snippet: str = Field(default="", sa_column=Column(LONG_TEXT_TYPE, nullable=False, default=""))
    body_text: str = Field(default="", sa_column=Column(LONG_TEXT_TYPE, nullable=False, default=""))
    content_hash: str = Field(index=True)
    crawl_status: str = "fetched"
    discovered_at: datetime = Field(default_factory=utc_now)
    crawled_at: datetime = Field(default_factory=utc_now)


class SummaryArtifact(SQLModel, table=True):
    """一份已生成的健康小结。"""

    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    report_id: str = Field(index=True)
    session_id: str = Field(index=True)
    markdown: str = Field(sa_column=Column(LONG_TEXT_TYPE, nullable=False))
    pdf_path: str
    created_at: datetime = Field(default_factory=utc_now)
