from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from sqlmodel import Field, SQLModel


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Report(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    file_name: str
    file_path: str
    raw_text: str = ""
    parse_status: str = "uploaded"
    parse_warnings_json: str = "[]"
    created_at: datetime = Field(default_factory=utc_now)


class LabItem(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    report_id: str = Field(index=True)
    name: str
    value_raw: str
    value_num: float | None = None
    unit: str = ""
    reference_range: str = ""
    status: str = "unknown"
    clinical_note: str | None = None


class ChatSession(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    report_id: str | None = Field(default=None, index=True)
    title: str = "健康咨询"
    created_at: datetime = Field(default_factory=utc_now)


class ChatMessage(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    session_id: str = Field(index=True)
    role: str
    content: str
    intent: str | None = None
    safety_level: str = "safe"
    citations_json: str = "[]"
    created_at: datetime = Field(default_factory=utc_now)


class KnowledgeDoc(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    url: str = Field(unique=True, index=True)
    title: str
    source_domain: str
    source_org: str = ""
    trust_tier: str = "C"
    content_type: str = "article"
    published_at: datetime | None = None
    snippet: str = ""
    body_text: str = ""
    content_hash: str = Field(index=True)
    crawl_status: str = "fetched"
    discovered_at: datetime = Field(default_factory=utc_now)
    crawled_at: datetime = Field(default_factory=utc_now)


class SummaryArtifact(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    report_id: str = Field(index=True)
    session_id: str = Field(index=True)
    markdown: str
    pdf_path: str
    created_at: datetime = Field(default_factory=utc_now)
