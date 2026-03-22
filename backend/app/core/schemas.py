from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


IntentName = Literal[
    "report_follow_up",
    "term_explanation",
    "symptom_rag_advice",
    "collect_more_info",
    "safety_handoff",
]
TrustTier = Literal["A", "B", "C"]
SafetyLevel = Literal["safe", "caution", "handoff"]
LabStatus = Literal["high", "low", "normal", "unknown"]


class LabItem(BaseModel):
    name: str
    value_raw: str
    value_num: float | None = None
    unit: str = ""
    reference_range: str = ""
    status: LabStatus = "unknown"
    clinical_note: str | None = None


class ReportParseResult(BaseModel):
    report_id: str
    file_name: str
    items: list[LabItem]
    abnormal_items: list[LabItem]
    raw_text: str
    parse_warnings: list[str] = Field(default_factory=list)
    parse_status: str = "parsed"


class Citation(BaseModel):
    source_type: str
    doc_id: str
    title: str
    url: str
    trust_tier: TrustTier
    snippet: str


class KnowledgeDoc(BaseModel):
    doc_id: str
    title: str
    url: str
    source_domain: str
    source_org: str
    trust_tier: TrustTier
    content_type: str
    published_at: datetime | None = None
    snippet: str


class AgentDebug(BaseModel):
    analysis: dict[str, object] = Field(default_factory=dict)
    plan: dict[str, object] = Field(default_factory=dict)
    synthesis: dict[str, object] = Field(default_factory=dict)


class AgentResponse(BaseModel):
    session_id: str
    intent: IntentName
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    used_tools: list[str] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)
    safety_level: SafetyLevel = "safe"
    handoff_required: bool = False
    debug: AgentDebug | None = None


class SummaryArtifact(BaseModel):
    summary_id: str
    markdown: str
    pdf_path: str
    created_at: datetime


class ChatRequest(BaseModel):
    session_id: str | None = None
    report_id: str | None = None
    message: str


class SummaryRequest(BaseModel):
    session_id: str
    report_id: str


class KnowledgeSourcesResponse(BaseModel):
    total_docs: int
    trust_breakdown: dict[str, int]
    recent_docs: list[KnowledgeDoc]
