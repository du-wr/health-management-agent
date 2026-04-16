from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# 这一层定义的是“接口之间交换的数据结构”，不是数据库表。
# 可以把它理解成前后端、服务之间共同遵守的契约。
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
    """单个体检/检验指标的标准化表示。"""
    name: str
    value_raw: str
    value_num: float | None = None
    unit: str = ""
    reference_range: str = ""
    status: LabStatus = "unknown"
    clinical_note: str | None = None


class ReportParseResult(BaseModel):
    """前端真正消费的一份报告解析结果。"""
    report_id: str
    file_name: str
    items: list[LabItem]
    abnormal_items: list[LabItem]
    raw_text: str
    parse_warnings: list[str] = Field(default_factory=list)
    parse_status: str = "parsed"


class Citation(BaseModel):
    """回答中的一条来源引用。"""
    source_type: str
    doc_id: str
    title: str
    url: str
    trust_tier: TrustTier
    snippet: str


class KnowledgeDoc(BaseModel):
    """对知识文档的简化表达，主要用于接口返回。"""
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
    """给前端调试面板展示的中间结果。"""
    analysis: dict[str, object] = Field(default_factory=dict)
    plan: dict[str, object] = Field(default_factory=dict)
    synthesis: dict[str, object] = Field(default_factory=dict)


class AgentResponse(BaseModel):
    """一次聊天回答的标准返回结构。"""
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
    """健康小结生成结果。"""
    summary_id: str
    markdown: str
    pdf_path: str
    created_at: datetime


class SessionReportInfo(BaseModel):
    """会话中当前绑定报告的简要信息。"""
    report_id: str
    file_name: str
    parse_status: str


class SessionSummary(BaseModel):
    """左侧会话列表需要的摘要信息。"""
    session_id: str
    title: str
    created_at: datetime
    last_message_at: datetime
    message_count: int
    last_message_preview: str
    report: SessionReportInfo | None = None


class SessionDetail(BaseModel):
    """单个会话的完整概览信息。"""
    session_id: str
    title: str
    created_at: datetime
    last_message_at: datetime
    message_count: int
    report: SessionReportInfo | None = None


class SessionMessage(BaseModel):
    """会话历史消息的接口表达。"""
    message_id: str
    role: str
    content: str
    intent: str | None = None
    safety_level: str = "safe"
    citations: list[Citation] = Field(default_factory=list)
    created_at: datetime


class ChatRequest(BaseModel):
    """聊天请求体。"""
    session_id: str | None = None
    report_id: str | None = None
    message: str


class SessionCreateRequest(BaseModel):
    """新建会话请求体。"""
    title: str | None = None


class SessionRenameRequest(BaseModel):
    """会话重命名请求体。"""
    title: str


class SummaryRequest(BaseModel):
    """小结生成请求体。"""
    session_id: str
    report_id: str


class KnowledgeSourcesResponse(BaseModel):
    """知识库统计接口的返回结构。"""
    total_docs: int
    trust_breakdown: dict[str, int]
    recent_docs: list[KnowledgeDoc]
