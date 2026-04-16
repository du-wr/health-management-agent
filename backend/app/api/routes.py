import json
import time
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from sqlmodel import Session, select

from app.core.config import get_settings
from app.core.database import engine, get_session
from app.core.schemas import (
    AgentResponse,
    ChatRequest,
    KnowledgeDoc,
    KnowledgeSourcesResponse,
    ReportParseResult,
    SessionCreateRequest,
    SessionDetail,
    SessionMessage,
    SessionRenameRequest,
    SessionSummary,
    SummaryArtifact,
    SummaryRequest,
)
from app.models.entities import SummaryArtifact as SummaryArtifactEntity
from app.services.knowledge_service import knowledge_service
from app.services.react_agent import react_agent_service
from app.services.report_progress_service import report_progress_service
from app.services.report_queue_service import report_queue_service
from app.services.report_service import report_service
from app.services.session_service import session_service
from app.services.summary_service import summary_service


# 这一层只做“路由分发”和“协议转换”：
# 1. 接收 HTTP 请求
# 2. 调用真正的业务服务
# 3. 把业务结果包装成 HTTP / SSE 响应
# 具体业务逻辑尽量都放在 services/ 中，避免 routes.py 变成“大杂烩”。
router = APIRouter()
settings = get_settings()
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _format_sse(event: str, data: object) -> str:
    """把一条事件格式化成标准 SSE 文本块。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.get("/sessions", response_model=list[SessionSummary])
def list_sessions(session: Session = Depends(get_session)) -> list[SessionSummary]:
    """返回左侧边栏需要的会话列表。"""
    return session_service.list_sessions(session)


@router.post("/sessions", response_model=SessionSummary)
def create_session(
    request: SessionCreateRequest,
    session: Session = Depends(get_session),
) -> SessionSummary:
    """创建一个新的空白会话。"""
    try:
        return session_service.create_session(session, request.title)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/sessions/{session_id}", response_model=SessionDetail)
def get_session_detail(session_id: str, session: Session = Depends(get_session)) -> SessionDetail:
    """读取单个会话的概览信息。"""
    try:
        return session_service.get_session_detail(session, session_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.patch("/sessions/{session_id}", response_model=SessionDetail)
def rename_session(
    session_id: str,
    request: SessionRenameRequest,
    session: Session = Depends(get_session),
) -> SessionDetail:
    """修改会话标题。"""
    try:
        return session_service.rename_session(session, session_id, request.title)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str, session: Session = Depends(get_session)) -> dict[str, str]:
    """删除指定会话，并清理该会话关联的消息和健康小结。"""
    try:
        session_service.delete_session(session, session_id)
        return {"session_id": session_id, "status": "deleted"}
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/sessions/{session_id}/messages", response_model=list[SessionMessage])
def get_session_messages(session_id: str, session: Session = Depends(get_session)) -> list[SessionMessage]:
    """返回一个会话下的历史消息列表。"""
    try:
        return session_service.list_messages(session, session_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/sessions/{session_id}/summaries/latest", response_model=SummaryArtifact)
def get_latest_summary(session_id: str, session: Session = Depends(get_session)) -> SummaryArtifact:
    """返回当前会话最新一份健康小结。"""
    try:
        session_service.get_session_entity(session, session_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    artifact = session.exec(
        select(SummaryArtifactEntity)
        .where(SummaryArtifactEntity.session_id == session_id)
        .order_by(SummaryArtifactEntity.created_at.desc())
    ).first()
    if not artifact:
        raise HTTPException(status_code=404, detail="Summary not found.")
    return SummaryArtifact(
        summary_id=artifact.id,
        markdown=artifact.markdown,
        pdf_path=artifact.pdf_path,
        created_at=artifact.created_at,
    )


@router.get("/sessions/{session_id}/summaries", response_model=list[SummaryArtifact])
def list_session_summaries(session_id: str, session: Session = Depends(get_session)) -> list[SummaryArtifact]:
    """返回当前会话下全部健康小结，供前端展示历史列表。"""
    try:
        session_service.get_session_entity(session, session_id)
        return summary_service.list_for_session(session, session_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/sessions/{session_id}/reports", response_model=ReportParseResult)
async def upload_report_for_session(
    session_id: str,
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
) -> ReportParseResult:
    """给当前会话上传并绑定一份报告。"""
    try:
        result = await report_service.create_upload(session, file, settings.upload_path)
        session_service.bind_report(session, session_id, result.report_id)
        report_queue_service.enqueue_report(session, result.report_id)
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/sessions/{session_id}/summaries/generate", response_model=SummaryArtifact)
def generate_summary_for_session(
    session_id: str,
    session: Session = Depends(get_session),
) -> SummaryArtifact:
    """基于当前会话上下文生成健康小结。"""
    try:
        return summary_service.generate_for_session(
            session=session,
            session_id=session_id,
            output_dir=settings.output_path,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/reports/upload", response_model=ReportParseResult)
async def upload_report(
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
) -> ReportParseResult:
    """上传报告并启动后台解析。"""
    try:
        result = await report_service.create_upload(session, file, settings.upload_path)
        report_queue_service.enqueue_report(session, result.report_id)
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/reports/{report_id}", response_model=ReportParseResult)
def get_report(report_id: str, session: Session = Depends(get_session)) -> ReportParseResult:
    """返回当前报告的结构化结果。"""
    try:
        return report_service.get_report(session, report_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/reports/{report_id}/stream")
def stream_report_progress(report_id: str, session: Session = Depends(get_session)) -> StreamingResponse:
    """通过 SSE 持续推送报告解析进度。"""
    try:
        initial = report_service.get_report(session, report_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    def event_generator():
        # 如果前端订阅时报告已经结束，这里直接补发最终状态。
        state = report_progress_service.get_state(report_id)
        if state is None and initial.parse_status in {"parsed", "needs_review", "error"}:
            payload = {
                "report_id": report_id,
                "stage": "completed" if initial.parse_status != "error" else "failed",
                "label": "报告解析完成" if initial.parse_status != "error" else "报告解析失败",
                "progress": 100,
                "parse_status": initial.parse_status,
                "done": True,
                "error": None,
            }
            yield _format_sse("progress", payload)
            yield _format_sse("final", initial.model_dump(mode="json"))
            return

        last_version = -1
        keepalive_at = time.monotonic()
        while True:
            # 只有进度版本真正变化时才推送，避免前端收到重复事件。
            latest = report_progress_service.get_state(report_id)
            if latest and latest.version != last_version:
                last_version = latest.version
                yield _format_sse("progress", latest.to_payload())
                if latest.done:
                    with Session(engine) as final_session:
                        final_report = report_service.get_report(final_session, report_id)
                    yield _format_sse("final", final_report.model_dump(mode="json"))
                    return

            now = time.monotonic()
            if now - keepalive_at >= 10:
                # SSE 保活，防止长连接被中间层提前回收。
                keepalive_at = now
                yield ": keepalive\n\n"

            time.sleep(0.25)

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=SSE_HEADERS)


@router.post("/agent/chat", response_model=AgentResponse)
def chat(request: ChatRequest, session: Session = Depends(get_session)) -> AgentResponse:
    """同步聊天接口，适合调试或非流式调用。"""
    try:
        return react_agent_service.respond(
            session,
            session_id=request.session_id,
            report_id=request.report_id,
            message=request.message,
            output_dir=settings.output_path,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/agent/chat/stream")
def chat_stream(request: ChatRequest, session: Session = Depends(get_session)) -> StreamingResponse:
    """流式聊天接口。

    事件类型包括 session / status / delta / final / error。
    """
    def event_generator():
        try:
            for event in react_agent_service.stream_respond(
                session,
                session_id=request.session_id,
                report_id=request.report_id,
                message=request.message,
                output_dir=settings.output_path,
            ):
                yield _format_sse(event["event"], event["data"])
        except Exception as exc:
            yield _format_sse("error", {"detail": str(exc)})

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=SSE_HEADERS)


@router.post("/summaries/generate", response_model=SummaryArtifact)
def generate_summary(request: SummaryRequest, session: Session = Depends(get_session)) -> SummaryArtifact:
    """根据报告生成 Markdown 和 PDF 小结。"""
    try:
        return summary_service.generate(
            session,
            report_id=request.report_id,
            session_id=request.session_id,
            output_dir=settings.output_path,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/summaries/{summary_id}/pdf")
def download_summary_pdf(summary_id: str, session: Session = Depends(get_session)) -> FileResponse:
    """下载已生成的 PDF 文件。"""
    artifact = session.get(SummaryArtifactEntity, summary_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Summary not found.")
    pdf_path = Path(artifact.pdf_path)
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF file not found.")
    return FileResponse(path=pdf_path, filename=pdf_path.name, media_type="application/pdf")


@router.get("/knowledge/sources", response_model=KnowledgeSourcesResponse)
def knowledge_sources(session: Session = Depends(get_session)) -> KnowledgeSourcesResponse:
    """返回知识库统计信息，主要用于调试和了解覆盖情况。"""
    total, breakdown, recent = knowledge_service.source_stats(session)
    return KnowledgeSourcesResponse(
        total_docs=total,
        trust_breakdown=breakdown,
        recent_docs=[
            KnowledgeDoc(
                doc_id=doc.id,
                title=doc.title,
                url=doc.url,
                source_domain=doc.source_domain,
                source_org=doc.source_org,
                trust_tier=doc.trust_tier,  # type: ignore[arg-type]
                content_type=doc.content_type,
                published_at=doc.published_at,
                snippet=doc.snippet,
            )
            for doc in recent
        ],
    )
