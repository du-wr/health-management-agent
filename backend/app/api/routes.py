import json
import time
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from sqlmodel import Session

from app.core.config import get_settings
from app.core.database import engine, get_session
from app.core.schemas import (
    AgentResponse,
    ChatRequest,
    KnowledgeDoc,
    KnowledgeSourcesResponse,
    ReportParseResult,
    SummaryArtifact,
    SummaryRequest,
)
from app.models.entities import SummaryArtifact as SummaryArtifactEntity
from app.services.knowledge_service import knowledge_service
from app.services.react_agent import react_agent_service
from app.services.report_progress_service import report_progress_service
from app.services.report_service import report_service
from app.services.summary_service import summary_service


router = APIRouter()
settings = get_settings()
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _format_sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/reports/upload", response_model=ReportParseResult)
async def upload_report(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
) -> ReportParseResult:
    try:
        result = await report_service.create_upload(session, file, settings.upload_path)
        background_tasks.add_task(report_service.process_report, result.report_id)
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/reports/{report_id}", response_model=ReportParseResult)
def get_report(report_id: str, session: Session = Depends(get_session)) -> ReportParseResult:
    try:
        return report_service.get_report(session, report_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/reports/{report_id}/stream")
def stream_report_progress(report_id: str, session: Session = Depends(get_session)) -> StreamingResponse:
    try:
        initial = report_service.get_report(session, report_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    def event_generator():
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
                keepalive_at = now
                yield ": keepalive\n\n"

            time.sleep(0.25)

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=SSE_HEADERS)


@router.post("/agent/chat", response_model=AgentResponse)
def chat(request: ChatRequest, session: Session = Depends(get_session)) -> AgentResponse:
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
    artifact = session.get(SummaryArtifactEntity, summary_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Summary not found.")
    pdf_path = Path(artifact.pdf_path)
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF file not found.")
    return FileResponse(path=pdf_path, filename=pdf_path.name, media_type="application/pdf")


@router.get("/knowledge/sources", response_model=KnowledgeSourcesResponse)
def knowledge_sources(session: Session = Depends(get_session)) -> KnowledgeSourcesResponse:
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
