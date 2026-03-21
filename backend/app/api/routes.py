from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlmodel import Session

from app.core.config import get_settings
from app.core.database import get_session
from app.core.schemas import (
    AgentResponse,
    ChatRequest,
    KnowledgeBootstrapResponse,
    KnowledgeDoc,
    KnowledgeSourcesResponse,
    ReportParseResult,
    SummaryArtifact,
    SummaryRequest,
)
from app.models.entities import SummaryArtifact as SummaryArtifactEntity
from app.services.knowledge_service import knowledge_service
from app.services.react_agent import react_agent_service
from app.services.report_service import report_service
from app.services.summary_service import summary_service


router = APIRouter()
settings = get_settings()


@router.post("/reports/upload", response_model=ReportParseResult)
async def upload_report(
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
) -> ReportParseResult:
    try:
        return await report_service.parse_upload(session, file, settings.upload_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/reports/{report_id}", response_model=ReportParseResult)
def get_report(report_id: str, session: Session = Depends(get_session)) -> ReportParseResult:
    try:
        return report_service.get_report(session, report_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


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


@router.post("/knowledge/bootstrap", response_model=KnowledgeBootstrapResponse)
def bootstrap_knowledge(session: Session = Depends(get_session)) -> KnowledgeBootstrapResponse:
    if not settings.allow_knowledge_bootstrap:
        raise HTTPException(status_code=403, detail="Knowledge bootstrap is disabled.")
    try:
        ingested, skipped = knowledge_service.bootstrap(session)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return KnowledgeBootstrapResponse(
        ingested=ingested,
        skipped=skipped,
        message="Knowledge bootstrap completed.",
    )


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
