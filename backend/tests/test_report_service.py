import asyncio
from io import BytesIO
from pathlib import Path

from fastapi import UploadFile
from sqlmodel import Session, SQLModel, create_engine

from app.services.report_progress_service import report_progress_service
from app.services.report_service import report_service


def make_session() -> Session:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    return Session(engine)


def test_determine_status_between_range() -> None:
    assert report_service._determine_status(6.2, "3.9-6.1") == "high"
    assert report_service._determine_status(3.2, "3.9-6.1") == "low"
    assert report_service._determine_status(4.8, "3.9-6.1") == "normal"


def test_determine_status_single_bound() -> None:
    assert report_service._determine_status(5.8, "< 5.7") == "high"
    assert report_service._determine_status(42.0, ">= 40") == "normal"
    assert report_service._determine_status(30.0, ">= 40") == "low"


def test_create_upload_returns_processing_status(tmp_path: Path) -> None:
    with make_session() as session:
        upload = UploadFile(filename="report.pdf", file=BytesIO(b"dummy"))
        result = asyncio.run(report_service.create_upload(session, upload, tmp_path))
        assert result.parse_status == "processing"
        assert result.items == []
        progress = report_progress_service.get_state(result.report_id)
        assert progress is not None
        assert progress.stage == "queued"
        assert progress.progress == 5
