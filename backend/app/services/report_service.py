from __future__ import annotations

import json
import re
from pathlib import Path
from uuid import uuid4

import pdfplumber
from fastapi import UploadFile
from sqlmodel import Session, delete, select

from app.core.schemas import LabItem as LabItemSchema
from app.core.schemas import ReportParseResult
from app.models.entities import LabItem, Report
from app.services.llm import llm_service


LAB_LINE_PATTERN = re.compile(
    r"(?P<name>[\u4e00-\u9fffA-Za-z0-9()\-]+)\s+"
    r"(?P<value>[<>]?\d+(?:\.\d+)?)\s*"
    r"(?P<unit>[A-Za-z%/\u4e00-\u9fff]+)?\s*"
    r"(?P<range>(?:\d+(?:\.\d+)?\s*[-~]\s*\d+(?:\.\d+)?)|(?:[<>]=?\s*\d+(?:\.\d+)?))?"
)


class ReportService:
    async def parse_upload(self, session: Session, file: UploadFile, upload_dir: Path) -> ReportParseResult:
        suffix = Path(file.filename or "upload.bin").suffix or ".bin"
        file_name = file.filename or f"report{suffix}"
        storage_name = f"{uuid4()}{suffix}"
        saved_path = upload_dir / storage_name
        saved_path.write_bytes(await file.read())

        report = Report(file_name=file_name, file_path=str(saved_path))
        session.add(report)
        session.commit()
        session.refresh(report)

        raw_text, warnings = self.extract_text(saved_path)
        items = self.extract_lab_items(raw_text)

        session.exec(delete(LabItem).where(LabItem.report_id == report.id))
        for item in items:
            session.add(
                LabItem(
                    report_id=report.id,
                    name=item.name,
                    value_raw=item.value_raw,
                    value_num=item.value_num,
                    unit=item.unit,
                    reference_range=item.reference_range,
                    status=item.status,
                    clinical_note=item.clinical_note,
                )
            )

        report.raw_text = raw_text
        report.parse_status = "parsed" if items else "needs_review"
        report.parse_warnings_json = json.dumps(warnings, ensure_ascii=False)
        session.add(report)
        session.commit()

        abnormal_items = [item for item in items if item.status in {"high", "low"}]
        return ReportParseResult(
            report_id=report.id,
            file_name=report.file_name,
            items=items,
            abnormal_items=abnormal_items,
            raw_text=raw_text,
            parse_warnings=warnings,
            parse_status=report.parse_status,
        )

    def get_report(self, session: Session, report_id: str) -> ReportParseResult:
        report = session.get(Report, report_id)
        if not report:
            raise ValueError("Report not found.")

        items = session.exec(select(LabItem).where(LabItem.report_id == report_id)).all()
        schema_items = [
            LabItemSchema(
                name=item.name,
                value_raw=item.value_raw,
                value_num=item.value_num,
                unit=item.unit,
                reference_range=item.reference_range,
                status=item.status,
                clinical_note=item.clinical_note,
            )
            for item in items
        ]
        warnings = json.loads(report.parse_warnings_json or "[]")
        abnormal_items = [item for item in schema_items if item.status in {"high", "low"}]
        return ReportParseResult(
            report_id=report.id,
            file_name=report.file_name,
            items=schema_items,
            abnormal_items=abnormal_items,
            raw_text=report.raw_text,
            parse_warnings=warnings,
            parse_status=report.parse_status,
        )

    def extract_text(self, path: Path) -> tuple[str, list[str]]:
        warnings: list[str] = []
        if path.suffix.lower() == ".pdf":
            text_chunks: list[str] = []
            try:
                with pdfplumber.open(path) as pdf:
                    for page in pdf.pages:
                        text_chunks.append(page.extract_text() or "")
            except Exception as exc:
                warnings.append(f"PDF parse failed: {exc}")
            raw_text = "\n".join(chunk for chunk in text_chunks if chunk).strip()
            if len(raw_text) >= 40:
                return raw_text, warnings
            warnings.append("PDF text was too short. Configure OCR for scanned reports.")
            return raw_text, warnings

        if path.suffix.lower() in {".png", ".jpg", ".jpeg"} and llm_service.is_configured:
            try:
                ocr_text = llm_service.image_to_text(
                    path,
                    "Extract all text from this Chinese medical checkup report in reading order.",
                )
                return ocr_text.strip(), warnings
            except Exception as exc:
                warnings.append(f"OCR failed: {exc}")

        warnings.append("OCR is unavailable in the current environment.")
        return "", warnings

    def extract_lab_items(self, raw_text: str) -> list[LabItemSchema]:
        if not raw_text:
            return []

        if llm_service.is_configured:
            try:
                payload = llm_service.chat_json(
                    system_prompt=(
                        "You extract lab items from Chinese medical checkup text. "
                        "Return a JSON object with an items array. "
                        'Each item must contain: name, value_raw, value_num, unit, reference_range, clinical_note.'
                    ),
                    user_prompt=raw_text[:12000],
                )
                items = []
                for item in payload.get("items", []):
                    parsed = self._finalize_item(
                        name=str(item.get("name", "")).strip(),
                        value_raw=str(item.get("value_raw", "")).strip(),
                        value_num=item.get("value_num"),
                        unit=str(item.get("unit", "")).strip(),
                        reference_range=str(item.get("reference_range", "")).strip(),
                        clinical_note=str(item.get("clinical_note", "")).strip() or None,
                    )
                    if parsed:
                        items.append(parsed)
                if items:
                    return items
            except Exception:
                pass

        items: list[LabItemSchema] = []
        for line in raw_text.splitlines():
            match = LAB_LINE_PATTERN.search(line)
            if not match:
                continue
            parsed = self._finalize_item(
                name=match.group("name"),
                value_raw=match.group("value"),
                value_num=float(match.group("value")),
                unit=(match.group("unit") or "").strip(),
                reference_range=(match.group("range") or "").strip(),
                clinical_note=None,
            )
            if parsed:
                items.append(parsed)
        return items

    def _finalize_item(
        self,
        name: str,
        value_raw: str,
        value_num: float | str | None,
        unit: str,
        reference_range: str,
        clinical_note: str | None,
    ) -> LabItemSchema | None:
        if not name or not value_raw:
            return None

        numeric_value: float | None = None
        if isinstance(value_num, (int, float)):
            numeric_value = float(value_num)
        else:
            try:
                numeric_value = float(str(value_num)) if value_num is not None else float(value_raw)
            except Exception:
                numeric_value = None

        return LabItemSchema(
            name=name.strip(),
            value_raw=str(value_raw).strip(),
            value_num=numeric_value,
            unit=unit.strip(),
            reference_range=reference_range.strip(),
            status=self._determine_status(numeric_value, reference_range),
            clinical_note=clinical_note,
        )

    def _determine_status(self, value_num: float | None, reference_range: str) -> str:
        if value_num is None or not reference_range:
            return "unknown"

        ref = reference_range.replace(" ", "")
        between = re.match(r"(?P<low>\d+(?:\.\d+)?)\s*[-~]\s*(?P<high>\d+(?:\.\d+)?)", ref)
        if between:
            low = float(between.group("low"))
            high = float(between.group("high"))
            if value_num < low:
                return "low"
            if value_num > high:
                return "high"
            return "normal"

        lt = re.match(r"<=?(?P<limit>\d+(?:\.\d+)?)", ref)
        if lt:
            return "high" if value_num > float(lt.group("limit")) else "normal"

        gt = re.match(r">=?(?P<limit>\d+(?:\.\d+)?)", ref)
        if gt:
            return "low" if value_num < float(gt.group("limit")) else "normal"

        return "unknown"


report_service = ReportService()
