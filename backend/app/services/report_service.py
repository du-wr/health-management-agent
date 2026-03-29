from __future__ import annotations

import json
import re
from pathlib import Path
from uuid import uuid4

import pdfplumber
from fastapi import UploadFile
from sqlmodel import Session, delete, select

from app.core.database import engine
from app.core.schemas import LabItem as LabItemSchema
from app.core.schemas import ReportParseResult
from app.models.entities import LabItem, Report
from app.services.llm import llm_service
from app.services.prompt_templates import lab_extraction_system_prompt, report_ocr_prompt
from app.services.report_progress_service import report_progress_service


LAB_LINE_PATTERN = re.compile(
    r"(?P<name>[\u4e00-\u9fffA-Za-z0-9()\-]+)\s+"
    r"(?P<value>[<>]?\d+(?:\.\d+)?)\s*"
    r"(?P<unit>[A-Za-z%/\u4e00-\u9fff]+)?\s*"
    r"(?P<range>(?:\d+(?:\.\d+)?\s*[-~]\s*\d+(?:\.\d+)?)|(?:[<>]=?\s*\d+(?:\.\d+)?))?"
)


class ReportService:
    """处理体检报告上传、文本提取、结构化抽取和异常判断。"""

    async def create_upload(self, session: Session, file: UploadFile, upload_dir: Path) -> ReportParseResult:
        """保存上传文件，并创建一条“处理中”的报告记录。"""
        suffix = Path(file.filename or "upload.bin").suffix or ".bin"
        file_name = file.filename or f"report{suffix}"
        storage_name = f"{uuid4()}{suffix}"
        saved_path = upload_dir / storage_name
        saved_path.write_bytes(await file.read())

        report = Report(
            file_name=file_name,
            file_path=str(saved_path),
            parse_status="processing",
            parse_warnings_json="[]",
        )
        session.add(report)
        session.commit()
        session.refresh(report)
        report_progress_service.initialize(report.id, parse_status=report.parse_status)
        return self._build_report_result(report, [])

    def process_report(self, report_id: str) -> None:
        """后台任务入口：新开一个数据库会话完成整份报告解析。"""
        with Session(engine) as session:
            self.process_report_with_session(session, report_id)

    def process_report_with_session(self, session: Session, report_id: str) -> None:
        """真正的报告处理主流程。"""
        report = session.get(Report, report_id)
        if not report:
            return

        report.parse_status = "processing"
        session.add(report)
        session.commit()
        report_progress_service.update(
            report.id,
            stage="processing",
            label="开始解析报告",
            progress=10,
            parse_status=report.parse_status,
        )

        try:
            raw_text, warnings = self.extract_text(Path(report.file_path), report_id=report.id)
            report_progress_service.update(
                report.id,
                stage="parsing_items",
                label="正在抽取结构化指标",
                progress=68,
                parse_status=report.parse_status,
            )
            items = self.extract_lab_items(raw_text)

            report_progress_service.update(
                report.id,
                stage="saving",
                label="正在保存解析结果",
                progress=90,
                parse_status=report.parse_status,
            )
            # 每次重新解析都会覆盖旧的指标，避免同一份报告重复累积旧数据。
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
            report_progress_service.mark_complete(report.id, parse_status=report.parse_status)
        except Exception as exc:
            report.parse_status = "error"
            report.parse_warnings_json = json.dumps([f"Parse failed: {exc}"], ensure_ascii=False)
            session.add(report)
            session.commit()
            report_progress_service.mark_failed(report.id, error=str(exc))

    def get_report(self, session: Session, report_id: str) -> ReportParseResult:
        """返回单份报告及其所有结构化指标。"""
        report = session.get(Report, report_id)
        if not report:
            raise ValueError("Report not found.")

        items = session.exec(select(LabItem).where(LabItem.report_id == report_id)).all()
        return self._build_report_result(report, items)

    def extract_text(self, path: Path, report_id: str | None = None) -> tuple[str, list[str]]:
        """从文件中提取文本。

        处理策略：
        - 文本型 PDF：优先 pdfplumber
        - 图片：如果配置了视觉模型，就走 OCR
        - 提取失败：返回 warning，由上层决定如何展示
        """
        warnings: list[str] = []
        if path.suffix.lower() == ".pdf":
            if report_id:
                report_progress_service.update(
                    report_id,
                    stage="extracting_text",
                    label="正在提取 PDF 文本",
                    progress=24,
                    parse_status="processing",
                )
            text_chunks: list[str] = []
            try:
                with pdfplumber.open(path) as pdf:
                    for page in pdf.pages:
                        text_chunks.append(page.extract_text() or "")
            except Exception as exc:
                warnings.append(f"PDF parse failed: {exc}")
            raw_text = "\n".join(chunk for chunk in text_chunks if chunk).strip()
            if len(raw_text) >= 40:
                # 长度太短时通常说明 PDF 更像扫描件，而不是真正的文本 PDF。
                return raw_text, warnings
            warnings.append("PDF text was too short. Configure OCR for scanned reports.")
            if report_id:
                report_progress_service.update(
                    report_id,
                    stage="review_needed",
                    label="文本提取不足，当前报告可能需要 OCR 或人工复核",
                    progress=58,
                    parse_status="processing",
                )
            return raw_text, warnings

        if path.suffix.lower() in {".png", ".jpg", ".jpeg"} and llm_service.is_configured:
            try:
                if report_id:
                    report_progress_service.update(
                        report_id,
                        stage="ocr",
                        label="正在执行图片 OCR",
                        progress=42,
                        parse_status="processing",
                    )
                ocr_text = llm_service.image_to_text(path, report_ocr_prompt())
                return ocr_text.strip(), warnings
            except Exception as exc:
                warnings.append(f"OCR failed: {exc}")

        if report_id:
            report_progress_service.update(
                report_id,
                stage="ocr_unavailable",
                label="当前环境无法完成 OCR，已返回已有文本结果",
                progress=60,
                parse_status="processing",
            )
        warnings.append("OCR is unavailable in the current environment.")
        return "", warnings

    def extract_lab_items(self, raw_text: str) -> list[LabItemSchema]:
        """把原始文本转成结构化指标列表。"""
        if not raw_text:
            return []

        if llm_service.is_configured:
            try:
                payload = llm_service.chat_json_fast(
                    system_prompt=lab_extraction_system_prompt(),
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
                # 结构化抽取失败时不要中断整条链，直接退回正则兜底。
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

    def _build_report_result(self, report: Report, items: list[LabItem | LabItemSchema]) -> ReportParseResult:
        """把 ORM 对象转换成前端真正消费的 schema。"""
        schema_items: list[LabItemSchema] = []
        for item in items:
            if isinstance(item, LabItemSchema):
                schema_items.append(item)
            else:
                schema_items.append(
                    LabItemSchema(
                        name=item.name,
                        value_raw=item.value_raw,
                        value_num=item.value_num,
                        unit=item.unit,
                        reference_range=item.reference_range,
                        status=item.status,
                        clinical_note=item.clinical_note,
                    )
                )
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

    def _finalize_item(
        self,
        name: str,
        value_raw: str,
        value_num: float | str | None,
        unit: str,
        reference_range: str,
        clinical_note: str | None,
    ) -> LabItemSchema | None:
        """对单个指标做最后清洗，并计算 high / low / normal 状态。"""
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
        """根据参考范围做规则判断。

        这一步故意不用大模型，因为高低判断是确定性逻辑。
        """
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
