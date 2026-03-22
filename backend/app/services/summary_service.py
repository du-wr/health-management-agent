from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas
from sqlmodel import Session

from app.core.schemas import SummaryArtifact as SummaryArtifactSchema
from app.models.entities import SummaryArtifact
from app.services.knowledge_service import knowledge_service
from app.services.llm import llm_service
from app.services.prompt_templates import summary_generation_system_prompt, summary_generation_user_prompt
from app.services.report_service import report_service


class SummaryService:
    def generate(self, session: Session, report_id: str, session_id: str, output_dir: Path) -> SummaryArtifactSchema:
        report = report_service.get_report(session, report_id)
        if report.parse_status not in {"parsed", "needs_review"}:
            raise ValueError("Report is still processing.")

        explanations = knowledge_service.explain_lab_items(
            session,
            [item.name for item in report.abnormal_items[:6]] or [item.name for item in report.items[:6]],
        )
        report_summary = {
            "report_id": report.report_id,
            "file_name": report.file_name,
            "abnormal_items": [item.model_dump(mode="json") for item in report.abnormal_items[:8]],
            "item_count": len(report.items),
            "abnormal_count": len(report.abnormal_items),
        }
        markdown = self._build_markdown(report_summary, explanations)

        pdf_dir = output_dir / "pdf"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = pdf_dir / f"{report_id}-{session_id}.pdf"
        self._write_pdf(markdown, pdf_path)

        artifact = SummaryArtifact(
            report_id=report_id,
            session_id=session_id,
            markdown=markdown,
            pdf_path=str(pdf_path),
        )
        session.add(artifact)
        session.commit()
        session.refresh(artifact)
        return SummaryArtifactSchema(
            summary_id=artifact.id,
            markdown=artifact.markdown,
            pdf_path=artifact.pdf_path,
            created_at=artifact.created_at,
        )

    def _build_markdown(self, report_summary: dict[str, object], explanations: list[dict[str, str]]) -> str:
        if llm_service.is_configured:
            try:
                markdown = llm_service.chat_text_max(
                    system_prompt=summary_generation_system_prompt(),
                    user_prompt=summary_generation_user_prompt(report_summary, explanations),
                )
                if markdown.strip():
                    return markdown.strip()
            except Exception:
                pass

        abnormal_items = report_summary.get("abnormal_items", [])
        if isinstance(abnormal_items, list) and abnormal_items:
            abnormal_lines = [
                f"- {item['name']}: {item['value_raw']}{item.get('unit', '')}（参考范围：{item.get('reference_range') or '未提供'}）"
                for item in abnormal_items[:8]
                if isinstance(item, dict)
            ]
        else:
            abnormal_lines = ["- 暂未识别到明确异常项，建议结合原始报告和医生意见复核。"]

        explanation_lines = [f"- {item['title']}：{item['snippet']}" for item in explanations[:5]]
        if not explanation_lines:
            explanation_lines = ["- 当前没有足够的指标解释信息，建议结合复查结果和医生意见继续评估。"]

        return "\n".join(
            [
                "# 健康小结",
                "",
                "## 异常指标摘要",
                *abnormal_lines,
                "",
                "## 综合解读与风险提示",
                *explanation_lines,
                "",
                "## 生活方式建议",
                "- 结合异常指标关注饮食结构、运动、体重变化和规律复查。",
                "- 避免自行用药或仅凭单次体检结果下结论。",
                "",
                "## 推荐就医科室",
                "- 如为血脂、血糖、肝肾功能等异常，可优先考虑全科、内分泌科或相关专科进一步评估。",
            ]
        )

    def _write_pdf(self, markdown: str, path: Path) -> None:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        pdf = canvas.Canvas(str(path), pagesize=A4)
        pdf.setTitle("健康小结")
        pdf.setFont("STSong-Light", 12)
        _, height = A4
        y = height - 48
        for line in markdown.splitlines():
            if y < 48:
                pdf.showPage()
                pdf.setFont("STSong-Light", 12)
                y = height - 48
            pdf.drawString(48, y, line[:80])
            y -= 20
        pdf.save()


summary_service = SummaryService()
