from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas
from sqlmodel import Session

from app.core.schemas import SummaryArtifact as SummaryArtifactSchema
from app.models.entities import SummaryArtifact
from app.services.report_service import report_service


class SummaryService:
    def generate(self, session: Session, report_id: str, session_id: str, output_dir: Path) -> SummaryArtifactSchema:
        report = report_service.get_report(session, report_id)
        missing_reference = "\u672a\u63d0\u4f9b"
        abnormal_lines = [
            f"- {item.name}: {item.value_raw}{item.unit} "
            f"(\u53c2\u8003\u8303\u56f4\uff1a{item.reference_range or missing_reference})"
            for item in report.abnormal_items
        ] or [
            "- \u672a\u68c0\u6d4b\u5230\u660e\u786e\u5f02\u5e38\u9879\uff0c\u5efa\u8bae\u7ed3\u5408\u539f\u59cb\u62a5\u544a\u548c\u533b\u751f\u610f\u89c1\u590d\u6838\u3002"
        ]

        markdown = "\n".join(
            [
                "# \u5065\u5eb7\u5c0f\u7ed3",
                "",
                "## \u5f02\u5e38\u6307\u6807\u6458\u8981",
                *abnormal_lines,
                "",
                "## \u7efc\u5408\u89e3\u8bfb\u4e0e\u98ce\u9669\u63d0\u793a",
                "\u672c\u5c0f\u7ed3\u57fa\u4e8e\u4e0a\u4f20\u7684\u4f53\u68c0/\u68c0\u9a8c\u7ed3\u679c\u81ea\u52a8\u751f\u6210\uff0c"
                "\u4ec5\u63d0\u4f9b\u98ce\u9669\u65b9\u5411\u548c\u590d\u67e5\u5efa\u8bae\uff0c\u4e0d\u6784\u6210\u8bca\u65ad\u610f\u89c1\u3002",
                "",
                "## \u751f\u6d3b\u65b9\u5f0f\u5efa\u8bae",
                "\u5efa\u8bae\u7ed3\u5408\u5f02\u5e38\u6307\u6807\u5173\u6ce8\u4f5c\u606f\u3001\u996e\u98df\u3001\u8fd0\u52a8\u4e0e\u590d\u67e5\u5b89\u6392\uff0c"
                "\u907f\u514d\u81ea\u884c\u7528\u836f\u6216\u5ffd\u89c6\u6301\u7eed\u75c7\u72b6\u3002",
                "",
                "## \u63a8\u8350\u5c31\u533b\u79d1\u5ba4",
                "\u82e5\u4e3a\u8840\u8102\u3001\u8840\u7cd6\u3001\u809d\u80be\u529f\u80fd\u7b49\u5f02\u5e38\uff0c"
                "\u53ef\u4f18\u5148\u8003\u8651\u5168\u79d1\u3001\u5185\u5206\u6ccc\u79d1\u6216\u76f8\u5e94\u4e13\u79d1\u8fdb\u4e00\u6b65\u8bc4\u4f30\u3002",
            ]
        )

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

    def _write_pdf(self, markdown: str, path: Path) -> None:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        pdf = canvas.Canvas(str(path), pagesize=A4)
        pdf.setTitle("\u5065\u5eb7\u5c0f\u7ed3")
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
