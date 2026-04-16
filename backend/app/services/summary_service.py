from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas
from sqlmodel import Session, select

from app.core.schemas import SummaryArtifact as SummaryArtifactSchema
from app.models.entities import SummaryArtifact
from app.services.knowledge_service import knowledge_service
from app.services.llm import llm_service
from app.services.prompt_templates import summary_generation_system_prompt, summary_generation_user_prompt
from app.services.report_service import report_service
from app.services.session_service import session_service


class SummaryService:
    """健康小结服务。"""

    def list_for_session(self, session: Session, session_id: str) -> list[SummaryArtifactSchema]:
        """返回当前会话下全部健康小结，按创建时间倒序排列。"""
        artifacts = session.exec(
            select(SummaryArtifact)
            .where(SummaryArtifact.session_id == session_id)
            .order_by(SummaryArtifact.created_at.desc())
        ).all()
        return [self._to_schema(artifact) for artifact in artifacts]

    def generate(self, session: Session, report_id: str, session_id: str, output_dir: Path) -> SummaryArtifactSchema:
        """生成一份 Markdown + PDF 健康小结。"""
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
        conversation_context = self._build_conversation_context(session, session_id)
        markdown = self._build_markdown(report_summary, explanations, conversation_context)

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
        return self._to_schema(artifact)

    def generate_for_session(self, session: Session, session_id: str, output_dir: Path) -> SummaryArtifactSchema:
        """基于当前会话上下文生成健康小结。"""
        chat_session = session_service.get_session_entity(session, session_id)
        if not chat_session.report_id:
            raise ValueError("Current session has no bound report.")
        return self.generate(
            session=session,
            report_id=chat_session.report_id,
            session_id=session_id,
            output_dir=output_dir,
        )

    def _build_markdown(
        self,
        report_summary: dict[str, object],
        explanations: list[dict[str, str]],
        conversation_context: list[dict[str, object]],
    ) -> str:
        """优先走大模型生成 Markdown，失败时再回退到模板兜底。"""
        if llm_service.is_configured:
            try:
                markdown = llm_service.chat_text_max(
                    system_prompt=summary_generation_system_prompt(),
                    user_prompt=summary_generation_user_prompt(report_summary, explanations, conversation_context),
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
                *self._build_focus_lines(conversation_context),
                *explanation_lines,
                "",
                "## 生活方式建议",
                "- 结合异常指标关注饮食结构、运动、体重变化和规律复查。",
                "- 避免自行用药，或仅凭单次体检结果直接下结论。",
                "",
                "## 推荐就医科室",
                "- 如为血脂、血糖、肝肾功能等异常，可优先考虑全科、内分泌科或相关专科进一步评估。",
            ]
        )

    def _build_conversation_context(self, session: Session, session_id: str) -> list[dict[str, object]]:
        """提取最近几轮对话，作为健康小结的上下文素材。"""
        messages = session_service.get_recent_messages(session, session_id, limit=8)
        return [
            {
                "role": message.role,
                "content": message.content,
                "created_at": message.created_at.isoformat(),
            }
            for message in messages
        ]

    def _build_focus_lines(self, conversation_context: list[dict[str, object]]) -> list[str]:
        """把最近对话里用户最关心的点压缩成几行提示。"""
        user_messages = [
            str(item.get("content") or "").strip()
            for item in conversation_context
            if item.get("role") == "user" and str(item.get("content") or "").strip()
        ]
        if not user_messages:
            return []
        return [
            "- 本次对话重点关注：",
            *[f"  - {content[:60]}" for content in user_messages[-3:]],
        ]

    def _write_pdf(self, markdown: str, path: Path) -> None:
        """把 Markdown 文本按简单排版写成 PDF。"""
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

    def _to_schema(self, artifact: SummaryArtifact) -> SummaryArtifactSchema:
        """统一把数据库里的小结记录转换成接口层结构。"""
        return SummaryArtifactSchema(
            summary_id=artifact.id,
            markdown=artifact.markdown,
            pdf_path=artifact.pdf_path,
            created_at=artifact.created_at,
        )


summary_service = SummaryService()
