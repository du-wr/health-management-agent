from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field
from sqlmodel import Session, select

from app.models.entities import LabItem, Report, SessionReportLink


REPORT_READY_STATUSES = {"parsed", "needs_review"}


class NormalizedLabItem(BaseModel):
    """标准化后的指标结果。"""

    original_name: str
    normalized_name: str
    canonical_code: str
    display_name: str
    value_raw: str
    value_num: float | None = None
    unit: str = ""
    reference_range: str = ""
    status: str = "unknown"


class TrendComparisonItem(BaseModel):
    """同一指标在两份报告之间的趋势比较。"""

    normalized_name: str
    display_name: str
    current_report_id: str
    previous_report_id: str
    current_value: float
    previous_value: float
    unit: str = ""
    delta: float
    direction: str
    summary: str


class ReportTrendResult(BaseModel):
    """报告趋势比较结果。"""

    session_id: str
    current_report_id: str
    previous_report_id: str | None = None
    comparisons: list[TrendComparisonItem] = Field(default_factory=list)
    summary: str = ""


class ReportRiskFlag(BaseModel):
    """规则型风险提示。

    这里只做“需要重点关注”的标记，不做诊断。
    """

    level: str
    item_name: str
    normalized_name: str
    reason: str
    suggested_action: str


class ReportToolService:
    """服务长期健康管理的报告工具链。"""

    _ALIASES: dict[str, tuple[str, tuple[str, ...]]] = {
        "ldl_c": ("低密度脂蛋白胆固醇", ("低密度脂蛋白", "低密度脂蛋白胆固醇", "ldl", "ldlc", "ldl-c")),
        "hdl_c": ("高密度脂蛋白胆固醇", ("高密度脂蛋白", "高密度脂蛋白胆固醇", "hdl", "hdlc", "hdl-c")),
        "total_cholesterol": ("总胆固醇", ("总胆固醇", "胆固醇", "tc")),
        "triglyceride": ("甘油三酯", ("甘油三酯", "三酰甘油", "tg")),
        "fasting_glucose": ("空腹血糖", ("空腹血糖", "葡萄糖", "血糖", "glu", "glucose", "fpg")),
        "hba1c": ("糖化血红蛋白", ("糖化血红蛋白", "hba1c", "糖化血红蛋白a1c")),
        "uric_acid": ("尿酸", ("尿酸", "ua", "uricacid")),
        "alt": ("丙氨酸氨基转移酶", ("谷丙转氨酶", "丙氨酸氨基转移酶", "alt")),
        "ast": ("天门冬氨酸氨基转移酶", ("谷草转氨酶", "天门冬氨酸氨基转移酶", "ast")),
        "creatinine": ("肌酐", ("肌酐", "cr", "creatinine")),
        "hemoglobin": ("血红蛋白", ("血红蛋白", "hemoglobin", "hb")),
    }

    def normalize_lab_items(self, items: list[dict[str, Any]]) -> list[NormalizedLabItem]:
        """把报告指标名规范化，便于趋势比较和风险规则复用。"""

        normalized: list[NormalizedLabItem] = []
        for item in items:
            original_name = str(item.get("name") or "").strip()
            normalized_name = self._normalize_key(original_name)
            canonical_code = normalized_name
            display_name = original_name or "未命名指标"
            for code, (alias_display, aliases) in self._ALIASES.items():
                normalized_aliases = {self._normalize_key(alias) for alias in aliases}
                if normalized_name in normalized_aliases:
                    canonical_code = code
                    display_name = alias_display
                    break
            normalized.append(
                NormalizedLabItem(
                    original_name=original_name,
                    normalized_name=normalized_name,
                    canonical_code=canonical_code,
                    display_name=display_name,
                    value_raw=str(item.get("value_raw") or "").strip(),
                    value_num=self._to_float(item.get("value_num")),
                    unit=str(item.get("unit") or "").strip(),
                    reference_range=str(item.get("reference_range") or "").strip(),
                    status=str(item.get("status") or "unknown").strip() or "unknown",
                )
            )
        return normalized

    def compare_report_trends(
        self,
        session: Session,
        session_id: str,
        current_report_id: str,
        *,
        focus_item_names: list[str] | None = None,
    ) -> ReportTrendResult:
        """比较当前报告与同会话上一份报告的重叠指标趋势。"""

        current_report = session.get(Report, current_report_id)
        if current_report is None:
            return ReportTrendResult(session_id=session_id, current_report_id=current_report_id, summary="未找到当前报告，暂时无法比较趋势。")

        previous_report = self._find_previous_report(session, session_id, current_report_id)
        if previous_report is None:
            return ReportTrendResult(
                session_id=session_id,
                current_report_id=current_report_id,
                summary="当前会话下还没有更早的已解析报告，暂时无法做趋势比较。",
            )

        current_items = self._load_report_items(session, current_report_id)
        previous_items = self._load_report_items(session, previous_report.id)
        current_normalized = self.normalize_lab_items(current_items)
        previous_normalized = self.normalize_lab_items(previous_items)
        previous_by_code = {item.canonical_code: item for item in previous_normalized if item.value_num is not None}
        focus_codes = {self._match_focus_code(name) for name in (focus_item_names or [])}
        focus_codes.discard("")

        comparisons: list[TrendComparisonItem] = []
        for item in current_normalized:
            if item.value_num is None:
                continue
            if focus_codes and item.canonical_code not in focus_codes and item.normalized_name not in focus_codes:
                continue
            previous_item = previous_by_code.get(item.canonical_code)
            if previous_item is None or previous_item.value_num is None:
                continue
            delta = round(item.value_num - previous_item.value_num, 4)
            direction = self._trend_direction(delta)
            if direction == "flat":
                summary = f"{item.display_name} 与上一份报告相比基本稳定。"
            else:
                summary = (
                    f"{item.display_name} 较上一份报告{self._direction_label(direction)} "
                    f"{abs(delta):g}{item.unit or ''}。"
                )
            comparisons.append(
                TrendComparisonItem(
                    normalized_name=item.canonical_code,
                    display_name=item.display_name,
                    current_report_id=current_report_id,
                    previous_report_id=previous_report.id,
                    current_value=item.value_num,
                    previous_value=previous_item.value_num,
                    unit=item.unit or previous_item.unit,
                    delta=delta,
                    direction=direction,
                    summary=summary,
                )
            )

        summary = "当前报告可与上一份已解析报告进行趋势对比。" if comparisons else "已找到历史报告，但当前重点指标缺少可直接比较的数值。"
        return ReportTrendResult(
            session_id=session_id,
            current_report_id=current_report_id,
            previous_report_id=previous_report.id,
            comparisons=comparisons[:6],
            summary=summary,
        )

    def build_report_risk_flags(
        self,
        items: list[dict[str, Any]],
        normalized_items: list[NormalizedLabItem] | None = None,
    ) -> list[ReportRiskFlag]:
        """按保守规则给出需要重点关注的风险提示。"""

        normalized_items = normalized_items or self.normalize_lab_items(items)
        flags: list[ReportRiskFlag] = []
        for item in normalized_items:
            value_num = item.value_num
            if value_num is None:
                continue
            rule_flag = self._match_rule_flag(item)
            if rule_flag is not None:
                flags.append(rule_flag)
                continue
            if item.status in {"high", "low"}:
                flags.append(
                    ReportRiskFlag(
                        level="monitor",
                        item_name=item.original_name or item.display_name,
                        normalized_name=item.canonical_code,
                        reason=f"{item.display_name} 当前结果偏离参考范围，建议结合原始报告和复查趋势继续跟踪。",
                        suggested_action="建议保留本次结果，并结合饮食、用药、复查时间点与症状变化继续观察。",
                    )
                )
        return flags[:8]

    def _find_previous_report(self, session: Session, session_id: str, current_report_id: str) -> Report | None:
        links = session.exec(
            select(SessionReportLink)
            .where(SessionReportLink.session_id == session_id)
            .order_by(SessionReportLink.linked_at.desc())
        ).all()
        for link in links:
            if link.report_id == current_report_id:
                continue
            report = session.get(Report, link.report_id)
            if report and report.parse_status in REPORT_READY_STATUSES:
                return report
        return None

    def ensure_session_report_link(self, session: Session, session_id: str, report_id: str) -> None:
        """确保会话和报告存在一条历史关联记录。"""

        existing = session.exec(
            select(SessionReportLink).where(
                SessionReportLink.session_id == session_id,
                SessionReportLink.report_id == report_id,
            )
        ).first()
        if existing is not None:
            return
        session.add(SessionReportLink(session_id=session_id, report_id=report_id))
        session.commit()

    def delete_session_links(self, session: Session, session_id: str) -> None:
        """删除会话时顺带清理历史报告关联。"""

        links = session.exec(select(SessionReportLink).where(SessionReportLink.session_id == session_id)).all()
        for link in links:
            session.delete(link)

    def _load_report_items(self, session: Session, report_id: str) -> list[dict[str, Any]]:
        items = session.exec(select(LabItem).where(LabItem.report_id == report_id)).all()
        return [
            {
                "name": item.name,
                "value_raw": item.value_raw,
                "value_num": item.value_num,
                "unit": item.unit,
                "reference_range": item.reference_range,
                "status": item.status,
            }
            for item in items
        ]

    def _match_focus_code(self, name: str) -> str:
        normalized_name = self._normalize_key(name)
        for code, (_, aliases) in self._ALIASES.items():
            if normalized_name in {self._normalize_key(alias) for alias in aliases}:
                return code
        return normalized_name

    def _match_rule_flag(self, item: NormalizedLabItem) -> ReportRiskFlag | None:
        value_num = item.value_num
        if value_num is None:
            return None

        if item.canonical_code == "fasting_glucose" and value_num >= 7.0:
            return self._high_priority_flag(item, "空腹血糖明显升高，建议尽快结合原始报告与复查安排进一步评估。")
        if item.canonical_code == "hba1c" and value_num >= 6.5:
            return self._high_priority_flag(item, "糖化血红蛋白明显升高，提示需要尽快核对既往结果并安排进一步评估。")
        if item.canonical_code == "triglyceride" and value_num >= 5.6:
            return self._high_priority_flag(item, "甘油三酯显著升高，建议尽快复核报告并线下咨询医生。")
        if item.canonical_code == "ldl_c" and value_num >= 4.9:
            return self._prompt_review_flag(item, "低密度脂蛋白胆固醇明显升高，建议结合既往血脂结果和心血管风险因素尽快复查。")
        if item.canonical_code == "uric_acid" and value_num >= 540:
            return self._prompt_review_flag(item, "尿酸偏高幅度较大，建议结合饮食、饮水和既往发作情况尽快复查。")
        if item.canonical_code in {"alt", "ast"} and value_num >= 120:
            return self._prompt_review_flag(item, "转氨酶升高幅度较大，建议结合近期用药、饮酒和感染情况尽快复查。")
        if item.canonical_code == "creatinine" and value_num >= 180:
            return self._prompt_review_flag(item, "肌酐升高较明显，建议尽快结合肾功能复查和线下评估。")
        if item.canonical_code == "hemoglobin" and value_num < 90:
            return self._prompt_review_flag(item, "血红蛋白降低较明显，建议尽快结合症状和进一步检查评估。")
        return None

    def _high_priority_flag(self, item: NormalizedLabItem, reason: str) -> ReportRiskFlag:
        return ReportRiskFlag(
            level="high_priority",
            item_name=item.original_name or item.display_name,
            normalized_name=item.canonical_code,
            reason=reason,
            suggested_action="建议尽快核对原始报告、近期症状和既往结果，并线下咨询医生。",
        )

    def _prompt_review_flag(self, item: NormalizedLabItem, reason: str) -> ReportRiskFlag:
        return ReportRiskFlag(
            level="prompt_review",
            item_name=item.original_name or item.display_name,
            normalized_name=item.canonical_code,
            reason=reason,
            suggested_action="建议优先安排复查，并结合生活方式、用药和症状变化继续评估。",
        )

    def _normalize_key(self, name: str) -> str:
        return "".join(ch for ch in str(name).lower() if ch.isalnum())

    def _to_float(self, value: Any) -> float | None:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _trend_direction(self, delta: float) -> str:
        if abs(delta) < 1e-6:
            return "flat"
        return "up" if delta > 0 else "down"

    def _direction_label(self, direction: str) -> str:
        if direction == "up":
            return "上升"
        if direction == "down":
            return "下降"
        return "变化"


report_tool_service = ReportToolService()
