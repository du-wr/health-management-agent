from __future__ import annotations

from app.core.schemas import IntentName


class RoutingService:
    def route(self, message: str, has_report: bool) -> IntentName:
        text = message.strip()
        if any(
            keyword in text
            for keyword in (
                "\u662f\u4ec0\u4e48",
                "\u4ec0\u4e48\u610f\u601d",
                "\u89e3\u91ca\u4e00\u4e0b",
                "\u542b\u4e49",
            )
        ):
            if not has_report or any(
                keyword in text for keyword in ("\u672f\u8bed", "\u6307\u6807", "\u9879\u76ee")
            ):
                return "term_explanation"
        if has_report and any(
            keyword in text
            for keyword in (
                "\u62a5\u544a",
                "\u4f53\u68c0",
                "\u6307\u6807",
                "\u5f02\u5e38",
                "\u590d\u67e5",
                "\u6302\u4ec0\u4e48\u79d1",
            )
        ):
            return "report_follow_up"
        if any(
            keyword in text
            for keyword in (
                "\u75c7\u72b6",
                "\u4e0d\u8212\u670d",
                "\u75bc",
                "\u53d1\u70e7",
                "\u54b3\u55fd",
                "\u5934\u6655",
                "\u8179\u6cfb",
                "\u80f8\u95f7",
            )
        ):
            return "symptom_rag_advice"
        if len(text) < 6:
            return "collect_more_info"
        return "term_explanation" if not has_report else "report_follow_up"


routing_service = RoutingService()
