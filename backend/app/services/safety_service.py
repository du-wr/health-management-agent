from __future__ import annotations

from dataclasses import dataclass


EMERGENCY_KEYWORDS = {
    "\u80f8\u75db",
    "\u547c\u5438\u56f0\u96be",
    "\u5598\u4e0d\u4e0a\u6c14",
    "\u6655\u5385",
    "\u660f\u5385",
    "\u9ed1\u4fbf",
    "\u4fbf\u8840",
    "\u62bd\u6410",
    "\u7672\u75eb\u53d1\u4f5c",
}
DIAGNOSIS_KEYWORDS = {
    "\u8bca\u65ad",
    "\u662f\u4e0d\u662f",
    "\u786e\u8bca",
    "\u5224\u65ad\u662f\u4e0d\u662f",
    "\u764c",
    "\u80bf\u7624",
}
PRESCRIPTION_KEYWORDS = {
    "\u5f00\u836f",
    "\u5904\u65b9",
    "\u5403\u4ec0\u4e48\u836f",
    "\u6362\u836f",
    "\u600e\u4e48\u7528\u836f",
    "\u5242\u91cf",
    "\u7528\u91cf",
    "\u7597\u7a0b",
}
REPLACE_DOCTOR_KEYWORDS = {
    "\u66ff\u4ee3\u533b\u751f",
    "\u4e0d\u7528\u770b\u533b\u751f",
    "\u4e0d\u53bb\u533b\u9662",
    "\u76f4\u63a5\u544a\u8bc9\u6211\u65b9\u6848",
}

DEFAULT_SAFETY_APPENDIX = (
    "\u91cd\u8981\u63d0\u793a\uff1a\u4ee5\u4e0a\u5185\u5bb9\u4ec5\u4f9b\u5065\u5eb7\u4fe1\u606f\u53c2\u8003\uff0c"
    "\u4e0d\u80fd\u66ff\u4ee3\u4e13\u4e1a\u533b\u7597\u5efa\u8bae\u3002"
    "\u5982\u75c7\u72b6\u52a0\u91cd\u6216\u51fa\u73b0\u6025\u75c7\u4fe1\u53f7\uff0c\u8bf7\u53ca\u65f6\u5c31\u533b\u3002"
)


@dataclass
class SafetyDecision:
    level: str
    handoff_required: bool
    reason: str | None = None


class SafetyService:
    def evaluate(self, message: str) -> SafetyDecision:
        text = message.strip()
        if any(keyword in text for keyword in EMERGENCY_KEYWORDS):
            return SafetyDecision(
                level="handoff",
                handoff_required=True,
                reason="\u68c0\u6d4b\u5230\u6025\u75c7\u98ce\u9669\u4fe1\u53f7\u3002",
            )
        if any(keyword in text for keyword in PRESCRIPTION_KEYWORDS):
            return SafetyDecision(
                level="handoff",
                handoff_required=True,
                reason="\u68c0\u6d4b\u5230\u5904\u65b9\u6216\u5242\u91cf\u8bf7\u6c42\u3002",
            )
        if any(keyword in text for keyword in REPLACE_DOCTOR_KEYWORDS):
            return SafetyDecision(
                level="handoff",
                handoff_required=True,
                reason="\u68c0\u6d4b\u5230\u66ff\u4ee3\u533b\u751f\u51b3\u7b56\u8bf7\u6c42\u3002",
            )
        if any(keyword in text for keyword in DIAGNOSIS_KEYWORDS):
            return SafetyDecision(
                level="handoff",
                handoff_required=True,
                reason="\u68c0\u6d4b\u5230\u8bca\u65ad\u6216\u764c\u75c7\u5224\u65ad\u8bf7\u6c42\u3002",
            )
        return SafetyDecision(level="safe", handoff_required=False)


safety_service = SafetyService()
