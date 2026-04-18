from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.core.schemas import AgentResponse
from app.services.routing_service import routing_service
from app.services.safety_service import SafetyDecision, safety_service


EvalCategory = Literal["routing", "safety", "report_follow_up", "auto_report_analysis", "memory"]


class AgentEvalHistoryMessage(BaseModel):
    """评测用的历史消息种子。"""

    role: Literal["user", "assistant"]
    content: str
    intent: str | None = None
    safety_level: str = "safe"


class AgentEvalLabItemSeed(BaseModel):
    """评测用的报告指标种子。"""

    name: str
    value_raw: str
    value_num: float | None = None
    unit: str = ""
    reference_range: str = ""
    status: str = "unknown"


class AgentEvalCase(BaseModel):
    """单条 Agent 评测用例。"""

    case_id: str
    name: str
    category: EvalCategory
    message: str
    has_report: bool = False
    report_file_name: str = "report.pdf"
    report_raw_text: str = ""
    report_parse_status: str = "parsed"
    report_items: list[AgentEvalLabItemSeed] = Field(default_factory=list)
    history_messages: list[AgentEvalHistoryMessage] = Field(default_factory=list)
    expected_intent: str | None = None
    expected_safety_level: str | None = None
    expected_handoff_required: bool | None = None
    expected_used_tools: list[str] = Field(default_factory=list)
    expected_memory_keys: list[str] = Field(default_factory=list)
    expected_follow_up_min: int | None = None
    answer_must_include: list[str] = Field(default_factory=list)
    answer_must_not_include: list[str] = Field(default_factory=list)


class AgentEvalResult(BaseModel):
    """单条评测结果。"""

    case_id: str
    category: EvalCategory
    passed: bool
    failures: list[str] = Field(default_factory=list)
    actual_intent: str | None = None
    actual_safety_level: str | None = None
    actual_handoff_required: bool | None = None
    actual_used_tools: list[str] = Field(default_factory=list)


class AgentEvalSummary(BaseModel):
    """一批用例的聚合结果。"""

    total: int
    passed: int
    failed: int
    results: list[AgentEvalResult] = Field(default_factory=list)
    category_summary: dict[str, dict[str, int]] = Field(default_factory=dict)


class AgentEvalService:
    """Agent 回归评测的轻量服务层。

    当前先聚焦三类职责：
    1. 从 JSON 文件加载结构化 case
    2. 对 routing / safety 做快速规则评测
    3. 对 AgentResponse 做统一断言，避免测试文件里散落重复校验逻辑
    """

    def load_cases(self, path: Path) -> list[AgentEvalCase]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_cases = payload.get("cases", payload) if isinstance(payload, dict) else payload
        if not isinstance(raw_cases, list):
            raise ValueError("评测文件格式错误，cases 必须是列表。")
        return [AgentEvalCase.model_validate(item) for item in raw_cases]

    def summarize(self, results: list[AgentEvalResult]) -> AgentEvalSummary:
        passed = sum(1 for item in results if item.passed)
        category_summary: dict[str, dict[str, int]] = {}
        for item in results:
            bucket = category_summary.setdefault(item.category, {"total": 0, "passed": 0, "failed": 0})
            bucket["total"] += 1
            if item.passed:
                bucket["passed"] += 1
            else:
                bucket["failed"] += 1
        return AgentEvalSummary(
            total=len(results),
            passed=passed,
            failed=len(results) - passed,
            results=results,
            category_summary=category_summary,
        )

    def evaluate_routing_case(self, case: AgentEvalCase) -> AgentEvalResult:
        actual_intent = routing_service.route(case.message, has_report=case.has_report)
        failures: list[str] = []
        if case.expected_intent and actual_intent != case.expected_intent:
            failures.append(f"意图不匹配：期望 {case.expected_intent}，实际 {actual_intent}")
        return AgentEvalResult(
            case_id=case.case_id,
            category=case.category,
            passed=not failures,
            failures=failures,
            actual_intent=actual_intent,
        )

    def evaluate_safety_case(self, case: AgentEvalCase) -> AgentEvalResult:
        decision = safety_service.evaluate(case.message)
        failures = self._validate_safety(case, decision)
        return AgentEvalResult(
            case_id=case.case_id,
            category=case.category,
            passed=not failures,
            failures=failures,
            actual_safety_level=decision.level,
            actual_handoff_required=decision.handoff_required,
        )

    def evaluate_response_case(self, case: AgentEvalCase, response: AgentResponse) -> AgentEvalResult:
        failures = self._validate_response(case, response)
        return AgentEvalResult(
            case_id=case.case_id,
            category=case.category,
            passed=not failures,
            failures=failures,
            actual_intent=response.intent,
            actual_safety_level=response.safety_level,
            actual_handoff_required=response.handoff_required,
            actual_used_tools=list(response.used_tools),
        )

    def _validate_safety(self, case: AgentEvalCase, decision: SafetyDecision) -> list[str]:
        failures: list[str] = []
        if case.expected_safety_level and decision.level != case.expected_safety_level:
            failures.append(f"安全级别不匹配：期望 {case.expected_safety_level}，实际 {decision.level}")
        if case.expected_handoff_required is not None and decision.handoff_required != case.expected_handoff_required:
            failures.append(
                f"handoff_required 不匹配：期望 {case.expected_handoff_required}，实际 {decision.handoff_required}"
            )
        return failures

    def _validate_response(self, case: AgentEvalCase, response: AgentResponse) -> list[str]:
        failures: list[str] = []

        if case.expected_intent and response.intent != case.expected_intent:
            failures.append(f"意图不匹配：期望 {case.expected_intent}，实际 {response.intent}")
        if case.expected_safety_level and response.safety_level != case.expected_safety_level:
            failures.append(f"安全级别不匹配：期望 {case.expected_safety_level}，实际 {response.safety_level}")
        if case.expected_handoff_required is not None and response.handoff_required != case.expected_handoff_required:
            failures.append(
                f"handoff_required 不匹配：期望 {case.expected_handoff_required}，实际 {response.handoff_required}"
            )

        for tool_name in case.expected_used_tools:
            if tool_name not in response.used_tools:
                failures.append(f"缺少预期工具：{tool_name}")

        for snippet in case.answer_must_include:
            if snippet not in response.answer:
                failures.append(f"回答缺少关键片段：{snippet}")

        for snippet in case.answer_must_not_include:
            if snippet and snippet in response.answer:
                failures.append(f"回答包含不应出现的片段：{snippet}")

        if case.expected_follow_up_min is not None and len(response.follow_up_questions) < case.expected_follow_up_min:
            failures.append(
                f"追问数量不足：期望至少 {case.expected_follow_up_min} 条，实际 {len(response.follow_up_questions)} 条"
            )

        if case.expected_memory_keys:
            debug_memory: dict[str, Any] = response.debug.memory if response.debug else {}
            for memory_key in case.expected_memory_keys:
                if memory_key not in debug_memory:
                    failures.append(f"缺少预期 memory 键：{memory_key}")

        return failures


agent_eval_service = AgentEvalService()
