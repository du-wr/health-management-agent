from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Iterator

from pydantic import BaseModel, Field
from sqlmodel import Session, select

from app.core.config import get_settings
from app.core.schemas import AgentDebug, AgentResponse, Citation
from app.models.entities import ChatMessage, ChatSession
from app.services.knowledge_service import knowledge_service
from app.services.llm import llm_service
from app.services.prompt_templates import (
    answer_composer_system_prompt,
    answer_composer_user_prompt,
    answer_repair_system_prompt,
    answer_repair_user_prompt,
    input_analysis_system_prompt,
    input_analysis_user_prompt,
    lab_batch_interpreter_system_prompt,
    lab_batch_interpreter_user_prompt,
    report_answer_polish_system_prompt,
    report_answer_polish_user_prompt,
    report_follow_up_planner_system_prompt,
    report_follow_up_planner_user_prompt,
    report_synthesis_system_prompt,
    report_synthesis_user_prompt,
    term_explanation_system_prompt,
)
from app.services.report_service import report_service
from app.services.routing_service import routing_service
from app.services.safety_service import DEFAULT_SAFETY_APPENDIX, safety_service
from app.services.who_service import who_service


logger = logging.getLogger(__name__)

DRUG_KEYWORDS = ("药", "用药", "药物", "药品")
REPORT_READY_STATUSES = {"parsed", "needs_review"}
WHO_SOURCE_SUFFIX = "来源于 WHO ICD-11"
TERM_HEADINGS = {"它是什么", "常见表现或特点", "常见诱因或易感因素", "什么时候需要就医", "温馨提示"}
REPORT_HEADINGS = {"主要异常解读", "综合解读", "后续建议"}
QUESTION_TAIL_PATTERN = re.compile(r"(是什么病|是什么|什么意思|啥意思|解释一下|科普一下)$")
HEADING_PREFIX_PATTERN = re.compile(r"^(?:#{1,6}\s*|\d+[.)、]\s*|-+\s*)")


class AgentExecutionResult(BaseModel):
    intent: str
    answer: str = ""
    citations: list[Citation] = Field(default_factory=list)
    used_tools: list[str] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)
    tool_outputs: list[dict[str, Any]] = Field(default_factory=list)
    conversation_history: list[dict[str, str]] = Field(default_factory=list)
    message: str = ""
    use_max: bool = True
    debug: dict[str, Any] = Field(default_factory=dict)


class BatchInterpretationResult(BaseModel):
    items: list[dict[str, Any]] = Field(default_factory=list)


class InputAnalysisResult(BaseModel):
    intent: str = "collect_more_info"
    rewritten_query: str = ""
    normalized_term: str = ""
    use_local_knowledge: bool = True
    use_who: bool = False
    use_max: bool = True
    reason: str = ""


class ReportFollowUpPlan(BaseModel):
    focus_item_names: list[str] = Field(default_factory=list)
    need_item_explanations: bool = True
    need_synthesis: bool = True
    need_next_steps: bool = True
    synthesis_axes: list[str] = Field(default_factory=list)
    follow_up_needed: bool = False
    reason: str = ""


class ReportSynthesisResult(BaseModel):
    summary: str = ""
    priority_axes: list[str] = Field(default_factory=list)
    combined_findings: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)


class ReactAgentService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.answer_cache: dict[str, AgentExecutionResult] = {}

    def respond(
        self,
        session: Session,
        session_id: str | None,
        report_id: str | None,
        message: str,
        output_dir: Path,
    ) -> AgentResponse:
        chat_session = self._get_or_create_session(session, session_id, report_id, message)
        self._store_message(session, chat_session.id, "user", message)
        immediate = self._handle_immediate_response(session, chat_session.id, report_id, message)
        if immediate:
            self._store_message(
                session,
                chat_session.id,
                "assistant",
                immediate.answer,
                immediate.intent,
                "handoff" if immediate.handoff_required else "safe",
                immediate.citations,
            )
            return immediate

        cache_key = self._cache_key(chat_session.id, report_id, message)
        execution = self.answer_cache.get(cache_key)
        if not isinstance(execution, AgentExecutionResult):
            execution = None
        if execution and self._is_incomplete_answer(self._strip_duplicate_appendix(execution.answer)):
            self.answer_cache.pop(cache_key, None)
            execution = None

        if not execution:
            execution = self._prepare_fast_path(session, chat_session.id, report_id, message)
            execution.answer = self._compose_answer(
                intent=execution.intent,
                message=execution.message,
                conversation_history=execution.conversation_history,
                tool_outputs=execution.tool_outputs,
                citations=execution.citations,
                used_tools=execution.used_tools,
                use_max=execution.use_max,
            )
            self.answer_cache[cache_key] = execution

        response = self._build_agent_response(chat_session.id, execution)
        self._store_message(
            session,
            chat_session.id,
            "assistant",
            response.answer,
            response.intent,
            response.safety_level,
            response.citations,
        )
        return response

    def stream_respond(
        self,
        session: Session,
        session_id: str | None,
        report_id: str | None,
        message: str,
        output_dir: Path,
    ) -> Iterator[dict[str, Any]]:
        chat_session = self._get_or_create_session(session, session_id, report_id, message)
        self._store_message(session, chat_session.id, "user", message)
        yield {"event": "session", "data": {"session_id": chat_session.id}}

        immediate = self._handle_immediate_response(session, chat_session.id, report_id, message)
        if immediate:
            yield {"event": "status", "data": {"label": "已完成安全检查"}}
            for chunk in self._chunk_text(immediate.answer):
                yield {"event": "delta", "data": {"text": chunk}}
            self._store_message(
                session,
                chat_session.id,
                "assistant",
                immediate.answer,
                immediate.intent,
                "handoff" if immediate.handoff_required else "safe",
                immediate.citations,
            )
            yield {"event": "final", "data": immediate.model_dump(mode="json")}
            return

        yield {"event": "status", "data": {"label": "分析问题中"}}
        execution = self._prepare_fast_path(session, chat_session.id, report_id, message)
        for tool in execution.used_tools:
            yield {"event": "status", "data": {"label": self._tool_status_label(tool)}}
        yield {"event": "status", "data": {"label": "生成回答中"}}

        execution.answer = self._compose_answer(
            intent=execution.intent,
            message=execution.message,
            conversation_history=execution.conversation_history,
            tool_outputs=execution.tool_outputs,
            citations=execution.citations,
            used_tools=execution.used_tools,
            use_max=execution.use_max,
        )
        self.answer_cache[self._cache_key(chat_session.id, report_id, message)] = execution
        response = self._build_agent_response(chat_session.id, execution)
        self._store_message(
            session,
            chat_session.id,
            "assistant",
            response.answer,
            response.intent,
            response.safety_level,
            response.citations,
        )
        for chunk in self._chunk_text(response.answer):
            yield {"event": "delta", "data": {"text": chunk}}
        yield {"event": "final", "data": response.model_dump(mode="json")}

    def _handle_immediate_response(
        self,
        session: Session,
        session_id: str,
        report_id: str | None,
        message: str,
    ) -> AgentResponse | None:
        decision = safety_service.evaluate(message)
        if decision.handoff_required:
            answer = f"{decision.reason or '当前问题涉及高风险医疗决策。'}\n\n请尽快联系医生或线下医疗机构获取专业评估。"
            return AgentResponse(
                session_id=session_id,
                intent="safety_handoff",
                answer=self._with_safety_appendix(answer),
                citations=[],
                used_tools=[],
                follow_up_questions=[],
                safety_level="handoff",
                handoff_required=True,
            )

        if report_id:
            report = report_service.get_report(session, report_id)
            if report.parse_status not in REPORT_READY_STATUSES:
                answer = "报告还在后台解析中，等解析完成后我再为你解读。" if report.parse_status != "error" else "报告解析失败，建议重新上传更清晰的文件。"
                return AgentResponse(
                    session_id=session_id,
                    intent="collect_more_info",
                    answer=self._with_safety_appendix(answer),
                    citations=[],
                    used_tools=[],
                    follow_up_questions=[],
                    safety_level="safe",
                    handoff_required=False,
                )
        return None

    def _prepare_fast_path(
        self,
        session: Session,
        chat_session_id: str,
        report_id: str | None,
        message: str,
    ) -> AgentExecutionResult:
        conversation_history = self._recent_conversation_context(session, chat_session_id)
        analysis = self._analyze_input(message, report_id is not None, conversation_history)
        if analysis.intent == "collect_more_info":
            return AgentExecutionResult(
                intent="collect_more_info",
                follow_up_questions=["请补充你最关注的是哪项指标、有没有不适，以及这些情况持续了多久。"],
                conversation_history=conversation_history,
                message=message,
                use_max=analysis.use_max,
                debug={"analysis": analysis.model_dump(mode="json")},
            )

        if report_id and analysis.intent == "report_follow_up":
            report = report_service.get_report(session, report_id)
            seed_focus_items = [item.model_dump(mode="json") for item in (report.abnormal_items[:8] or report.items[:8])]
            related_items = [item.model_dump(mode="json") for item in report.items[:12]]
            plan = self._plan_report_follow_up(message, conversation_history, seed_focus_items, related_items)
            focus_items = self._select_focus_items_from_plan(plan, seed_focus_items, related_items)
            interpretations, citations = self._interpret_lab_batch(session, focus_items, related_items)
            synthesis = self._build_report_synthesis(message, plan, interpretations, related_items)
            return AgentExecutionResult(
                intent="report_follow_up",
                citations=citations,
                used_tools=["search_report_items", "interpret_lab"],
                follow_up_questions=self._report_follow_up_questions(plan),
                tool_outputs=[
                    {"tool": "search_report_items", "result": {"abnormal_items": [item.model_dump(mode="json") for item in report.abnormal_items[:8]], "focus_items": focus_items, "related_items": related_items, "raw_text_excerpt": report.raw_text[:800], "analysis_reason": analysis.reason}},
                    {"tool": "report_follow_up_plan", "result": plan.model_dump(mode="json")},
                    {"tool": "interpret_lab", "result": {"items": interpretations}},
                    {"tool": "report_synthesis", "result": synthesis.model_dump(mode="json")},
                ],
                conversation_history=conversation_history,
                message=message,
                use_max=analysis.use_max,
                debug={"analysis": analysis.model_dump(mode="json"), "plan": plan.model_dump(mode="json"), "synthesis": synthesis.model_dump(mode="json")},
            )

        if analysis.intent == "term_explanation":
            return self._prepare_term_explanation(session, conversation_history, message, analysis)

        effective_query = analysis.rewritten_query or message
        tool_name = "query_drug" if any(token in effective_query for token in DRUG_KEYWORDS) else "retrieve_knowledge"
        docs = knowledge_service.retrieve(session, effective_query) if analysis.use_local_knowledge else []
        return AgentExecutionResult(
            intent=analysis.intent,
            citations=self._knowledge_citations(docs),
            used_tools=[tool_name],
            follow_up_questions=self._default_follow_up_questions(analysis.intent),
            tool_outputs=[{"tool": tool_name, "result": {"query": effective_query, "docs": knowledge_service.pack_docs(docs), "analysis_reason": analysis.reason}}],
            conversation_history=conversation_history,
            message=message,
            use_max=analysis.use_max,
            debug={"analysis": analysis.model_dump(mode="json")},
        )

    def _plan_report_follow_up(
        self,
        message: str,
        conversation_history: list[dict[str, str]],
        focus_items: list[dict[str, Any]],
        related_items: list[dict[str, Any]],
    ) -> ReportFollowUpPlan:
        fallback = ReportFollowUpPlan(
            focus_item_names=[item.get("name", "") for item in focus_items[:4] if item.get("name")],
            need_item_explanations=True,
            need_synthesis=True,
            need_next_steps=True,
            synthesis_axes=[],
            follow_up_needed=len(message.strip()) <= 6,
            reason="rule_fallback",
        )
        if not llm_service.is_configured:
            return fallback
        try:
            payload = llm_service.chat_json_fast(
                report_follow_up_planner_system_prompt(),
                report_follow_up_planner_user_prompt(
                    message=message,
                    conversation_history=[] if self._should_ignore_report_history(message) else conversation_history[-4:],
                    focus_items=focus_items,
                    related_items=related_items,
                ),
            )
            parsed = ReportFollowUpPlan.model_validate(payload)
            if not parsed.focus_item_names:
                parsed.focus_item_names = fallback.focus_item_names
            return parsed
        except Exception:
            logger.exception("Report follow-up planning failed")
            return fallback

    def _select_focus_items_from_plan(
        self,
        plan: ReportFollowUpPlan,
        focus_items: list[dict[str, Any]],
        related_items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not plan.focus_item_names:
            return focus_items[:6]
        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        pool = focus_items + related_items
        for name in plan.focus_item_names:
            matched = self._match_report_item_by_name(name, pool)
            if not matched:
                continue
            matched_name = str(matched.get("name") or "")
            if matched_name in seen:
                continue
            seen.add(matched_name)
            selected.append(matched)
        return selected[:6] or focus_items[:6]

    def _match_report_item_by_name(self, target_name: str, items: list[dict[str, Any]]) -> dict[str, Any] | None:
        normalized_target = self._normalize_item_name(target_name)
        for item in items:
            item_name = str(item.get("name") or "")
            normalized_name = self._normalize_item_name(item_name)
            if normalized_name == normalized_target or (normalized_target and (normalized_target in normalized_name or normalized_name in normalized_target)):
                return item
        return None

    def _normalize_item_name(self, name: str) -> str:
        return re.sub(r"[\s\\-_/()（）]+", "", name).lower()

    def _build_report_synthesis(
        self,
        message: str,
        plan: ReportFollowUpPlan,
        interpretations: list[dict[str, Any]],
        related_items: list[dict[str, Any]],
    ) -> ReportSynthesisResult:
        fallback = self._fallback_report_synthesis(plan, interpretations)
        if not llm_service.is_configured or not interpretations:
            return fallback
        try:
            payload = llm_service.chat_json_fast(
                report_synthesis_system_prompt(),
                report_synthesis_user_prompt(
                    message=message,
                    plan=plan.model_dump(mode="json"),
                    interpretations=interpretations,
                    related_items=related_items,
                ),
            )
            parsed = ReportSynthesisResult.model_validate(payload)
            if not parsed.summary:
                parsed.summary = fallback.summary
            if not parsed.priority_axes:
                parsed.priority_axes = fallback.priority_axes
            if not parsed.combined_findings:
                parsed.combined_findings = fallback.combined_findings
            if not parsed.next_steps:
                parsed.next_steps = fallback.next_steps
            return parsed
        except Exception:
            logger.exception("Report synthesis failed")
            return fallback

    def _fallback_report_synthesis(self, plan: ReportFollowUpPlan, interpretations: list[dict[str, Any]]) -> ReportSynthesisResult:
        axes = [axis for axis in plan.synthesis_axes[:4] if axis]
        if not axes:
            for item in interpretations[:4]:
                department = str(item.get("suggested_department") or "").strip()
                if department and department not in axes:
                    axes.append(department)
        names = [str(item.get("name") or "") for item in interpretations[:4] if item.get("name")]
        summary = "从当前报告看，建议优先关注这些方向：" + "、".join(axes) + "。" if axes else "这些异常需要结合参考范围、症状和复查结果一起综合判断。"
        findings = []
        if names:
            findings.append("当前更值得优先解释的异常项目包括：" + "、".join(names) + "。")
        findings.append("单次体检异常不等于明确疾病，仍需结合复查趋势和临床表现判断。")
        next_steps = []
        if plan.need_next_steps:
            next_steps.append("建议先结合原始报告参考范围、采血条件和近期症状一起判断。")
            if axes:
                next_steps.append("如异常持续存在，可优先咨询：" + "、".join(axes[:3]) + "。")
            next_steps.append("如涉及空腹状态、饮食、运动、体重变化或脱水情况，也建议一并回顾。")
        return ReportSynthesisResult(summary=summary, priority_axes=axes, combined_findings=findings, next_steps=next_steps)

    def _prepare_term_explanation(
        self,
        session: Session,
        conversation_history: list[dict[str, str]],
        message: str,
        analysis: InputAnalysisResult,
    ) -> AgentExecutionResult:
        effective_query = analysis.rewritten_query or analysis.normalized_term or self._strip_question_tail(message)
        docs = knowledge_service.retrieve(session, effective_query) if analysis.use_local_knowledge else []
        if not docs and effective_query != message:
            docs = knowledge_service.retrieve(session, message)
        tool_outputs = [{"tool": "retrieve_knowledge", "result": {"query": effective_query, "docs": knowledge_service.pack_docs(docs), "normalized_term": analysis.normalized_term, "analysis_reason": analysis.reason}}]
        citations = self._knowledge_citations(docs)
        used_tools = ["retrieve_knowledge"]

        if analysis.use_who and who_service.is_configured():
            try:
                who_result = self._lookup_who_with_candidates(analysis.normalized_term or effective_query or message, docs)
                if who_result.get("matches"):
                    tool_outputs.append({"tool": "lookup_icd11", "result": who_result})
                    used_tools.append("lookup_icd11")
                    citations.extend(self._who_citations(who_result))
                else:
                    logger.info("WHO ICD-11 returned no match for message=%s", message)
            except Exception:
                logger.exception("WHO ICD-11 lookup failed for query=%s", message)
        elif analysis.use_who:
            logger.info("WHO ICD-11 skipped because credentials are not configured")

        return AgentExecutionResult(
            intent="term_explanation",
            citations=self._dedupe_citations(citations),
            used_tools=used_tools,
            follow_up_questions=["如果你愿意，我可以继续解释它常见的检查方法、风险因素或一般处理思路。"],
            tool_outputs=tool_outputs,
            conversation_history=conversation_history,
            message=message,
            use_max=analysis.use_max,
            debug={"analysis": analysis.model_dump(mode="json")},
        )

    def _lookup_who_with_candidates(self, message: str, docs: list[Any]) -> dict[str, Any]:
        candidates = self._build_who_queries(message, docs)
        for query in candidates:
            result = who_service.search(query)
            if result.get("matches"):
                result["query_candidates"] = candidates
                result["matched_query"] = query
                return result
        return {"query": message, "matched_query": None, "query_candidates": candidates, "matches": []}

    def _build_who_queries(self, message: str, docs: list[Any]) -> list[str]:
        candidates: list[str] = []
        self._push_query_candidate(candidates, message)
        self._push_query_candidate(candidates, self._strip_question_tail(message))
        for doc in docs[:3]:
            self._push_query_candidate(candidates, getattr(doc, "title", ""))
            body_text = getattr(doc, "body_text", "") or ""
            alias_line = body_text.splitlines()[0] if body_text else ""
            for alias in alias_line.split():
                self._push_query_candidate(candidates, alias)
        return candidates

    def _push_query_candidate(self, candidates: list[str], query: str) -> None:
        normalized = self._normalize_query(query)
        if normalized and normalized not in candidates:
            candidates.append(normalized)

    def _interpret_lab_batch(
        self,
        session: Session,
        focus_items: list[dict[str, Any]],
        related_items: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[Citation]]:
        explanations = knowledge_service.explain_lab_items(session, [str(item["name"]) for item in focus_items])
        citations = [Citation(source_type="knowledge_doc", doc_id=item["doc_id"], title=item["title"], url=item["url"], trust_tier=item["trust_tier"], snippet=item["snippet"]) for item in explanations]
        if llm_service.is_configured and explanations:
            try:
                payload = llm_service.chat_json_fast(
                    lab_batch_interpreter_system_prompt(),
                    lab_batch_interpreter_user_prompt(focus_items, related_items, explanations),
                )
                parsed = BatchInterpretationResult.model_validate(payload)
                if parsed.items:
                    parsed_map = {item["name"]: item for item in parsed.items if item.get("name")}
                    return [{**item, **parsed_map.get(str(item.get("name")), {})} for item in focus_items], self._dedupe_citations(citations)
            except Exception:
                logger.exception("Batch lab interpretation failed")

        explanation_map = {item["name"]: item for item in explanations}
        fallback_items = []
        for item in focus_items:
            matched = explanation_map.get(str(item.get("name") or ""))
            fallback_items.append(
                {
                    **item,
                    "meaning": matched["snippet"] if matched else "当前本地知识库暂时没有足够解释，建议结合原始报告与复查结果一起判断。",
                    "common_reasons": "建议结合饮食、体重、空腹状态、作息和既往病史综合判断。",
                    "watch_points": "如异常持续存在，应结合复查和医生意见进一步评估。",
                    "suggested_department": "全科或相关专科",
                }
            )
        return fallback_items, self._dedupe_citations(citations)

    def _compose_answer(
        self,
        intent: str,
        message: str,
        conversation_history: list[dict[str, str]],
        tool_outputs: list[dict[str, Any]],
        citations: list[Citation],
        used_tools: list[str],
        use_max: bool,
    ) -> str:
        if intent == "report_follow_up":
            draft = self._build_report_follow_up_answer(tool_outputs)
            if llm_service.is_configured and use_max and draft:
                plan, synthesis = self._extract_report_debug_materials(tool_outputs)
                try:
                    polished = llm_service.chat_text_max(
                        report_answer_polish_system_prompt(),
                        report_answer_polish_user_prompt(message=message, draft_answer=draft, plan=plan, synthesis=synthesis),
                    ).strip()
                    if polished and not self._is_incomplete_answer(polished):
                        return polished
                except Exception:
                    logger.exception("Report answer polish failed")
            return draft or self._fallback_answer(intent, tool_outputs, citations, used_tools)

        if llm_service.is_configured and use_max:
            try:
                answer = llm_service.chat_text_max(
                    self._answer_system_prompt(intent),
                    answer_composer_user_prompt(
                        intent=intent,
                        message=message,
                        conversation_history=conversation_history,
                        tool_outputs=tool_outputs,
                        citations=[citation.model_dump(mode="json") for citation in citations],
                    ),
                ).strip()
                if answer and not self._is_incomplete_answer(answer):
                    return self._append_source_marker(intent, used_tools, answer)
                repaired = self._repair_incomplete_answer(intent, message, answer, tool_outputs, citations, used_tools)
                if repaired:
                    return repaired
            except Exception:
                logger.exception("Answer composition failed")
        return self._fallback_answer(intent, tool_outputs, citations, used_tools)

    def _extract_report_debug_materials(self, tool_outputs: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
        plan = next((item["result"] for item in tool_outputs if item["tool"] == "report_follow_up_plan"), {})
        synthesis = next((item["result"] for item in tool_outputs if item["tool"] == "report_synthesis"), {})
        return (plan if isinstance(plan, dict) else {}, synthesis if isinstance(synthesis, dict) else {})

    def _build_report_follow_up_answer(self, tool_outputs: list[dict[str, Any]]) -> str:
        lab_result = next((item["result"] for item in tool_outputs if item["tool"] == "interpret_lab"), {})
        synthesis_result = next((item["result"] for item in tool_outputs if item["tool"] == "report_synthesis"), {})
        items = lab_result.get("items", []) if isinstance(lab_result, dict) else []
        if not items:
            return ""
        lines = ["### 主要异常解读"]
        for item in items:
            name = str(item.get("name") or "").strip()
            value_text = f"{item.get('value_raw', '')}{item.get('unit', '')}".strip()
            meaning = str(item.get("meaning") or "").strip()
            common_reasons = str(item.get("common_reasons") or "").strip()
            watch_points = str(item.get("watch_points") or "").strip()
            entry = f"- **{name}**"
            if value_text:
                entry += f" {value_text}"
            if meaning:
                entry += f"：{meaning}"
            lines.append(entry)
            if common_reasons:
                lines.append(f"常见影响因素：{common_reasons}")
            if watch_points:
                lines.append(f"需要关注：{watch_points}")
        lines.extend(["", "### 综合解读"])
        summary = str(synthesis_result.get("summary") or "").strip() if isinstance(synthesis_result, dict) else ""
        lines.append(summary or "这些异常需要结合参考范围、症状和后续复查结果一起综合判断。")
        for finding in (synthesis_result.get("combined_findings", []) if isinstance(synthesis_result, dict) else [])[:4]:
            lines.append(f"- {finding}")
        lines.extend(["", "### 后续建议"])
        next_steps = synthesis_result.get("next_steps", []) if isinstance(synthesis_result, dict) else []
        if next_steps:
            lines.extend([f"- {step}" for step in next_steps[:4]])
        else:
            lines.append("- 建议先结合原始报告参考范围、采血条件和近期症状一起判断。")
            lines.append("- 如异常持续存在，可到相应专科进一步评估。")
        return "\n".join(lines).strip()

    def _analyze_input(self, message: str, has_report: bool, conversation_history: list[dict[str, str]]) -> InputAnalysisResult:
        rule_intent = routing_service.route(message, has_report)
        stripped = self._strip_question_tail(message)
        history = conversation_history if rule_intent != "report_follow_up" else ([] if self._should_ignore_report_history(message) else conversation_history[-4:])
        fallback = InputAnalysisResult(
            intent=rule_intent,
            rewritten_query=stripped,
            normalized_term=stripped if rule_intent == "term_explanation" else "",
            use_local_knowledge=True,
            use_who=rule_intent == "term_explanation",
            use_max=rule_intent != "collect_more_info",
            reason="rule_fallback",
        )
        if not llm_service.is_configured:
            return fallback
        try:
            payload = llm_service.chat_json_fast(
                input_analysis_system_prompt(),
                input_analysis_user_prompt(message, has_report, history),
            )
            parsed = InputAnalysisResult.model_validate(payload)
            if parsed.intent not in {"report_follow_up", "term_explanation", "symptom_rag_advice", "collect_more_info"}:
                return fallback
            if not parsed.rewritten_query:
                parsed.rewritten_query = fallback.rewritten_query
            if parsed.intent == "term_explanation" and not parsed.normalized_term:
                parsed.normalized_term = parsed.rewritten_query or fallback.normalized_term
            return parsed
        except Exception:
            logger.exception("Input analysis failed")
            return fallback

    def _answer_system_prompt(self, intent: str) -> str:
        return term_explanation_system_prompt() if intent == "term_explanation" else answer_composer_system_prompt()

    def _repair_incomplete_answer(
        self,
        intent: str,
        message: str,
        partial_answer: str,
        tool_outputs: list[dict[str, Any]],
        citations: list[Citation],
        used_tools: list[str],
    ) -> str:
        if intent == "report_follow_up":
            return self._build_report_follow_up_answer(tool_outputs)
        if llm_service.is_configured and partial_answer:
            try:
                repaired = llm_service.chat_text_max(
                    answer_repair_system_prompt(),
                    answer_repair_user_prompt(
                        intent=intent,
                        message=message,
                        partial_answer=partial_answer,
                        tool_outputs=tool_outputs,
                        citations=[citation.model_dump(mode="json") for citation in citations],
                    ),
                ).strip()
                if repaired and not self._is_incomplete_answer(repaired):
                    return self._append_source_marker(intent, used_tools, repaired)
            except Exception:
                logger.exception("Answer repair failed")
        return self._fallback_answer(intent, tool_outputs, citations, used_tools)

    def _fallback_answer(self, intent: str, tool_outputs: list[dict[str, Any]], citations: list[Citation], used_tools: list[str]) -> str:
        if intent == "collect_more_info":
            return "目前信息还不够，我需要先补充几个关键点，才能更准确地解释。"
        if intent == "report_follow_up":
            return self._build_report_follow_up_answer(tool_outputs) or "我已经读取当前报告，但还没有拿到足够清晰的指标解释。"
        if intent == "term_explanation":
            return self._fallback_term_explanation(tool_outputs, used_tools)
        if citations:
            lines = [citations[0].snippet]
            if len(citations) > 1:
                lines.extend(["", "### 参考信息"])
                lines.extend([f"- {citation.title}：{citation.snippet}" for citation in citations[:3]])
            return "\n".join(lines)
        return "当前本地知识库里没有找到足够相关的内容，建议你补充更具体的问题描述。"

    def _fallback_term_explanation(self, tool_outputs: list[dict[str, Any]], used_tools: list[str]) -> str:
        who_matches: list[dict[str, Any]] = []
        docs: list[dict[str, Any]] = []
        for item in tool_outputs:
            result = item.get("result", {})
            if item.get("tool") == "lookup_icd11" and isinstance(result, dict):
                who_matches = result.get("matches", [])
            if item.get("tool") == "retrieve_knowledge" and isinstance(result, dict):
                docs = result.get("docs", [])
        if who_matches:
            primary = who_matches[0]
            title = str(primary.get("title") or "该名词").strip()
            definition = str(primary.get("definition") or "").strip()
            code = str(primary.get("code") or "").strip()
            feature_text = str(docs[0].get("detail", "")).strip()[:220] if docs else ""
            answer = "\n".join([
                "### 它是什么",
                f"{title}：{definition}" if definition else f"{title}是 WHO ICD-11 中可检索到的相关医学名词。",
                "",
                "### 常见表现或特点",
                feature_text or "常见表现会因疾病类型、严重程度和受累部位不同而变化。",
                "",
                "### 常见诱因或易感因素",
                "是否容易发生，通常与基础疾病、局部环境、免疫状态、暴露史和生活方式等因素有关，需要结合具体情况判断。",
                "",
                "### 什么时候需要就医",
                "如果症状持续存在、逐渐加重，或者已经影响日常生活，建议到相关专科进一步评估。",
                "",
                "### 温馨提示",
                f"WHO ICD-11 编码：{code}。" if code else "WHO ICD-11 可作为术语标准化参考。",
            ])
            return self._append_source_marker("term_explanation", used_tools, answer)
        if docs:
            primary = docs[0]
            title = str(primary.get("title") or "该术语").strip()
            snippet = str(primary.get("snippet") or "").strip()
            detail = str(primary.get("detail") or "").strip()
            return "\n".join([
                "### 它是什么",
                f"{title}一般是指：{snippet}" if snippet else f"{title}是一个常见医学名词，需要结合具体情境理解。",
                "",
                "### 常见表现或特点",
                detail[:220] if detail else "常见表现会因具体类型和严重程度不同而变化，建议结合症状持续时间和范围一起判断。",
                "",
                "### 常见诱因或易感因素",
                "常见诱因可能与局部环境、个人卫生、基础疾病、免疫状态或长期反复刺激有关，需要结合具体场景判断。",
                "",
                "### 什么时候需要就医",
                "如果症状持续存在、反复加重，或者已经影响日常生活，建议到相关专科进一步评估。",
                "",
                "### 温馨提示",
                "网络科普只能帮助理解术语，不能替代线下面诊和个体化诊断。",
            ])
        return "\n".join([
            "### 它是什么",
            "这是一个医学相关概念，但当前本地知识库和标准术语结果都不足以支持更具体的解释。",
            "",
            "### 常见表现或特点",
            "不同疾病或指标在临床上的表现差异较大，需要结合实际症状和检查结果判断。",
            "",
            "### 常见诱因或易感因素",
            "诱因通常与体质、基础疾病、环境暴露和生活方式等因素有关。",
            "",
            "### 什么时候需要就医",
            "如果你已经出现明确不适，或者这个问题和现有检查异常有关，建议到相关专科进一步咨询。",
            "",
            "### 温馨提示",
            "如果你愿意，可以继续告诉我这个名词出现在什么场景里，或者你最关心的具体点。",
        ])

    def _append_source_marker(self, intent: str, used_tools: list[str], answer: str) -> str:
        normalized = answer.strip()
        if intent == "term_explanation" and "lookup_icd11" in used_tools and WHO_SOURCE_SUFFIX not in normalized:
            normalized = f"{normalized}\n\n{WHO_SOURCE_SUFFIX}"
        return normalized

    def _build_agent_response(self, session_id: str, execution: AgentExecutionResult) -> AgentResponse:
        return AgentResponse(
            session_id=session_id,
            intent=execution.intent,  # type: ignore[arg-type]
            answer=self._with_safety_appendix(execution.answer),
            citations=self._dedupe_citations(execution.citations),
            used_tools=execution.used_tools,
            follow_up_questions=execution.follow_up_questions or self._default_follow_up_questions(execution.intent),
            safety_level="safe",
            handoff_required=False,
            debug=AgentDebug.model_validate(execution.debug) if execution.debug else None,
        )

    def _with_safety_appendix(self, answer: str) -> str:
        stripped = self._strip_duplicate_appendix(answer)
        return f"{stripped}\n\n{DEFAULT_SAFETY_APPENDIX}" if stripped else DEFAULT_SAFETY_APPENDIX

    def _strip_duplicate_appendix(self, answer: str) -> str:
        cleaned = answer.strip()
        while DEFAULT_SAFETY_APPENDIX in cleaned:
            cleaned = cleaned.replace(DEFAULT_SAFETY_APPENDIX, "").strip()
        return cleaned

    def _is_incomplete_answer(self, answer: str) -> bool:
        text = self._strip_duplicate_appendix(answer).strip()
        if not text:
            return True
        raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not raw_lines:
            return True
        for line in raw_lines:
            if re.search(r"(包括|如下|提示)[:：]$", line):
                return True
        normalized_lines = [self._normalize_heading_line(line) for line in raw_lines]
        known_headings = TERM_HEADINGS | REPORT_HEADINGS
        if all(line in known_headings for line in normalized_lines):
            return True
        for index, line in enumerate(normalized_lines):
            if line in known_headings and (index == len(normalized_lines) - 1 or normalized_lines[index + 1] in known_headings):
                return True
        content_lines = [line for line in normalized_lines if line not in known_headings]
        return len(content_lines) <= 1 and any(line in known_headings for line in normalized_lines)

    def _normalize_heading_line(self, line: str) -> str:
        stripped = HEADING_PREFIX_PATTERN.sub("", line.strip())
        return stripped.strip(":： ").replace("**", "")

    def _default_follow_up_questions(self, intent: str) -> list[str]:
        if intent == "report_follow_up":
            return ["如果你愿意，我可以继续解释某一项异常指标为什么值得关注。"]
        if intent == "term_explanation":
            return ["如果你愿意，我可以继续解释它和哪些检查或症状有关。"]
        if intent == "symptom_rag_advice":
            return ["如果方便，请补充症状持续时间、严重程度，以及是否伴有发热、疼痛或体重变化。"]
        return []

    def _report_follow_up_questions(self, plan: ReportFollowUpPlan) -> list[str]:
        if plan.follow_up_needed:
            return ["如果方便，请告诉我你最担心的是哪一项指标，或者是否有近期不适和复查结果。"]
        if plan.need_next_steps:
            return ["如果你愿意，我可以继续说明这些异常下一步一般怎么复查、关注哪些生活方式因素。"]
        return self._default_follow_up_questions("report_follow_up")

    def _cache_key(self, session_id: str, report_id: str | None, message: str) -> str:
        normalized = re.sub(r"\s+", " ", message.strip())
        return f"{session_id}::{report_id or '-'}::{normalized}"

    def _strip_question_tail(self, message: str) -> str:
        stripped = message.strip().rstrip("：:。？！!?")
        stripped = QUESTION_TAIL_PATTERN.sub("", stripped).strip()
        return stripped or message.strip()

    def _normalize_query(self, query: str) -> str:
        return re.sub(r"\s+", " ", query.strip("：:。？！!? ")).strip()

    def _should_ignore_report_history(self, message: str) -> bool:
        text = message.strip()
        broad_keywords = ("报告", "体检", "指标", "结果", "分析", "异常")
        follow_keywords = ("这个", "那个", "这项", "那项", "它", "前一个", "上一个", "刚才")
        return any(keyword in text for keyword in broad_keywords) and not any(keyword in text for keyword in follow_keywords)

    def _chunk_text(self, text: str) -> list[str]:
        normalized = text.strip()
        if not normalized:
            return []
        parts = [part.strip() for part in normalized.split("\n\n") if part.strip()]
        if len(parts) <= 1:
            return [normalized]
        return [part + ("\n\n" if index < len(parts) - 1 else "") for index, part in enumerate(parts)]

    def _tool_status_label(self, tool: str) -> str:
        return {
            "search_report_items": "读取报告内容中",
            "interpret_lab": "解释异常指标中",
            "retrieve_knowledge": "检索本地知识中",
            "lookup_icd11": "检索 WHO ICD-11 中",
            "query_drug": "整理药物科普信息中",
        }.get(tool, f"执行工具中：{tool}")

    def _recent_conversation_context(self, session: Session, session_id: str) -> list[dict[str, str]]:
        limit = self.settings.short_term_context_turns * 2 + 1
        messages = session.exec(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.desc())
            .limit(limit)
        ).all()
        messages.reverse()
        return [{"role": message.role, "content": message.content} for message in messages]

    def _get_or_create_session(self, session: Session, session_id: str | None, report_id: str | None, message: str) -> ChatSession:
        if session_id:
            existing = session.get(ChatSession, session_id)
            if existing:
                return existing
        chat_session = ChatSession(report_id=report_id, title=self._strip_question_tail(message)[:20] or "健康咨询")
        session.add(chat_session)
        session.commit()
        session.refresh(chat_session)
        return chat_session

    def _store_message(
        self,
        session: Session,
        session_id: str,
        role: str,
        content: str,
        intent: str | None = None,
        safety_level: str = "safe",
        citations: list[Citation] | None = None,
    ) -> None:
        message = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            intent=intent,
            safety_level=safety_level,
            citations_json=json.dumps([citation.model_dump(mode="json") for citation in citations or []], ensure_ascii=False),
        )
        session.add(message)
        session.commit()

    def _knowledge_citations(self, docs: list[Any]) -> list[Citation]:
        return [
            Citation(
                source_type="knowledge_doc",
                doc_id=str(getattr(doc, "id")),
                title=str(getattr(doc, "title")),
                url=str(getattr(doc, "url")),
                trust_tier=str(getattr(doc, "trust_tier")),
                snippet=str(getattr(doc, "snippet")),
            )
            for doc in docs
        ]

    def _who_citations(self, who_result: dict[str, Any]) -> list[Citation]:
        citations: list[Citation] = []
        for match in who_result.get("matches", []):
            identifier = str(match.get("code") or match.get("uri") or match.get("title") or "who")
            citations.append(
                Citation(
                    source_type="who_icd11",
                    doc_id=identifier,
                    title=str(match.get("title") or "WHO ICD-11"),
                    url=str(match.get("uri") or "https://icd.who.int/icdapi"),
                    trust_tier="A",
                    snippet=str(match.get("definition") or match.get("title") or "WHO ICD-11 术语结果"),
                )
            )
        return citations

    def _dedupe_citations(self, citations: list[Citation]) -> list[Citation]:
        deduped: list[Citation] = []
        seen: set[tuple[str, str, str]] = set()
        for citation in citations:
            key = (citation.source_type, citation.doc_id, citation.title)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(citation)
        return deduped


react_agent_service = ReactAgentService()
