from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Iterator

from pydantic import BaseModel, Field
from sqlmodel import Session, select

from app.agent_graph.graph import build_entry_graph
from app.core.config import get_settings
from app.core.schemas import AgentDebug, AgentResponse, Citation
from app.models.entities import ChatMessage, ChatSession
from app.services.cache_service import cache_service
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
from app.services.session_service import session_service
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
    """一次 Agent 执行的中间结果。

    这是内部对象，不直接暴露给前端。
    它负责把后续生成答案还会用到的材料都串在一起。
    """

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
    """批量指标解释模型的结构化输出。"""

    items: list[dict[str, Any]] = Field(default_factory=list)


class InputAnalysisResult(BaseModel):
    """输入分析阶段的输出。"""

    intent: str = "collect_more_info"
    rewritten_query: str = ""
    normalized_term: str = ""
    use_local_knowledge: bool = True
    use_who: bool = False
    use_max: bool = True
    reason: str = ""


class ReportFollowUpPlan(BaseModel):
    """报告追问拆解计划。"""

    focus_item_names: list[str] = Field(default_factory=list)
    need_item_explanations: bool = True
    need_synthesis: bool = True
    need_next_steps: bool = True
    synthesis_axes: list[str] = Field(default_factory=list)
    follow_up_needed: bool = False
    reason: str = ""


class ReportSynthesisResult(BaseModel):
    """综合异常层的中间结果。"""

    summary: str = ""
    priority_axes: list[str] = Field(default_factory=list)
    combined_findings: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)


class AgentEntryResult(BaseModel):
    """LangGraph 鍏ュ彛鍥剧殑绠€鍖栬緭鍑恒€?
    杩欏眰鍙礋璐ｆ槸鍚﹂渶瑕佺珛鍗崇粨鏉熶互鍙婅繖杞殑鎰忓浘鍒嗘祦锛?
    鍥炵瓟鐢熸垚銆佺紦瀛樺拰钀藉簱浠嶇劧娌跨敤鍘熸湁閫昏緫锛岄伩鍏嶄竴娆℃€ф敼鍔ㄨ繃澶с€?
    """

    conversation_history: list[dict[str, str]] = Field(default_factory=list)
    analysis: InputAnalysisResult | None = None
    immediate_response: AgentResponse | None = None
    execution: AgentExecutionResult | None = None


class ReactAgentService:
    """整个健康咨询 Agent 的主协调器。"""

    def __init__(self) -> None:
        """初始化配置和内存级回答缓存。"""
        self.settings = get_settings()
        # 简单内存缓存：避免同一问题在短时间内重复走整条 Agent 链。
        self.answer_cache: dict[str, AgentExecutionResult] = {}

        self.entry_graph = build_entry_graph(self)

    def respond(
        self,
        session: Session,
        session_id: str | None,
        report_id: str | None,
        message: str,
        output_dir: Path,
    ) -> AgentResponse:
        """同步回答入口。"""
        # 主流程：
        # 1. 找到或创建会话
        # 2. 先落用户消息
        # 3. 处理安全拦截、报告未完成等即时返回场景
        # 4. 命中缓存则直接复用
        # 5. 否则执行完整 Agent 主链
        # 6. 落助手消息并返回
        # 流式接口和同步接口使用同一条业务链，只是输出方式不同。
        # 这里会把执行过程拆成 session / status / delta / final 四类事件。
        # 流式接口和同步接口使用同一条业务链，只是输出方式不同。
        # 这里会把执行过程拆成 session / status / delta / final 四类事件。
        chat_session = self._get_or_create_session(session, session_id, report_id, message)
        # 如果这是一个空白新会话的首轮提问，就把标题自动改成问题摘要，
        # 这样左侧会话列表不会永远显示“新对话”。
        session_service.auto_title_if_needed(session, chat_session.id, message)
        self._store_message(session, chat_session.id, "user", message)
        # 鍏ュ彛鍥惧厛鍋氫笂涓嬫枃鍔犺浇銆佸畨鍏ㄦ妤€鏌ャ€佹姤鍛婄姸鎬佸垽鏂拰鎰忓浘鍒嗘祦銆?
        entry_result = self._run_entry_graph(session, chat_session.id, report_id, message)
        if entry_result.immediate_response:
            self._store_message(
                session,
                chat_session.id,
                "assistant",
                entry_result.immediate_response.answer,
                entry_result.immediate_response.intent,
                "handoff" if entry_result.immediate_response.handoff_required else "safe",
                entry_result.immediate_response.citations,
            )
            return entry_result.immediate_response

        cache_key = self._cache_key(chat_session.id, report_id, message)
        cached_response = self._load_cached_response(session, cache_key, chat_session.id)
        if cached_response:
            self._store_message(
                session,
                chat_session.id,
                "assistant",
                cached_response.answer,
                cached_response.intent,
                cached_response.safety_level,
                cached_response.citations,
            )
            return cached_response
        execution = self.answer_cache.get(cache_key)
        if not isinstance(execution, AgentExecutionResult):
            cache_service.delete_agent_response(session, cache_key)
            execution = None
        if execution and self._is_incomplete_answer(self._strip_duplicate_appendix(execution.answer)):
            # 如果缓存里已经是一条半成品答案，宁可丢掉重算。
            self.answer_cache.pop(cache_key, None)
            execution = None

        if not execution:
            execution = entry_result.execution or self._prepare_fast_path(
                session,
                chat_session.id,
                report_id,
                message,
                conversation_history=entry_result.conversation_history,
                analysis=entry_result.analysis,
            )
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
        self._save_cached_response(session, cache_key, report_id, message, response)
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
        """流式回答入口。

        和 `respond()` 的主要区别在于它会把执行过程拆成多个 SSE 事件推给前端。
        """
        chat_session = self._get_or_create_session(session, session_id, report_id, message)
        # 流式接口与同步接口共享同一套自动命名规则，保证会话列表表现一致。
        session_service.auto_title_if_needed(session, chat_session.id, message)
        self._store_message(session, chat_session.id, "user", message)
        yield {"event": "session", "data": {"session_id": chat_session.id}}

        entry_result = self._run_entry_graph(session, chat_session.id, report_id, message)
        if entry_result.immediate_response:
            yield {"event": "status", "data": {"label": "已完成安全检查"}}
            for chunk in self._chunk_text(entry_result.immediate_response.answer):
                yield {"event": "delta", "data": {"text": chunk}}
            self._store_message(
                session,
                chat_session.id,
                "assistant",
                entry_result.immediate_response.answer,
                entry_result.immediate_response.intent,
                "handoff" if entry_result.immediate_response.handoff_required else "safe",
                entry_result.immediate_response.citations,
            )
            yield {"event": "final", "data": entry_result.immediate_response.model_dump(mode="json")}
            return

        yield {"event": "status", "data": {"label": "分析问题中"}}
        cache_key = self._cache_key(chat_session.id, report_id, message)
        cached_response = self._load_cached_response(session, cache_key, chat_session.id)
        if cached_response:
            yield {"event": "status", "data": {"label": "命中缓存，正在返回历史结果"}}
            self._store_message(
                session,
                chat_session.id,
                "assistant",
                cached_response.answer,
                cached_response.intent,
                cached_response.safety_level,
                cached_response.citations,
            )
            for chunk in self._chunk_text(cached_response.answer):
                yield {"event": "delta", "data": {"text": chunk}}
            yield {"event": "final", "data": cached_response.model_dump(mode="json")}
            return

        execution = entry_result.execution or self._prepare_fast_path(
            session,
            chat_session.id,
            report_id,
            message,
            conversation_history=entry_result.conversation_history,
            analysis=entry_result.analysis,
        )
        for tool in execution.used_tools:
            yield {"event": "status", "data": {"label": self._tool_status_label(tool)}}
        yield {"event": "status", "data": {"label": "生成回答中"}}

        # 娴佸紡鍦烘櫙涓嬮渶瑕佽竟鐢熸垚杈规帹閫乨elta锛屼絾缁撴潫鏃朵粛瑕佹敹鍙栧畬鏁寸瓟妗堢敤浜庣紦瀛樺拰钀藉簱銆?
        stream_generator = self._stream_compose_answer(
            intent=execution.intent,
            message=execution.message,
            conversation_history=execution.conversation_history,
            tool_outputs=execution.tool_outputs,
            citations=execution.citations,
            used_tools=execution.used_tools,
            use_max=execution.use_max,
        )
        while True:
            try:
                chunk = next(stream_generator)
            except StopIteration as stop:
                execution.answer = stop.value or ""
                break
            yield {"event": "delta", "data": {"text": chunk}}
        self.answer_cache[cache_key] = execution
        response = self._build_agent_response(chat_session.id, execution)
        self._save_cached_response(session, cache_key, report_id, message, response)
        self._store_message(
            session,
            chat_session.id,
            "assistant",
            response.answer,
            response.intent,
            response.safety_level,
            response.citations,
        )
        yield {"event": "final", "data": response.model_dump(mode="json")}

    def _run_entry_graph(
        self,
        session: Session,
        session_id: str,
        report_id: str | None,
        message: str,
    ) -> AgentEntryResult:
        """杩愯 LangGraph 鍏ュ彛鍥俱€?
        濡傛灉鍥惧湪褰撳墠鐜鎴栬姹傞噷鍑虹幇寮傚父锛屽氨绔嬪嵆閫€鍥炲埌鍘熸湁鍏ュ彛閫昏緫锛?
        閬垮厤鍥犱负鏋舵瀯鍗囩骇褰卞搷鍒板凡缁忔甯稿伐浣滅殑鍔熻兘銆?
        """
        conversation_history = self._recent_conversation_context(session, session_id)
        try:
            state = self.entry_graph.invoke(
                {
                    "session": session,
                    "chat_session_id": session_id,
                    "report_id": report_id,
                    "message": message,
                }
            )
            return AgentEntryResult(
                conversation_history=state.get("conversation_history") or conversation_history,
                analysis=state.get("analysis"),
                immediate_response=state.get("immediate_response"),
                execution=(
                    AgentExecutionResult.model_validate(state["execution_data"])
                    if state.get("execution_data")
                    else None
                ),
            )
        except Exception:
            logger.exception("Entry graph failed, fallback to legacy entry path")
            immediate = self._handle_immediate_response(session, session_id, report_id, message)
            if immediate:
                return AgentEntryResult(conversation_history=conversation_history, immediate_response=immediate)
            return AgentEntryResult(
                conversation_history=conversation_history,
                analysis=self._analyze_input(message, report_id is not None, conversation_history),
                execution=None,
            )

    def _build_safety_handoff_response(self, session_id: str, reason: str | None) -> AgentResponse:
        """鎶婂畨鍏ㄩ檷绾ф剰瑙佺粺涓€缁勮鎴愬彲鐩存帴杩斿洖鐨?AgentResponse銆?"""
        answer = f"{reason or '当前问题涉及高风险医疗决策。'}\n\n请尽快联系医生或线下医疗机构获取专业评估。"
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

    def _build_report_not_ready_response(self, session_id: str, parse_status: str) -> AgentResponse:
        """鎶婃姤鍛婃湭瀹屾垚鐨勫満鏅粺涓€缁勮锛岄伩鍏嶅澶勫垎鏀悇鑷啓涓€閬嶆枃妗堛€?"""
        answer = "报告还在后台解析中，等解析完成后我再为你解读。" if parse_status != "error" else "报告解析失败，建议重新上传更清晰的文件。"
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

    def _handle_immediate_response(
        self,
        session: Session,
        session_id: str,
        report_id: str | None,
        message: str,
    ) -> AgentResponse | None:
        """处理需要立即结束的分支，例如安全拦截或报告未解析完成。"""
        # 这里放的是“先于一切智能流程”的快速判断。
        # 一旦命中高风险或报告未完成，就不再继续后面的 Agent 推理。
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
        conversation_history: list[dict[str, str]] | None = None,
        analysis: InputAnalysisResult | None = None,
    ) -> AgentExecutionResult:
        """Agent 主链的前半段：先用结构化、可控、较便宜的步骤准备信息。"""
        # 先读取短期上下文，再基于上下文做输入分析。
        # 这一层的目标是准备材料，而不是直接生成答案。
        conversation_history = conversation_history or self._recent_conversation_context(session, chat_session_id)
        analysis = analysis or self._analyze_input(message, report_id is not None, conversation_history)
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

    def _build_collect_more_info_execution(
        self,
        conversation_history: list[dict[str, str]],
        message: str,
        analysis: InputAnalysisResult,
    ) -> AgentExecutionResult:
        """组装信息不足场景的结构化执行结果。"""
        return AgentExecutionResult(
            intent="collect_more_info",
            follow_up_questions=["请补充你最关注的是哪项指标、有没有不适，以及这些情况持续了多久。"],
            conversation_history=conversation_history,
            message=message,
            use_max=analysis.use_max,
            debug={"analysis": analysis.model_dump(mode="json")},
        )

    def _build_report_follow_up_execution(
        self,
        session: Session,
        report_id: str | None,
        conversation_history: list[dict[str, str]],
        message: str,
        analysis: InputAnalysisResult,
    ) -> AgentExecutionResult:
        """组装报告追问场景的结构化执行结果。"""
        if not report_id:
            return self._build_collect_more_info_execution(conversation_history, message, analysis)
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
                {
                    "tool": "search_report_items",
                    "result": {
                        "abnormal_items": [item.model_dump(mode="json") for item in report.abnormal_items[:8]],
                        "focus_items": focus_items,
                        "related_items": related_items,
                        "raw_text_excerpt": report.raw_text[:800],
                        "analysis_reason": analysis.reason,
                    },
                },
                {"tool": "report_follow_up_plan", "result": plan.model_dump(mode="json")},
                {"tool": "interpret_lab", "result": {"items": interpretations}},
                {"tool": "report_synthesis", "result": synthesis.model_dump(mode="json")},
            ],
            conversation_history=conversation_history,
            message=message,
            use_max=analysis.use_max,
            debug={
                "analysis": analysis.model_dump(mode="json"),
                "plan": plan.model_dump(mode="json"),
                "synthesis": synthesis.model_dump(mode="json"),
            },
        )

    def _build_retrieval_execution(
        self,
        session: Session,
        conversation_history: list[dict[str, str]],
        message: str,
        analysis: InputAnalysisResult,
    ) -> AgentExecutionResult:
        """组装普通知识检索场景的结构化执行结果。"""
        effective_query = analysis.rewritten_query or message
        tool_name = "query_drug" if any(token in effective_query for token in DRUG_KEYWORDS) else "retrieve_knowledge"
        docs = knowledge_service.retrieve(session, effective_query) if analysis.use_local_knowledge else []
        return AgentExecutionResult(
            intent=analysis.intent,
            citations=self._knowledge_citations(docs),
            used_tools=[tool_name],
            follow_up_questions=self._default_follow_up_questions(analysis.intent),
            tool_outputs=[
                {
                    "tool": tool_name,
                    "result": {
                        "query": effective_query,
                        "docs": knowledge_service.pack_docs(docs),
                        "analysis_reason": analysis.reason,
                    },
                }
            ],
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
        """让 fast 模型把宽泛的报告问题拆成可执行计划。"""
        # 先准备一个规则版 fallback，保证模型失败时报告链仍然能继续跑。
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
        """根据计划挑出本轮应该重点解释的指标。"""
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
        """在一批报告指标里，根据名字模糊匹配目标项。"""
        normalized_target = self._normalize_item_name(target_name)
        for item in items:
            item_name = str(item.get("name") or "")
            normalized_name = self._normalize_item_name(item_name)
            if normalized_name == normalized_target or (normalized_target and (normalized_target in normalized_name or normalized_name in normalized_target)):
                return item
        return None

    def _normalize_item_name(self, name: str) -> str:
        """把指标名标准化，方便做宽松匹配。"""
        return re.sub(r"[\s\\-_/()（）]+", "", name).lower()

    def _build_report_synthesis(
        self,
        message: str,
        plan: ReportFollowUpPlan,
        interpretations: list[dict[str, Any]],
        related_items: list[dict[str, Any]],
    ) -> ReportSynthesisResult:
        """把多个单项解释提升为综合异常层。"""
        # 如果模型失败，至少也要有一版“综合结论”可用。
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
        """当综合异常层模型失败时，用规则拼一版可用的综合结果。"""
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
        """处理医学名词解释。优先本地知识，其次 WHO，最后再由模型润色。"""
        # 术语解释优先用“改写后或标准化后的术语”去查资料，而不是死磕用户原话。
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
        """按多个候选查询词依次尝试 WHO，直到命中为止。"""
        candidates = self._build_who_queries(message, docs)
        for query in candidates:
            result = who_service.search(query)
            if result.get("matches"):
                result["query_candidates"] = candidates
                result["matched_query"] = query
                return result
        return {"query": message, "matched_query": None, "query_candidates": candidates, "matches": []}

    def _build_who_queries(self, message: str, docs: list[Any]) -> list[str]:
        """构建 WHO 查询候选词列表。"""
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
        """把一个候选查询词清洗并去重后塞进列表。"""
        normalized = self._normalize_query(query)
        if normalized and normalized not in candidates:
            candidates.append(normalized)

    def _interpret_lab_batch(
        self,
        session: Session,
        focus_items: list[dict[str, Any]],
        related_items: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[Citation]]:
        """批量解释多个指标。

        这里先命中本地知识，再用 fast 模型把这些知识整理成更适合后续消费的结构。
        """
        # 先查本地知识，再决定是否让 fast 模型做结构整理。
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
        """组织最终答案。

        报告问题优先从本地结构化素材拼草稿，再视情况让 max 模型润色；
        这样可以显著降低“模型只输出几个标题”的概率。
        """
        # 不同意图的最终组织方式不同，报告类问题和术语类问题不会共用完全相同的策略。
        # 报告类回答如果已经有本地结构化草稿，优先直接用它，
        # 不必把“是否完整”这件事完全交给大模型碰运气。
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

    def _stream_compose_answer(
        self,
        intent: str,
        message: str,
        conversation_history: list[dict[str, str]],
        tool_outputs: list[dict[str, Any]],
        citations: list[Citation],
        used_tools: list[str],
        use_max: bool,
    ) -> Iterator[str]:
        """涓撲緵 SSE 流式接口使用的答案生成器。
        它会边生成边产出 delta，并在结束时返回最终答案，方便沿用现有缓存和落库逻辑。
        """
        if intent == "report_follow_up":
            draft = self._build_report_follow_up_answer(tool_outputs)
            if llm_service.is_configured and use_max and draft:
                plan, synthesis = self._extract_report_debug_materials(tool_outputs)
                streamed_chunks: list[str] = []
                try:
                    for chunk in llm_service.chat_text_stream_max(
                        report_answer_polish_system_prompt(),
                        report_answer_polish_user_prompt(message=message, draft_answer=draft, plan=plan, synthesis=synthesis),
                    ):
                        if not chunk:
                            continue
                        streamed_chunks.append(chunk)
                        yield chunk
                    polished = "".join(streamed_chunks).strip()
                    if polished and not self._is_incomplete_answer(polished):
                        return polished
                except Exception:
                    logger.exception("Report answer stream polish failed")
            return draft or self._fallback_answer(intent, tool_outputs, citations, used_tools)

        if llm_service.is_configured and use_max:
            streamed_chunks: list[str] = []
            try:
                for chunk in llm_service.chat_text_stream_max(
                    self._answer_system_prompt(intent),
                    answer_composer_user_prompt(
                        intent=intent,
                        message=message,
                        conversation_history=conversation_history,
                        tool_outputs=tool_outputs,
                        citations=[citation.model_dump(mode="json") for citation in citations],
                    ),
                ):
                    if not chunk:
                        continue
                    streamed_chunks.append(chunk)
                    yield chunk
                answer = "".join(streamed_chunks).strip()
                if answer and not self._is_incomplete_answer(answer):
                    return self._append_source_marker(intent, used_tools, answer)
                repaired = self._repair_incomplete_answer(intent, message, answer, tool_outputs, citations, used_tools)
                if repaired:
                    return repaired
            except Exception:
                logger.exception("Answer stream composition failed")
        return self._fallback_answer(intent, tool_outputs, citations, used_tools)

    def _extract_report_debug_materials(self, tool_outputs: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
        """从工具输出中抽取 plan 和 synthesis，供调试或润色阶段使用。"""
        plan = next((item["result"] for item in tool_outputs if item["tool"] == "report_follow_up_plan"), {})
        synthesis = next((item["result"] for item in tool_outputs if item["tool"] == "report_synthesis"), {})
        return (plan if isinstance(plan, dict) else {}, synthesis if isinstance(synthesis, dict) else {})

    def _build_report_follow_up_answer(self, tool_outputs: list[dict[str, Any]]) -> str:
        """把报告解释素材直接组装成一版完整可用的答案。"""
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
        """输入分析：先判断是什么问题、该怎么改写、需不需要 WHO / max。"""
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
        """根据意图选择最终回答阶段要使用的 system prompt。"""
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
        """当模型输出半成品时，尝试再修一轮。"""
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
        """最后的本地兜底。目标不是最优雅，而是保证结果完整可读。"""
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
        """术语解释链的本地兜底版本。"""
        # 这里会尽量利用已经查到的 WHO / 本地知识结果手工拼接答案，
        # 避免在模型不可用时完全失去术语解释能力。
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
        """在需要时给术语解释追加 WHO 来源标记。"""
        normalized = answer.strip()
        if intent == "term_explanation" and "lookup_icd11" in used_tools and WHO_SOURCE_SUFFIX not in normalized:
            normalized = f"{normalized}\n\n{WHO_SOURCE_SUFFIX}"
        return normalized

    def _build_agent_response(self, session_id: str, execution: AgentExecutionResult) -> AgentResponse:
        """把内部执行结果转换成 API 返回结构。"""
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
        """给最终答案追加统一安全提示，并避免重复追加。"""
        stripped = self._strip_duplicate_appendix(answer)
        return f"{stripped}\n\n{DEFAULT_SAFETY_APPENDIX}" if stripped else DEFAULT_SAFETY_APPENDIX

    def _strip_duplicate_appendix(self, answer: str) -> str:
        """移除已经重复出现的默认安全提示。"""
        cleaned = answer.strip()
        while DEFAULT_SAFETY_APPENDIX in cleaned:
            cleaned = cleaned.replace(DEFAULT_SAFETY_APPENDIX, "").strip()
        return cleaned

    def _is_incomplete_answer(self, answer: str) -> bool:
        """识别明显的半成品回答。"""
        # 这里专门拦截常见失败模式：
        # - 只有标题没有正文
        # - 以“包括：”“如下：”等开放式短句结尾
        # - 内容太少但标题很多
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
        """把标题行去掉序号、Markdown 符号和结尾冒号。"""
        stripped = HEADING_PREFIX_PATTERN.sub("", line.strip())
        return stripped.strip(":： ").replace("**", "")

    def _default_follow_up_questions(self, intent: str) -> list[str]:
        """给不同意图准备一条默认追问建议。"""
        if intent == "report_follow_up":
            return ["如果你愿意，我可以继续解释某一项异常指标为什么值得关注。"]
        if intent == "term_explanation":
            return ["如果你愿意，我可以继续解释它和哪些检查或症状有关。"]
        if intent == "symptom_rag_advice":
            return ["如果方便，请补充症状持续时间、严重程度，以及是否伴有发热、疼痛或体重变化。"]
        return []

    def _report_follow_up_questions(self, plan: ReportFollowUpPlan) -> list[str]:
        """根据报告追问计划动态生成下一轮建议问题。"""
        if plan.follow_up_needed:
            return ["如果方便，请告诉我你最担心的是哪一项指标，或者是否有近期不适和复查结果。"]
        if plan.need_next_steps:
            return ["如果你愿意，我可以继续说明这些异常下一步一般怎么复查、关注哪些生活方式因素。"]
        return self._default_follow_up_questions("report_follow_up")

    def _load_cached_response(self, session: Session, cache_key: str, session_id: str) -> AgentResponse | None:
        """先查持久化缓存，再做完整性校验。"""
        cached = cache_service.load_agent_response(session, cache_key)
        if not cached:
            return None
        if self._is_incomplete_answer(self._strip_duplicate_appendix(cached.answer)):
            cache_service.delete_agent_response(session, cache_key)
            return None
        return cached.model_copy(update={"session_id": session_id})

    def _save_cached_response(
        self,
        session: Session,
        cache_key: str,
        report_id: str | None,
        message: str,
        response: AgentResponse,
    ) -> None:
        """只缓存完整结果，避免把半成品答案落库。"""
        if self._is_incomplete_answer(self._strip_duplicate_appendix(response.answer)):
            cache_service.delete_agent_response(session, cache_key)
            return
        cache_service.save_agent_response(
            session=session,
            cache_key=cache_key,
            report_id=report_id,
            normalized_message=re.sub(r"\s+", " ", message.strip()),
            response=response,
        )

    def _cache_key(self, session_id: str, report_id: str | None, message: str) -> str:
        """生成回答缓存键。"""
        normalized = re.sub(r"\s+", " ", message.strip())
        return f"{session_id}::{report_id or '-'}::{normalized}"

    def _strip_question_tail(self, message: str) -> str:
        """去掉问题尾部的口语化短语，便于检索和标准化。"""
        stripped = message.strip().rstrip("：:。？！!?")
        stripped = QUESTION_TAIL_PATTERN.sub("", stripped).strip()
        return stripped or message.strip()

    def _normalize_query(self, query: str) -> str:
        """对任意查询词做轻量清洗。"""
        return re.sub(r"\s+", " ", query.strip("：:。？！!? ")).strip()

    def _should_ignore_report_history(self, message: str) -> bool:
        """判断这轮报告问题是否应该弱化历史上下文影响。"""
        text = message.strip()
        broad_keywords = ("报告", "体检", "指标", "结果", "分析", "异常")
        follow_keywords = ("这个", "那个", "这项", "那项", "它", "前一个", "上一个", "刚才")
        return any(keyword in text for keyword in broad_keywords) and not any(keyword in text for keyword in follow_keywords)

    def _chunk_text(self, text: str) -> list[str]:
        """把最终答案按段落切块，供流式输出使用。"""
        normalized = text.strip()
        if not normalized:
            return []
        parts = [part.strip() for part in normalized.split("\n\n") if part.strip()]
        if len(parts) <= 1:
            return [normalized]
        return [part + ("\n\n" if index < len(parts) - 1 else "") for index, part in enumerate(parts)]

    def _tool_status_label(self, tool: str) -> str:
        """把内部工具名转换成前端可读的阶段说明。"""
        return {
            "search_report_items": "读取报告内容中",
            "interpret_lab": "解释异常指标中",
            "retrieve_knowledge": "检索本地知识中",
            "lookup_icd11": "检索 WHO ICD-11 中",
            "query_drug": "整理药物科普信息中",
        }.get(tool, f"执行工具中：{tool}")

    def _recent_conversation_context(self, session: Session, session_id: str) -> list[dict[str, str]]:
        """读取最近 N 条对话，作为短期上下文。"""
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
        """优先复用已有会话，否则新建一段会话。"""
        # 会话是多轮上下文的载体。
        # 没有 session_id 时就新建；有 session_id 时优先复用。
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
        """把一条用户或助手消息持久化到数据库。"""
        # 这里不只是存纯文本，还会把意图、安全等级和 citations 一起落库。
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
        """把知识文档对象转换成标准 Citation 列表。"""
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
        """把 WHO 查询结果转换成 Citation 列表。"""
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
        """按来源类型 + 文档 id + 标题去重引用。"""
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
