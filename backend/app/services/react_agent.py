from __future__ import annotations

import json
from pathlib import Path

from sqlmodel import Session

from app.core.schemas import AgentResponse, Citation, PlannerOutput
from app.models.entities import ChatMessage, ChatSession
from app.services.knowledge_service import knowledge_service
from app.services.llm import llm_service
from app.services.report_service import report_service
from app.services.routing_service import routing_service
from app.services.safety_service import DEFAULT_SAFETY_APPENDIX, safety_service
from app.services.summary_service import summary_service
from app.services.who_service import who_service


class ReactAgentService:
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

        safety = safety_service.evaluate(message)
        if safety.handoff_required:
            answer = (
                f"{safety.reason} "
                "\u8be5\u95ee\u9898\u4e0d\u9002\u5408\u7531\u7cfb\u7edf\u76f4\u63a5\u7ed9\u51fa\u8bca\u65ad\u3001\u5904\u65b9\u6216\u6025\u75c7\u5904\u7f6e\u5efa\u8bae\u3002"
                "\u8bf7\u5c3d\u5feb\u8054\u7cfb\u533b\u751f\u6216\u7ebf\u4e0b\u533b\u7597\u673a\u6784\u3002"
            )
            response = AgentResponse(
                session_id=chat_session.id,
                intent="safety_handoff",
                answer=f"{answer}\n\n{DEFAULT_SAFETY_APPENDIX}",
                citations=[],
                used_tools=[],
                follow_up_questions=[
                    "\u5982\u679c\u65b9\u4fbf\uff0c\u8bf7\u8bf4\u660e\u5f53\u524d\u6700\u4e3b\u8981\u7684\u4e0d\u9002\u75c7\u72b6\u548c\u6301\u7eed\u65f6\u95f4\u3002"
                ],
                safety_level="handoff",
                handoff_required=True,
            )
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

        intent = routing_service.route(message, report_id is not None)
        plan = self._plan(intent, message, report_id)
        tool_outputs, citations = self._execute_plan(session, report_id, chat_session.id, plan, output_dir)
        answer = self._compose_answer(intent, message, tool_outputs, citations)
        response = AgentResponse(
            session_id=chat_session.id,
            intent=intent,
            answer=f"{answer}\n\n{DEFAULT_SAFETY_APPENDIX}",
            citations=citations,
            used_tools=[item["tool"] for item in tool_outputs],
            follow_up_questions=self._follow_up_questions(intent, report_id is not None),
            safety_level="safe",
            handoff_required=False,
        )
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

    def _plan(self, intent: str, message: str, report_id: str | None) -> PlannerOutput:
        fallback = {
            "report_follow_up": {
                "thoughts": "Read report abnormalities, then retrieve medical references.",
                "actions": [
                    {"tool": "search_report_items", "args": {"message": message}},
                    {"tool": "retrieve_knowledge", "args": {"query": message}},
                ],
            },
            "term_explanation": {
                "thoughts": "Retrieve background knowledge for the term first.",
                "actions": [{"tool": "retrieve_knowledge", "args": {"query": message}}],
            },
            "symptom_rag_advice": {
                "thoughts": "Retrieve symptom guidance and department suggestions.",
                "actions": [
                    {"tool": "search_symptoms", "args": {"query": message}},
                    {"tool": "retrieve_knowledge", "args": {"query": message}},
                ],
            },
            "collect_more_info": {
                "thoughts": "Ask for clarification because the context is insufficient.",
                "actions": [],
            },
        }

        if llm_service.is_configured:
            try:
                payload = llm_service.chat_json(
                    system_prompt=(
                        "You are a constrained planner for a medical consultation agent. "
                        "Return JSON only with fields thoughts and actions. "
                        "Use at most four actions. Tools available: "
                        "search_report_items, interpret_lab, search_symptoms, query_drug, "
                        "retrieve_knowledge, lookup_icd11, generate_health_summary."
                    ),
                    user_prompt=json.dumps(
                        {"intent": intent, "message": message, "has_report": bool(report_id)},
                        ensure_ascii=False,
                    ),
                )
                return PlannerOutput.model_validate(payload)
            except Exception:
                pass

        return PlannerOutput.model_validate(fallback.get(intent, fallback["term_explanation"]))

    def _execute_plan(
        self,
        session: Session,
        report_id: str | None,
        chat_session_id: str,
        plan: PlannerOutput,
        output_dir: Path,
    ) -> tuple[list[dict], list[Citation]]:
        outputs: list[dict] = []
        citations: list[Citation] = []
        for action in plan.actions[:4]:
            if action.tool == "search_report_items" and report_id:
                report = report_service.get_report(session, report_id)
                outputs.append({"tool": action.tool, "result": [item.model_dump() for item in report.abnormal_items[:10]]})
            elif action.tool == "interpret_lab" and report_id:
                report = report_service.get_report(session, report_id)
                outputs.append({"tool": action.tool, "result": [item.model_dump() for item in report.items[:5]]})
            elif action.tool in {"search_symptoms", "query_drug", "retrieve_knowledge"}:
                query = action.args.get("query") or action.args.get("message") or ""
                docs = knowledge_service.retrieve(session, query)
                outputs.append({"tool": action.tool, "result": [doc.title for doc in docs]})
                citations.extend(
                    Citation(
                        source_type="knowledge_doc",
                        doc_id=doc.id,
                        title=doc.title,
                        url=doc.url,
                        trust_tier=doc.trust_tier,  # type: ignore[arg-type]
                        snippet=doc.snippet,
                    )
                    for doc in docs
                )
            elif action.tool == "lookup_icd11":
                query = action.args.get("query") or action.args.get("disease") or ""
                outputs.append({"tool": action.tool, "result": who_service.search(query)})
            elif action.tool == "generate_health_summary" and report_id:
                summary = summary_service.generate(session, report_id, chat_session_id, output_dir)
                outputs.append({"tool": action.tool, "result": summary.model_dump()})
        return outputs, self._dedupe_citations(citations)

    def _compose_answer(self, intent: str, message: str, tool_outputs: list[dict], citations: list[Citation]) -> str:
        if intent == "collect_more_info":
            return (
                "\u76ee\u524d\u4fe1\u606f\u4e0d\u8db3\u4ee5\u7ed9\u51fa\u53ef\u9760\u7684\u65b9\u5411\u5224\u65ad\u3002"
                "\u8bf7\u8865\u5145\u4e3b\u8981\u5f02\u5e38\u6307\u6807\u3001\u75c7\u72b6\u6301\u7eed\u65f6\u95f4\uff0c"
                "\u6216\u76f4\u63a5\u4e0a\u4f20\u4f53\u68c0/\u68c0\u9a8c\u62a5\u544a\u3002"
            )

        if llm_service.is_configured:
            try:
                answer = llm_service.chat_text(
                    system_prompt=(
                        "Answer in Chinese. Do not diagnose, prescribe, or give dosage advice. "
                        "If only low-trust sources are available, explicitly say they are low trust."
                    ),
                    user_prompt=json.dumps(
                        {
                            "intent": intent,
                            "message": message,
                            "tool_outputs": tool_outputs,
                            "citations": [citation.model_dump() for citation in citations],
                        },
                        ensure_ascii=False,
                    ),
                )
                if answer.strip():
                    return answer.strip()
            except Exception:
                pass

        return self._fallback_answer(intent, tool_outputs, citations)

    def _fallback_answer(self, intent: str, tool_outputs: list[dict], citations: list[Citation]) -> str:
        if intent == "report_follow_up":
            report_output = next((item["result"] for item in tool_outputs if item["tool"] == "search_report_items"), [])
            if report_output:
                summary = "\uff1b".join(
                    f"{item['name']} {item['value_raw']}{item.get('unit', '')}\uff08{item['status']}\uff09"
                    for item in report_output[:5]
                )
                return (
                    f"\u4ece\u5f53\u524d\u62a5\u544a\u4e2d\u4f18\u5148\u5173\u6ce8\u8fd9\u4e9b\u5f02\u5e38\u9879\uff1a{summary}\u3002"
                    "\u5efa\u8bae\u7ed3\u5408\u539f\u59cb\u62a5\u544a\u91cc\u7684\u53c2\u8003\u8303\u56f4\u548c\u533b\u751f\u610f\u89c1\u51b3\u5b9a\u662f\u5426\u590d\u67e5\u3002"
                )
            return (
                "\u5df2\u6536\u5230\u4e0e\u62a5\u544a\u76f8\u5173\u7684\u95ee\u9898\uff0c"
                "\u4f46\u5f53\u524d\u672a\u63d0\u53d6\u5230\u660e\u786e\u5f02\u5e38\u9879\uff0c\u5efa\u8bae\u5148\u786e\u8ba4\u4e0a\u4f20\u62a5\u544a\u662f\u5426\u6e05\u6670\u3002"
            )
        if intent == "symptom_rag_advice":
            titles = [citation.title for citation in citations[:3]]
            source_text = "\uff1b".join(titles) if titles else "\u5f53\u524d\u77e5\u8bc6\u5e93\u6682\u65f6\u65e0\u8db3\u591f\u6761\u76ee"
            return (
                "\u6839\u636e\u77e5\u8bc6\u5e93\u68c0\u7d22\uff0c\u5efa\u8bae\u5148\u6309\u75c7\u72b6\u4e25\u91cd\u7a0b\u5ea6"
                "\u8bc4\u4f30\u5c31\u8bca\u7d27\u8feb\u6027\uff0c\u5e76\u4f18\u5148\u54a8\u8be2\u5168\u79d1\u6216\u76f8\u5173\u4e13\u79d1\u3002"
                f"\u53ef\u53c2\u8003\uff1a{source_text}\u3002"
            )
        if intent == "term_explanation":
            titles = [citation.title for citation in citations[:2]]
            source_text = "\uff1b".join(titles) if titles else "\u77e5\u8bc6\u5e93\u7ed3\u679c\u4e0d\u8db3"
            return f"\u8fd9\u4e2a\u95ee\u9898\u66f4\u9002\u5408\u505a\u672f\u8bed\u548c\u80cc\u666f\u89e3\u91ca\u3002\u5f53\u524d\u53ef\u53c2\u8003\uff1a{source_text}\u3002"
        return (
            "\u6211\u5df2\u7ed3\u5408\u5f53\u524d\u4e0a\u4e0b\u6587\u5b8c\u6210\u521d\u6b65\u5206\u6790\uff0c"
            "\u4f46\u5efa\u8bae\u4f60\u7ee7\u7eed\u8865\u5145\u62a5\u544a\u622a\u56fe\u3001\u5f02\u5e38\u6307\u6807\u6216\u75c7\u72b6\u7ec6\u8282\u3002"
        )

    def _follow_up_questions(self, intent: str, has_report: bool) -> list[str]:
        if intent == "collect_more_info":
            return [
                "\u8bf7\u8865\u5145\u6700\u5173\u6ce8\u7684\u5f02\u5e38\u6307\u6807\u3001\u5f53\u524d\u75c7\u72b6\u548c\u6301\u7eed\u65f6\u95f4\u3002"
            ]
        if has_report:
            return [
                "\u5982\u679c\u65b9\u4fbf\uff0c\u8bf7\u7ee7\u7eed\u8bf4\u660e\u4f60\u6700\u62c5\u5fc3\u7684\u662f\u54ea\u4e00\u9879\u6307\u6807\uff0c\u4ee5\u53ca\u662f\u5426\u5df2\u6709\u590d\u67e5\u7ed3\u679c\u3002"
            ]
        return [
            "\u5982\u679c\u65b9\u4fbf\uff0c\u8bf7\u8865\u5145\u5e74\u9f84\u3001\u4e3b\u8981\u75c7\u72b6\u548c\u6301\u7eed\u65f6\u95f4\uff0c\u6211\u53ef\u4ee5\u5e2e\u4f60\u7f29\u5c0f\u5173\u6ce8\u65b9\u5411\u3002"
        ]

    def _get_or_create_session(
        self,
        session: Session,
        session_id: str | None,
        report_id: str | None,
        message: str,
    ) -> ChatSession:
        if session_id:
            existing = session.get(ChatSession, session_id)
            if existing:
                return existing
        chat_session = ChatSession(report_id=report_id, title=message[:20] or "\u5065\u5eb7\u54a8\u8be2")
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
            citations_json=json.dumps([citation.model_dump() for citation in citations or []], ensure_ascii=False),
        )
        session.add(message)
        session.commit()

    def _dedupe_citations(self, citations: list[Citation]) -> list[Citation]:
        seen: set[str] = set()
        deduped: list[Citation] = []
        for citation in citations:
            if citation.doc_id in seen:
                continue
            seen.add(citation.doc_id)
            deduped.append(citation)
        return deduped


react_agent_service = ReactAgentService()
