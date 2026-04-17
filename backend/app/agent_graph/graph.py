from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from app.agent_graph.state import AgentComposeGraphState, AgentEntryGraphState
from app.services.report_service import report_service
from app.services.safety_service import safety_service


REPORT_READY_STATUSES = {"parsed", "needs_review"}


def build_entry_graph(service: Any):
    """构建 Agent 主链的前半段图。

    第二阶段开始，这张图除了入口判断，还会直接产出结构化执行结果：
    1. 读取短期上下文
    2. 安全闸门
    3. 报告状态闸门
    4. 意图路由
    5. 各 intent 的准备节点
    """

    builder = StateGraph(AgentEntryGraphState)

    builder.add_node("load_context", lambda state: _load_context_node(service, state))
    builder.add_node("safety_gate", lambda state: _safety_gate_node(service, state))
    builder.add_node("report_gate", lambda state: _report_gate_node(service, state))
    builder.add_node("route_intent", lambda state: _route_intent_node(service, state))
    builder.add_node("prepare_collect_more_info", lambda state: _prepare_collect_more_info_node(service, state))
    builder.add_node("prepare_report_follow_up", lambda state: _prepare_report_follow_up_node(service, state))
    builder.add_node("prepare_term_explanation", lambda state: _prepare_term_explanation_node(service, state))
    builder.add_node("prepare_retrieval", lambda state: _prepare_retrieval_node(service, state))

    builder.add_edge(START, "load_context")
    builder.add_edge("load_context", "safety_gate")
    builder.add_conditional_edges(
        "safety_gate",
        _after_immediate_gate,
        {"stop": END, "continue": "report_gate"},
    )
    builder.add_conditional_edges(
        "report_gate",
        _after_immediate_gate,
        {"stop": END, "continue": "route_intent"},
    )
    builder.add_conditional_edges(
        "route_intent",
        _route_to_prepare_node,
        {
            "prepare_collect_more_info": "prepare_collect_more_info",
            "prepare_report_follow_up": "prepare_report_follow_up",
            "prepare_term_explanation": "prepare_term_explanation",
            "prepare_retrieval": "prepare_retrieval",
        },
    )
    builder.add_edge("prepare_collect_more_info", END)
    builder.add_edge("prepare_report_follow_up", END)
    builder.add_edge("prepare_term_explanation", END)
    builder.add_edge("prepare_retrieval", END)

    return builder.compile()


def build_compose_graph(service: Any):
    """构建 Agent 主链的答案生成图。"""
    builder = StateGraph(AgentComposeGraphState)
    builder.add_node("compose_answer", lambda state: _compose_answer_node(service, state))
    builder.add_edge(START, "compose_answer")
    builder.add_edge("compose_answer", END)
    return builder.compile()


def _load_context_node(service: Any, state: AgentEntryGraphState) -> AgentEntryGraphState:
    """读取最近几轮对话，供后续安全判断和意图路由复用。"""
    session = state["session"]
    chat_session_id = state["chat_session_id"]
    return {"conversation_history": service._recent_conversation_context(session, chat_session_id)}


def _compose_answer_node(service: Any, state: AgentComposeGraphState) -> AgentComposeGraphState:
    """把已经准备好的结构化执行结果转换成最终答案。"""
    writer = _graph_event_emitter()
    if writer and state.get("stream"):
        writer({"type": "status", "label": "生成回答中"})
        stream_generator = service._stream_compose_answer(
            intent=state["intent"],
            message=state["message"],
            conversation_history=state.get("conversation_history", []),
            tool_outputs=state.get("tool_outputs", []),
            citations=service._citations_from_graph_state(state.get("citations", [])),
            used_tools=state.get("used_tools", []),
            use_max=bool(state.get("use_max", True)),
        )
        while True:
            try:
                chunk = next(stream_generator)
            except StopIteration as stop:
                return {"answer_text": stop.value or ""}
            if chunk:
                writer({"type": "delta", "text": chunk})
    answer = service._compose_answer(
        intent=state["intent"],
        message=state["message"],
        conversation_history=state.get("conversation_history", []),
        tool_outputs=state.get("tool_outputs", []),
        citations=service._citations_from_graph_state(state.get("citations", [])),
        used_tools=state.get("used_tools", []),
        use_max=bool(state.get("use_max", True)),
    )
    return {"answer_text": answer}


def _safety_gate_node(service: Any, state: AgentEntryGraphState) -> AgentEntryGraphState:
    """先于一切普通问答路径执行的医疗安全闸门。"""
    decision = safety_service.evaluate(state["message"])
    if decision.handoff_required:
        return {
            "immediate_response": service._build_safety_handoff_response(
                session_id=state["chat_session_id"],
                reason=decision.reason,
            )
        }
    return {}


def _report_gate_node(service: Any, state: AgentEntryGraphState) -> AgentEntryGraphState:
    """如果这轮绑定了报告，就先确认报告是否已经可用于解读。"""
    report_id = state.get("report_id")
    if not report_id:
        return {}
    report = report_service.get_report(state["session"], report_id)
    if report.parse_status not in REPORT_READY_STATUSES:
        return {
            "immediate_response": service._build_report_not_ready_response(
                session_id=state["chat_session_id"],
                parse_status=report.parse_status,
            )
        }
    return {}


def _route_intent_node(service: Any, state: AgentEntryGraphState) -> AgentEntryGraphState:
    """给出这一轮后续要走的意图。"""
    analysis = service._analyze_input(
        state["message"],
        state.get("report_id") is not None,
        state.get("conversation_history", []),
    )
    return {"analysis": analysis}


def _prepare_collect_more_info_node(service: Any, state: AgentEntryGraphState) -> AgentEntryGraphState:
    """准备信息不足场景的结构化执行结果。"""
    execution = service._build_collect_more_info_execution(
        conversation_history=state.get("conversation_history", []),
        message=state["message"],
        analysis=state["analysis"],
    )
    return {"execution_data": execution.model_dump(mode="json")}


def _prepare_report_follow_up_node(service: Any, state: AgentEntryGraphState) -> AgentEntryGraphState:
    """准备报告追问场景的结构化执行结果。"""
    execution = service._build_report_follow_up_execution(
        session=state["session"],
        report_id=state.get("report_id"),
        conversation_history=state.get("conversation_history", []),
        message=state["message"],
        analysis=state["analysis"],
        status_emitter=_graph_status_emitter(),
    )
    return {"execution_data": execution.model_dump(mode="json")}


def _prepare_term_explanation_node(service: Any, state: AgentEntryGraphState) -> AgentEntryGraphState:
    """准备术语解释场景的结构化执行结果。"""
    execution = service._prepare_term_explanation(
        session=state["session"],
        conversation_history=state.get("conversation_history", []),
        message=state["message"],
        analysis=state["analysis"],
        status_emitter=_graph_status_emitter(),
    )
    return {"execution_data": execution.model_dump(mode="json")}


def _prepare_retrieval_node(service: Any, state: AgentEntryGraphState) -> AgentEntryGraphState:
    """准备普通知识检索型场景的结构化执行结果。"""
    execution = service._build_retrieval_execution(
        session=state["session"],
        conversation_history=state.get("conversation_history", []),
        message=state["message"],
        analysis=state["analysis"],
        status_emitter=_graph_status_emitter(),
    )
    return {"execution_data": execution.model_dump(mode="json")}


def _after_immediate_gate(state: AgentEntryGraphState) -> str:
    """只要任一闸门给出了直接响应，就在图入口处结束。"""
    return "stop" if state.get("immediate_response") else "continue"


def _route_to_prepare_node(state: AgentEntryGraphState) -> str:
    """根据 intent 选择进入哪个准备节点。"""
    analysis = state.get("analysis")
    intent = getattr(analysis, "intent", None)
    if intent == "collect_more_info":
        return "prepare_collect_more_info"
    if intent == "report_follow_up":
        return "prepare_report_follow_up"
    if intent == "term_explanation":
        return "prepare_term_explanation"
    return "prepare_retrieval"


def _graph_status_emitter() -> Callable[[str], None] | None:
    """为图节点提供自定义流式状态写入器。"""
    writer = _graph_event_emitter()
    if not writer:
        return None
    return lambda label: writer({"type": "status", "label": label})


def _graph_event_emitter() -> Callable[[dict[str, Any]], None] | None:
    """返回 LangGraph 自定义流写入器，供节点推送状态或文本块。"""
    try:
        writer = get_stream_writer()
    except RuntimeError:
        return None
    return writer
