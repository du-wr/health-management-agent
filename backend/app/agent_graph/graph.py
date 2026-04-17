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
    builder.add_node("load_memory", lambda state: _load_memory_node(service, state))
    builder.add_node("safety_gate", lambda state: _safety_gate_node(service, state))
    builder.add_node("report_gate", lambda state: _report_gate_node(service, state))
    builder.add_node("route_intent", lambda state: _route_intent_node(service, state))
    builder.add_node("resolve_goal", lambda state: _resolve_goal_node(service, state))
    builder.add_node("plan_execution", lambda state: _plan_execution_node(service, state))
    builder.add_node("prepare_collect_more_info", lambda state: _prepare_collect_more_info_node(service, state))
    builder.add_node("prepare_report_follow_up", lambda state: _prepare_report_follow_up_node(service, state))
    builder.add_node("prepare_term_explanation", lambda state: _prepare_term_explanation_node(service, state))
    builder.add_node("prepare_retrieval", lambda state: _prepare_retrieval_node(service, state))
    builder.add_node("replan_execution", lambda state: _replan_execution_node(service, state))

    builder.add_edge(START, "load_context")
    builder.add_edge("load_context", "load_memory")
    builder.add_edge("load_memory", "safety_gate")
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
    builder.add_edge("route_intent", "resolve_goal")
    builder.add_edge("resolve_goal", "plan_execution")
    builder.add_conditional_edges(
        "plan_execution",
        _route_to_prepare_node,
        {
            "prepare_collect_more_info": "prepare_collect_more_info",
            "prepare_report_follow_up": "prepare_report_follow_up",
            "prepare_term_explanation": "prepare_term_explanation",
            "prepare_retrieval": "prepare_retrieval",
        },
    )
    builder.add_edge("prepare_collect_more_info", "replan_execution")
    builder.add_edge("prepare_report_follow_up", "replan_execution")
    builder.add_edge("prepare_term_explanation", "replan_execution")
    builder.add_edge("prepare_retrieval", "replan_execution")
    builder.add_edge("replan_execution", END)

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


def _load_memory_node(service: Any, state: AgentEntryGraphState) -> AgentEntryGraphState:
    """加载会话级摘要记忆和报告级长期洞察。"""

    memory_data = service._load_agent_memory(
        session=state["session"],
        session_id=state["chat_session_id"],
        report_id=state.get("report_id"),
    )
    return {"memory_data": memory_data}


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


def _resolve_goal_node(service: Any, state: AgentEntryGraphState) -> AgentEntryGraphState:
    """把单轮问题映射成长期健康管理目标。"""

    goal = service._resolve_goal(
        message=state["message"],
        analysis=state["analysis"],
        report_id=state.get("report_id"),
        conversation_history=state.get("conversation_history", []),
        memory_data=state.get("memory_data") or {},
    )
    return {"goal_data": goal.model_dump(mode="json")}


def _plan_execution_node(service: Any, state: AgentEntryGraphState) -> AgentEntryGraphState:
    """为当前目标生成显式执行计划。"""

    plan = service._build_execution_plan(
        message=state["message"],
        analysis=state["analysis"],
        goal_data=state.get("goal_data") or {},
        report_id=state.get("report_id"),
        conversation_history=state.get("conversation_history", []),
        memory_data=state.get("memory_data") or {},
    )
    return {"plan_data": plan.model_dump(mode="json")}


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
        memory_data=state.get("memory_data") or {},
        planner_data=state.get("plan_data") or {},
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
        memory_data=state.get("memory_data") or {},
        planner_data=state.get("plan_data") or {},
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
        memory_data=state.get("memory_data") or {},
        planner_data=state.get("plan_data") or {},
        status_emitter=_graph_status_emitter(),
    )
    return {"execution_data": execution.model_dump(mode="json")}


def _replan_execution_node(service: Any, state: AgentEntryGraphState) -> AgentEntryGraphState:
    """根据准备好的工具结果判断是否需要重规划。"""

    execution_data = state.get("execution_data") or {}
    replan = service._replan_after_execution(
        analysis=state["analysis"],
        goal_data=state.get("goal_data") or {},
        plan_data=state.get("plan_data") or {},
        execution_data=execution_data,
        conversation_history=state.get("conversation_history", []),
        message=state["message"],
        memory_data=state.get("memory_data") or {},
    )
    updated_execution = service._apply_replan_to_execution(
        execution_data=execution_data,
        replan_data=replan.model_dump(mode="json"),
        goal_data=state.get("goal_data") or {},
        plan_data=state.get("plan_data") or {},
        analysis=state["analysis"],
        conversation_history=state.get("conversation_history", []),
        message=state["message"],
        memory_data=state.get("memory_data") or {},
    )
    return {
        "replan_data": replan.model_dump(mode="json"),
        "execution_data": updated_execution,
    }


def _after_immediate_gate(state: AgentEntryGraphState) -> str:
    """只要任一闸门给出了直接响应，就在图入口处结束。"""
    return "stop" if state.get("immediate_response") else "continue"


def _route_to_prepare_node(state: AgentEntryGraphState) -> str:
    """根据 intent 选择进入哪个准备节点。"""
    plan_data = state.get("plan_data") or {}
    intent = plan_data.get("intent")
    if not intent:
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
