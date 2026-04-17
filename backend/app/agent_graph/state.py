from __future__ import annotations

from typing import Any, TypedDict

from sqlmodel import Session

from app.core.schemas import AgentResponse


class AgentEntryGraphState(TypedDict, total=False):
    """Agent 入口图的状态。

    这张图只承载入口决策所需的最小字段：
    - 当前数据库会话
    - 当前聊天会话 id
    - 当前绑定报告
    - 用户消息
    - 短期上下文
    - 入口阶段直接返回的响应
    - 路由分析结果
    """

    session: Session
    chat_session_id: str
    report_id: str | None
    message: str
    conversation_history: list[dict[str, str]]
    memory_data: dict[str, Any] | None
    analysis: Any
    goal_data: dict[str, Any] | None
    plan_data: dict[str, Any] | None
    replan_data: dict[str, Any] | None
    immediate_response: AgentResponse | None
    execution_data: dict[str, Any] | None
    answer_text: str | None


class AgentComposeGraphState(TypedDict, total=False):
    """Agent 生成图的状态。

    这张图只负责把已经准备好的结构化执行结果转换成最终答案，
    尽量不再重复处理路由、安全和工具准备。
    """

    intent: str
    message: str
    conversation_history: list[dict[str, str]]
    tool_outputs: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    used_tools: list[str]
    use_max: bool
    stream: bool
    answer_text: str | None
