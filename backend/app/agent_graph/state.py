from __future__ import annotations

from typing import Any, TypedDict

from sqlmodel import Session

from app.core.schemas import AgentResponse


class AgentEntryGraphState(TypedDict, total=False):
    """Agent 入口图的状态。

    第一阶段只承载入口决策需要的最小字段：
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
    analysis: Any
    immediate_response: AgentResponse | None
    execution_data: dict[str, Any] | None
