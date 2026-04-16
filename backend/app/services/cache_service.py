from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from sqlmodel import Session

from app.core.config import get_settings
from app.core.schemas import AgentResponse
from app.models.entities import AgentAnswerCache


class CacheService:
    """负责 Agent 最终回答的持久化缓存。"""

    def __init__(self) -> None:
        self.settings = get_settings()

    def load_agent_response(self, session: Session, cache_key: str) -> AgentResponse | None:
        """读取缓存中的最终回答。

        这里只缓存“已经可以直接返回给前端”的 AgentResponse，
        避免把复杂的中间执行状态长期固化到数据库里。
        """
        cached = session.get(AgentAnswerCache, cache_key)
        if not cached:
            return None
        now = datetime.now(timezone.utc)
        expires_at = cached.expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if expires_at <= now:
            session.delete(cached)
            session.commit()
            return None
        try:
            return AgentResponse.model_validate_json(cached.response_json)
        except Exception:
            session.delete(cached)
            session.commit()
            return None

    def save_agent_response(
        self,
        session: Session,
        cache_key: str,
        report_id: str | None,
        normalized_message: str,
        response: AgentResponse,
    ) -> None:
        """保存最终回答缓存。

        这里采用简单的覆盖写入策略，便于后续继续演进到 Redis 或多级缓存。
        """
        expires_at = datetime.now(timezone.utc) + timedelta(
            seconds=max(self.settings.agent_response_cache_ttl_seconds, 60)
        )
        existing = session.get(AgentAnswerCache, cache_key)
        payload = response.model_dump_json()
        if existing:
            existing.report_id = report_id
            existing.normalized_message = normalized_message
            existing.response_json = payload
            existing.answer_text = response.answer
            existing.expires_at = expires_at
            session.add(existing)
        else:
            session.add(
                AgentAnswerCache(
                    cache_key=cache_key,
                    report_id=report_id,
                    normalized_message=normalized_message,
                    response_json=payload,
                    answer_text=response.answer,
                    expires_at=expires_at,
                )
            )
        session.commit()

    def delete_agent_response(self, session: Session, cache_key: str) -> None:
        """删除指定缓存键，用于清理半成品或失效结果。"""
        cached = session.get(AgentAnswerCache, cache_key)
        if not cached:
            return
        session.delete(cached)
        session.commit()


cache_service = CacheService()
