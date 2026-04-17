from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from sqlmodel import Session

from app.core.config import Settings, get_settings
from app.core.schemas import AgentResponse
from app.models.entities import AgentAnswerCache

try:
    from redis import Redis
except Exception:  # pragma: no cover - 本地未安装 redis 时允许退化到数据库缓存
    Redis = None  # type: ignore[assignment]


class CacheService:
    """负责 Agent 最终回答的缓存。

    当前实现采用“两级缓存”：
    1. Redis：命中快，适合作为主缓存
    2. 数据库表：作为持久化兜底，避免 Redis 未配置时功能失效
    """

    def __init__(self, settings: Settings | None = None, redis_client: Any | None = None) -> None:
        self.settings = settings or get_settings()
        self.redis_client = redis_client if redis_client is not None else self._build_redis_client()

    def load_agent_response(self, session: Session, cache_key: str) -> AgentResponse | None:
        """优先从 Redis 读缓存，未命中时回退到数据库表。"""
        cached = self._load_from_redis(cache_key)
        if cached:
            return cached

        cached = self._load_from_db(session, cache_key)
        if cached:
            # 数据库命中后顺手回填 Redis，下次读取会更快。
            self._save_to_redis(cache_key, cached)
        return cached

    def save_agent_response(
        self,
        session: Session,
        cache_key: str,
        report_id: str | None,
        normalized_message: str,
        response: AgentResponse,
    ) -> None:
        """同时写入 Redis 和数据库表，保证缓存和兜底数据一致。"""
        self._save_to_redis(cache_key, response)
        self._save_to_db(session, cache_key, report_id, normalized_message, response)

    def delete_agent_response(self, session: Session, cache_key: str) -> None:
        """删除指定缓存键，用于清理半成品或失效结果。"""
        self._delete_from_redis(cache_key)
        self._delete_from_db(session, cache_key)

    def _build_redis_client(self) -> Any | None:
        """根据配置尝试创建 Redis 客户端。"""
        if not self.settings.redis_enabled or Redis is None:
            return None
        try:
            return Redis.from_url(
                self.settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
        except Exception:
            # Redis 初始化失败时直接退化到数据库缓存，避免影响主流程。
            return None

    def _redis_cache_key(self, cache_key: str) -> str:
        """统一 Redis key 命名，避免和其他业务混淆。"""
        prefix = self.settings.redis_key_prefix.strip() or "medical_agent"
        return f"{prefix}:agent_response:{cache_key}"

    def _redis_ttl_seconds(self) -> int:
        """返回 Redis 缓存 TTL。"""
        ttl = self.settings.redis_cache_ttl_seconds or self.settings.agent_response_cache_ttl_seconds
        return max(ttl, 60)

    def _load_from_redis(self, cache_key: str) -> AgentResponse | None:
        """从 Redis 读取缓存。"""
        if self.redis_client is None:
            return None
        try:
            payload = self.redis_client.get(self._redis_cache_key(cache_key))
        except Exception:
            return None
        if not payload:
            return None
        try:
            return AgentResponse.model_validate_json(payload)
        except Exception:
            self._delete_from_redis(cache_key)
            return None

    def _save_to_redis(self, cache_key: str, response: AgentResponse) -> None:
        """把最终答案写入 Redis。"""
        if self.redis_client is None:
            return
        try:
            self.redis_client.set(
                self._redis_cache_key(cache_key),
                response.model_dump_json(),
                ex=self._redis_ttl_seconds(),
            )
        except Exception:
            return

    def _delete_from_redis(self, cache_key: str) -> None:
        """删除 Redis 中的缓存键。"""
        if self.redis_client is None:
            return
        try:
            self.redis_client.delete(self._redis_cache_key(cache_key))
        except Exception:
            return

    def _load_from_db(self, session: Session, cache_key: str) -> AgentResponse | None:
        """从数据库表读取缓存。"""
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

    def _save_to_db(
        self,
        session: Session,
        cache_key: str,
        report_id: str | None,
        normalized_message: str,
        response: AgentResponse,
    ) -> None:
        """把最终答案写入数据库表。"""
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

    def _delete_from_db(self, session: Session, cache_key: str) -> None:
        """删除数据库表中的缓存键。"""
        cached = session.get(AgentAnswerCache, cache_key)
        if not cached:
            return
        session.delete(cached)
        session.commit()


cache_service = CacheService()
