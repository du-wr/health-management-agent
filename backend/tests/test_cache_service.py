from datetime import datetime, timedelta, timezone

from sqlmodel import Session, SQLModel, create_engine

from app.core.config import Settings
from app.core.schemas import AgentResponse
from app.models.entities import AgentAnswerCache
from app.services.cache_service import CacheService


class FakeRedis:
    """用于测试的最小 Redis 替身。"""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.expirations: dict[str, int] = {}

    def get(self, key: str) -> str | None:
        return self.store.get(key)

    def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.store[key] = value
        if ex is not None:
            self.expirations[key] = ex

    def delete(self, key: str) -> None:
        self.store.pop(key, None)
        self.expirations.pop(key, None)


def make_session() -> Session:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    return Session(engine)


def build_settings() -> Settings:
    return Settings(
        REDIS_URL="redis://localhost:6379/0",
        REDIS_CACHE_TTL_SECONDS=321,
        REDIS_KEY_PREFIX="agent2-test",
    )


def build_response() -> AgentResponse:
    return AgentResponse(
        session_id="session-1",
        intent="term_explanation",
        answer="这是缓存回答。",
        citations=[],
        used_tools=["retrieve_knowledge"],
        follow_up_questions=[],
        safety_level="safe",
        handoff_required=False,
        debug=None,
    )


def test_cache_service_prefers_redis_before_db() -> None:
    fake_redis = FakeRedis()
    cache_service = CacheService(settings=build_settings(), redis_client=fake_redis)
    cache_key = "session-1::-::低密度脂蛋白"
    response = build_response()

    with make_session() as session:
        cache_service.save_agent_response(
            session=session,
            cache_key=cache_key,
            report_id=None,
            normalized_message="低密度脂蛋白",
            response=response,
        )

        cached = session.get(AgentAnswerCache, cache_key)
        assert cached is not None
        session.delete(cached)
        session.commit()

        loaded = cache_service.load_agent_response(session, cache_key)

    assert loaded is not None
    assert loaded.answer == response.answer
    assert fake_redis.expirations[cache_service._redis_cache_key(cache_key)] == 321


def test_cache_service_backfills_redis_after_db_hit() -> None:
    fake_redis = FakeRedis()
    cache_service = CacheService(settings=build_settings(), redis_client=fake_redis)
    cache_key = "session-1::-::总胆固醇"
    response = build_response().model_copy(update={"answer": "这是数据库兜底回答。"})

    with make_session() as session:
        session.add(
            AgentAnswerCache(
                cache_key=cache_key,
                report_id=None,
                normalized_message="总胆固醇",
                response_json=response.model_dump_json(),
                answer_text=response.answer,
                expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
            )
        )
        session.commit()

        loaded = cache_service.load_agent_response(session, cache_key)

    assert loaded is not None
    assert loaded.answer == response.answer
    assert fake_redis.get(cache_service._redis_cache_key(cache_key)) is not None
