from collections.abc import Generator

from sqlmodel import Session, SQLModel, create_engine

from app.core.config import get_settings


# 数据库层只做两件事：
# 1. 创建引擎和 Session
# 2. 提供初始化和依赖注入能力
settings = get_settings()


def _build_engine():
    """根据当前数据库类型构建 SQLAlchemy 引擎。"""
    if settings.is_sqlite:
        return create_engine(
            settings.resolved_database_url,
            connect_args={"check_same_thread": False},
        )
    return create_engine(
        settings.resolved_database_url,
        pool_pre_ping=True,
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
        pool_timeout=settings.database_pool_timeout_seconds,
        pool_recycle=settings.database_pool_recycle_seconds,
    )


engine = _build_engine()


def init_db() -> None:
    """初始化数据库表和 FTS 辅助表。"""
    from app.models import entities  # noqa: F401

    # 先让 SQLModel 根据 ORM 定义创建普通表。
    SQLModel.metadata.create_all(engine)
    if not settings.is_sqlite:
        return

    with engine.begin() as connection:
        # 额外创建一个 SQLite FTS5 虚拟表，供知识库文本检索使用。
        connection.exec_driver_sql(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_doc_fts
            USING fts5(
                doc_id UNINDEXED,
                title,
                snippet,
                body_text,
                trust_tier UNINDEXED,
                source_domain UNINDEXED
            );
            """
        )


def get_session() -> Generator[Session, None, None]:
    """给 FastAPI 路由提供数据库会话依赖。"""
    with Session(engine) as session:
        yield session
