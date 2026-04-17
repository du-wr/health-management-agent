from collections.abc import Generator
from typing import Any

from sqlalchemy import inspect, text
from sqlmodel import Session, SQLModel, create_engine

from app.core.config import Settings, get_settings


# 数据库层只做三件事：
# 1. 根据配置创建统一 engine
# 2. 初始化 ORM 表和 SQLite 专属 FTS 表
# 3. 给 FastAPI / worker 提供 Session 依赖
settings = get_settings()


def _build_engine_kwargs(app_settings: Settings | None = None) -> dict[str, Any]:
    """根据当前数据库类型返回建连参数。"""
    app_settings = app_settings or settings
    if app_settings.is_sqlite:
        # SQLite 只在单机开发时使用，不需要连接池参数。
        return {"connect_args": {"check_same_thread": False}}
    # MySQL / PostgreSQL 这类服务型数据库统一走连接池参数。
    return {
        "pool_pre_ping": True,
        "pool_size": app_settings.database_pool_size,
        "max_overflow": app_settings.database_max_overflow,
        "pool_timeout": app_settings.database_pool_timeout_seconds,
        "pool_recycle": app_settings.database_pool_recycle_seconds,
    }


def _build_engine():
    """创建全局 SQLAlchemy engine。"""
    return create_engine(settings.resolved_database_url, **_build_engine_kwargs())


engine = _build_engine()


def init_db(db_engine=None, app_settings: Settings | None = None) -> None:
    """初始化数据库表和 SQLite 专属全文检索表。"""
    app_settings = app_settings or settings
    db_engine = db_engine or engine

    from app.models import entities  # noqa: F401

    # 先创建所有 ORM 普通表。
    SQLModel.metadata.create_all(db_engine)
    if not app_settings.is_sqlite:
        # MySQL / PostgreSQL 不创建 SQLite FTS5 虚拟表。
        return

    with db_engine.begin() as connection:
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


def ensure_database_ready(db_engine=None, app_settings: Settings | None = None) -> None:
    """根据环境策略决定是自动建表，还是强制要求先执行迁移。"""
    app_settings = app_settings or settings
    db_engine = db_engine or engine

    if app_settings.schema_auto_init_enabled:
        init_db(db_engine=db_engine, app_settings=app_settings)
        return

    if not _has_table(db_engine, "alembic_version"):
        raise RuntimeError(
            "Database schema is not initialized for the current environment. "
            "Run `alembic upgrade head` before starting the application."
        )


def _has_table(db_engine, table_name: str) -> bool:
    """判断数据库里是否存在指定表。"""
    inspector = inspect(db_engine)
    return table_name in inspector.get_table_names()


def get_schema_version(db_engine=None) -> str | None:
    """读取当前数据库的 Alembic 版本号。"""
    db_engine = db_engine or engine
    if not _has_table(db_engine, "alembic_version"):
        return None

    with db_engine.connect() as connection:
        row = connection.execute(text("SELECT version_num FROM alembic_version")).first()
    if row is None:
        return None
    return str(row[0])


def get_session() -> Generator[Session, None, None]:
    """为 FastAPI 路由提供数据库会话依赖。"""
    with Session(engine) as session:
        yield session
