from sqlalchemy import text
import sqlalchemy as sa
from sqlmodel import Session, create_engine

from app.core.config import Settings
from app.core.database import ensure_database_ready, get_schema_version
from app.models.entities import AgentAnswerCache, ChatMessage, Report, SummaryArtifact


def test_sqlite_database_url_is_resolved_to_absolute_path() -> None:
    settings = Settings(_env_file=None, DATABASE_URL="sqlite:///../data/sqlite/test.db")

    assert settings.is_sqlite is True
    assert settings.sqlite_path is not None
    assert str(settings.sqlite_path).endswith("data\\sqlite\\test.db") or str(settings.sqlite_path).endswith("data/sqlite/test.db")
    assert settings.resolved_database_url.startswith("sqlite:///")


def test_postgres_database_url_keeps_raw_value() -> None:
    raw_url = "postgresql+psycopg://demo:demo@localhost:5432/medical_agent"
    settings = Settings(_env_file=None, DATABASE_URL=raw_url)

    assert settings.is_sqlite is False
    assert settings.sqlite_path is None
    assert settings.resolved_database_url == raw_url


def test_mysql_database_url_keeps_raw_value() -> None:
    raw_url = "mysql+pymysql://demo:demo@localhost:3306/medical_agent?charset=utf8mb4"
    settings = Settings(_env_file=None, DATABASE_URL=raw_url)

    assert settings.is_sqlite is False
    assert settings.sqlite_path is None
    assert settings.resolved_database_url == raw_url


def test_redis_enabled_depends_on_redis_url() -> None:
    disabled = Settings(_env_file=None, REDIS_URL="")
    enabled = Settings(_env_file=None, REDIS_URL="redis://127.0.0.1:6379/0")

    assert disabled.redis_enabled is False
    assert enabled.redis_enabled is True


def test_who_enabled_flag_can_disable_who_requests() -> None:
    settings = Settings(
        _env_file=None,
        WHO_ENABLED=False,
        WHO_CLIENT_ID="demo-id",
        WHO_CLIENT_SECRET="demo-secret",
    )

    assert settings.who_enabled is False


def test_schema_auto_init_follows_environment_defaults() -> None:
    dev_settings = Settings(_env_file=None, APP_ENV="development")
    prod_settings = Settings(_env_file=None, APP_ENV="production")
    forced_prod_settings = Settings(_env_file=None, APP_ENV="production", DB_AUTO_INIT_SCHEMA=True)

    assert dev_settings.schema_auto_init_enabled is True
    assert prod_settings.schema_auto_init_enabled is False
    assert forced_prod_settings.schema_auto_init_enabled is True


def test_ensure_database_ready_auto_inits_schema_in_development() -> None:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    settings = Settings(_env_file=None, APP_ENV="development")

    ensure_database_ready(db_engine=engine, app_settings=settings)

    with Session(engine) as session:
        tables = session.exec(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='report'")
        ).all()
    assert tables


def test_ensure_database_ready_requires_alembic_version_in_production() -> None:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    settings = Settings(_env_file=None, APP_ENV="production", DB_AUTO_INIT_SCHEMA=False)

    try:
        ensure_database_ready(db_engine=engine, app_settings=settings)
    except RuntimeError as exc:
        assert "alembic upgrade head" in str(exc)
    else:
        raise AssertionError("production environment should require alembic migration")


def test_get_schema_version_reads_alembic_version_table() -> None:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    with engine.begin() as connection:
        connection.exec_driver_sql("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)")
        connection.exec_driver_sql("INSERT INTO alembic_version (version_num) VALUES ('20260417_0001')")

    assert get_schema_version(engine) == "20260417_0001"


def test_long_text_fields_use_text_compatible_column_types() -> None:
    assert isinstance(Report.__table__.c.raw_text.type, sa.Text)
    assert isinstance(AgentAnswerCache.__table__.c.response_json.type, sa.Text)
    assert isinstance(AgentAnswerCache.__table__.c.answer_text.type, sa.Text)
    assert isinstance(ChatMessage.__table__.c.content.type, sa.Text)
    assert isinstance(SummaryArtifact.__table__.c.markdown.type, sa.Text)
