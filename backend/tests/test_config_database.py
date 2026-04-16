from app.core.config import Settings


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
