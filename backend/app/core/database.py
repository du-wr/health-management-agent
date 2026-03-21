from collections.abc import Generator

from sqlmodel import Session, SQLModel, create_engine

from app.core.config import get_settings


settings = get_settings()
engine = create_engine(
    f"sqlite:///{settings.sqlite_path}",
    connect_args={"check_same_thread": False},
)


def init_db() -> None:
    from app.models import entities  # noqa: F401

    SQLModel.metadata.create_all(engine)
    with engine.begin() as connection:
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
    with Session(engine) as session:
        yield session
