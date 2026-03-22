from sqlmodel import Session, SQLModel, create_engine

from app.services.knowledge_service import knowledge_service


def make_session() -> Session:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    return Session(engine)


def test_seed_local_knowledge_creates_many_docs() -> None:
    with make_session() as session:
        created = knowledge_service.seed_local_knowledge(session)
        assert created >= 40
        assert knowledge_service.count_docs(session) == created


def test_retrieve_returns_relevant_seed_doc() -> None:
    with make_session() as session:
        knowledge_service.seed_local_knowledge(session)
        docs = knowledge_service.retrieve(session, "低密度脂蛋白", limit=3)
        assert docs
        assert any("低密度脂蛋白胆固醇" in doc.title for doc in docs)


def test_explain_lab_items_matches_aliases() -> None:
    with make_session() as session:
        knowledge_service.seed_local_knowledge(session)
        explanations = knowledge_service.explain_lab_items(session, ["LDL-C", "总胆固醇"])
        titles = [item["title"] for item in explanations]
        assert "低密度脂蛋白胆固醇" in titles
        assert "总胆固醇" in titles
