from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone

from sqlmodel import Session, select

from app.models.entities import KnowledgeDoc
from app.services.knowledge_seed import LOCAL_KNOWLEDGE_SEEDS


TOKEN_SPLIT_PATTERN = re.compile(r"[\s,.;:，。；？！/()（）\-]+")


class KnowledgeService:
    def ensure_initialized(self, session: Session) -> dict[str, int | str]:
        existing_count = self.count_docs(session)
        if existing_count > 0:
            return {"seeded": 0, "status": "existing"}
        seeded = self.seed_local_knowledge(session)
        return {"seeded": seeded, "status": "seeded_only"}

    def count_docs(self, session: Session) -> int:
        return len(session.exec(select(KnowledgeDoc.id)).all())

    def seed_local_knowledge(self, session: Session) -> int:
        created = 0
        for item in LOCAL_KNOWLEDGE_SEEDS:
            url = f"seed://knowledge/{item['slug']}"
            existing = session.exec(select(KnowledgeDoc).where(KnowledgeDoc.url == url)).first()
            if existing:
                continue
            alias_text = " ".join(item.get("aliases", []))
            doc = KnowledgeDoc(
                url=url,
                title=str(item["title"]),
                source_domain="local.seed",
                source_org="local_seed",
                trust_tier="A",
                content_type="seed",
                snippet=str(item["snippet"]),
                body_text=f"{alias_text}\n{item['body_text']}",
                content_hash=hashlib.sha256(
                    f"{item['slug']}::{item['title']}::{alias_text}::{item['body_text']}".encode("utf-8")
                ).hexdigest(),
                crawl_status="seeded",
                crawled_at=datetime.now(timezone.utc),
            )
            self.ingest_doc(session, doc)
            created += 1
        return created

    def ingest_doc(self, session: Session, doc: KnowledgeDoc) -> None:
        session.add(doc)
        session.commit()
        session.refresh(doc)
        try:
            session.connection().exec_driver_sql(
                """
                INSERT INTO knowledge_doc_fts(doc_id, title, snippet, body_text, trust_tier, source_domain)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (doc.id, doc.title, doc.snippet, doc.body_text, doc.trust_tier, doc.source_domain),
            )
            session.commit()
        except Exception:
            session.rollback()

    def retrieve(self, session: Session, query: str, limit: int = 5) -> list[KnowledgeDoc]:
        normalized_query = query.strip()
        if not normalized_query:
            return []

        direct_matches = session.exec(
            select(KnowledgeDoc)
            .where(
                KnowledgeDoc.title.contains(normalized_query)
                | KnowledgeDoc.snippet.contains(normalized_query)
                | KnowledgeDoc.body_text.contains(normalized_query)
            )
            .limit(limit * 5)
        ).all()
        if direct_matches:
            direct_matches.sort(key=lambda doc: (self._trust_rank(doc.trust_tier), doc.title))
            return direct_matches[:limit]

        tokens = [token for token in TOKEN_SPLIT_PATTERN.split(normalized_query) if len(token) >= 2]
        candidates = session.exec(select(KnowledgeDoc)).all()
        scored: list[tuple[int, KnowledgeDoc]] = []
        for doc in candidates:
            haystack = f"{doc.title} {doc.snippet} {doc.body_text}".lower()
            score = 0
            for token in tokens:
                token_lower = token.lower()
                if token_lower in haystack:
                    score += 2 if token_lower in doc.title.lower() else 1
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda item: (-item[0], self._trust_rank(item[1].trust_tier), item[1].title))
        return [doc for _, doc in scored[:limit]]

    def explain_lab_items(self, session: Session, item_names: list[str]) -> list[dict[str, str]]:
        explanations: list[dict[str, str]] = []
        for item_name in item_names:
            matched = self._match_lab_doc(session, item_name)
            if not matched:
                continue
            explanations.append(
                {
                    "name": item_name,
                    "title": matched.title,
                    "snippet": matched.snippet,
                    "detail": matched.body_text,
                    "doc_id": matched.id,
                    "url": matched.url,
                    "trust_tier": matched.trust_tier,
                }
            )
        return explanations

    def source_stats(self, session: Session) -> tuple[int, dict[str, int], list[KnowledgeDoc]]:
        docs = session.exec(select(KnowledgeDoc).order_by(KnowledgeDoc.crawled_at.desc())).all()
        breakdown = {"A": 0, "B": 0, "C": 0}
        for doc in docs:
            breakdown[doc.trust_tier] = breakdown.get(doc.trust_tier, 0) + 1
        return len(docs), breakdown, docs[:10]

    def pack_docs(self, docs: list[KnowledgeDoc]) -> list[dict[str, str]]:
        return [
            {
                "doc_id": doc.id,
                "title": doc.title,
                "url": doc.url,
                "trust_tier": doc.trust_tier,
                "snippet": doc.snippet,
                "detail": doc.body_text[:600],
            }
            for doc in docs
        ]

    def _match_lab_doc(self, session: Session, item_name: str) -> KnowledgeDoc | None:
        normalized_item = self._normalize(item_name)
        candidates = session.exec(select(KnowledgeDoc)).all()
        best_doc: KnowledgeDoc | None = None
        best_score = 0
        for doc in candidates:
            haystack = self._normalize(f"{doc.title} {doc.snippet} {doc.body_text}")
            score = 0
            alias_line = doc.body_text.splitlines()[0] if doc.body_text else ""
            aliases = [self._normalize(alias) for alias in alias_line.split() if alias.strip()]
            if normalized_item == self._normalize(doc.title):
                score += 30
            if normalized_item in aliases:
                score += 25
            if normalized_item in haystack:
                score += 5
            title_normalized = self._normalize(doc.title)
            if title_normalized in normalized_item or normalized_item in title_normalized:
                score += 4
            for token in TOKEN_SPLIT_PATTERN.split(item_name):
                token_lower = token.lower().strip()
                if len(token_lower) < 2:
                    continue
                if token_lower in haystack:
                    score += 1
            if score > best_score:
                best_score = score
                best_doc = doc
        return best_doc if best_score > 0 else None

    def _trust_rank(self, trust_tier: str) -> int:
        return {"A": 0, "B": 1, "C": 2}.get(trust_tier, 3)

    def _normalize(self, text: str) -> str:
        return re.sub(r"[\s\-_/()（）]+", "", text.lower())


knowledge_service = KnowledgeService()
