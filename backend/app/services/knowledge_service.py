from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone

from sqlmodel import Session, select

from app.models.entities import KnowledgeDoc
from app.services.knowledge_seed import LOCAL_KNOWLEDGE_SEEDS


TOKEN_SPLIT_PATTERN = re.compile(r"[\s,.;:，。；？！/()（）\-]+")


class KnowledgeService:
    """本地知识库服务。

    这个项目最终选择了“本地种子知识库”而不是联网抓取，
    因为演示阶段更强调稳定、可控和可解释。
    """

    def ensure_initialized(self, session: Session) -> dict[str, int | str]:
        """如果数据库里还没有知识文档，就自动灌入本地种子。"""
        existing_count = self.count_docs(session)
        if existing_count > 0:
            return {"seeded": 0, "status": "existing"}
        seeded = self.seed_local_knowledge(session)
        return {"seeded": seeded, "status": "seeded_only"}

    def count_docs(self, session: Session) -> int:
        return len(session.exec(select(KnowledgeDoc.id)).all())

    def seed_local_knowledge(self, session: Session) -> int:
        """把 `knowledge_seed.py` 里的静态知识写入数据库。"""
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
        """写入主表，并尽量同步写入 FTS 辅助表。"""
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
        """根据查询词检索知识文档。

        当前策略比较朴素：
        1. 先做直接包含匹配
        2. 如果没有明显结果，再拆 token 做简单打分
        """
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
        """根据指标名获取解释素材。

        报告解读链并不是把异常项直接丢给模型，
        而是先尽量命中本地知识库，再把命中的内容作为后续综合和润色的原料。
        """
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
        """返回知识库规模和信任等级分布。"""
        docs = session.exec(select(KnowledgeDoc).order_by(KnowledgeDoc.crawled_at.desc())).all()
        breakdown = {"A": 0, "B": 0, "C": 0}
        for doc in docs:
            breakdown[doc.trust_tier] = breakdown.get(doc.trust_tier, 0) + 1
        return len(docs), breakdown, docs[:10]

    def pack_docs(self, docs: list[KnowledgeDoc]) -> list[dict[str, str]]:
        """把 ORM 文档压缩成更适合放进 prompt 的字典结构。"""
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
        """尽量把一条指标名映射到最合适的知识条目。

        这里不是复杂检索引擎，而是一个“标题 + 别名 + 正文”的简单打分器。
        对体检指标这种相对稳定的词表，效果通常足够。
        """
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
