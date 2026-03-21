from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
from urllib.parse import parse_qs, urlparse

import httpx
from bs4 import BeautifulSoup
from sqlmodel import Session, select

from app.core.config import get_settings
from app.models.entities import KnowledgeDoc


DISCOVERY_QUERIES = [
    "site:nhc.gov.cn \u5065\u5eb7\u79d1\u666e \u4f53\u68c0 \u6307\u6807 \u5f02\u5e38",
    "site:gov.cn \u5065\u5eb7 \u79d1\u666e \u5e38\u89c1\u75c7\u72b6 \u5c31\u533b",
    "site:nmpa.gov.cn \u836f\u54c1 \u79d1\u666e \u7981\u5fcc",
    "site:pkuph.cn \u5065\u5eb7\u79d1\u666e \u79d1\u5ba4",
    "site:pumch.cn \u5065\u5eb7\u6559\u80b2 \u79d1\u666e",
    "site:zs-hospital.sh.cn \u5065\u5eb7\u79d1\u666e \u6307\u6807",
    "site:dxy.cn \u5e38\u89c1\u75c7\u72b6 \u79d1\u666e",
]
TRUST_A_HINTS = ("gov.cn", "nhc.gov.cn", "nmpa.gov.cn", "edu.cn", "hospital", "pkuph.cn", "pumch.cn")
TRUST_B_HINTS = ("dxy.cn", "medlive.cn", "yxj.org.cn")
EXCLUSION_PATTERNS = (
    "\u8bba\u575b",
    "\u95ee\u7b54",
    "\u5e7f\u544a",
    "\u62db\u5546",
    "\u6302\u53f7\u7f51",
    "\u9884\u7ea6\u6302\u53f7",
    "\u63a8\u5e7f",
    "\u89c6\u9891",
)
CHINA_CONTEXT_PATTERNS = (
    "\u533b\u9662",
    "\u95e8\u8bca",
    "\u6302\u53f7",
    "\u590d\u67e5",
    "\u4e2d\u56fd",
    "\u56fd\u5bb6\u536b\u5065\u59d4",
    "\u56fd\u5bb6\u836f\u76d1\u5c40",
)


@dataclass
class ClassifiedDocument:
    title: str
    url: str
    source_domain: str
    source_org: str
    trust_tier: str
    content_type: str
    snippet: str
    body_text: str
    content_hash: str


class KnowledgeService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def bootstrap(self, session: Session) -> tuple[int, int]:
        ingested = 0
        skipped = 0
        for query in DISCOVERY_QUERIES:
            for candidate in self.discover(query):
                if session.exec(select(KnowledgeDoc).where(KnowledgeDoc.url == candidate["url"])).first():
                    skipped += 1
                    continue
                fetched = self.fetch(candidate["url"])
                if not fetched:
                    skipped += 1
                    continue
                classified = self.classify(fetched)
                if not classified:
                    skipped += 1
                    continue
                self.ingest(session, classified)
                ingested += 1
        return ingested, skipped

    def discover(self, query: str) -> list[dict[str, str]]:
        response = httpx.get(
            self.settings.bing_search_base,
            params={"q": query, "setlang": "zh-Hans"},
            headers={"User-Agent": self.settings.user_agent},
            timeout=20.0,
            follow_redirects=True,
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")
        candidates: list[dict[str, str]] = []
        for item in soup.select("li.b_algo")[:8]:
            anchor = item.select_one("h2 a")
            if not anchor or not anchor.get("href"):
                continue
            href = anchor["href"]
            if "bing.com" in href and "url" in href:
                parsed = parse_qs(urlparse(href).query)
                href = parsed.get("u", [href])[0]
            candidates.append({"title": anchor.get_text(" ", strip=True), "url": href})
        return candidates

    def fetch(self, url: str) -> dict[str, str] | None:
        try:
            response = httpx.get(
                url,
                headers={"User-Agent": self.settings.user_agent},
                timeout=20.0,
                follow_redirects=True,
            )
            response.raise_for_status()
        except Exception:
            return None
        if "text/html" not in response.headers.get("content-type", ""):
            return None
        soup = BeautifulSoup(response.text, "lxml")
        title = soup.title.get_text(strip=True) if soup.title else url
        paragraphs: list[str] = []
        for node in soup.select("article p, .article p, .content p, p"):
            text = self._clean_text(node.get_text(" ", strip=True))
            if len(text) >= 20:
                paragraphs.append(text)
        body_text = "\n".join(paragraphs[:40]).strip()
        if not body_text:
            return None
        return {"title": title, "url": str(response.url), "body_text": body_text}

    def classify(self, fetched: dict[str, str]) -> ClassifiedDocument | None:
        title = self._clean_text(fetched["title"])
        body_text = fetched["body_text"]
        if not self._is_chinese(title + body_text):
            return None
        if any(pattern in title + body_text for pattern in EXCLUSION_PATTERNS):
            return None
        if not any(pattern in body_text for pattern in CHINA_CONTEXT_PATTERNS):
            return None
        domain = urlparse(fetched["url"]).netloc.lower()
        return ClassifiedDocument(
            title=title,
            url=fetched["url"],
            source_domain=domain,
            source_org=self._infer_org(domain),
            trust_tier=self._get_trust_tier(domain),
            content_type="article",
            snippet=body_text[:180],
            body_text=body_text,
            content_hash=hashlib.sha256(f"{fetched['url']}::{body_text}".encode("utf-8")).hexdigest(),
        )

    def ingest(self, session: Session, classified: ClassifiedDocument) -> None:
        doc = KnowledgeDoc(
            url=classified.url,
            title=classified.title,
            source_domain=classified.source_domain,
            source_org=classified.source_org,
            trust_tier=classified.trust_tier,
            content_type=classified.content_type,
            snippet=classified.snippet,
            body_text=classified.body_text,
            content_hash=classified.content_hash,
            crawl_status="ingested",
            crawled_at=datetime.now(timezone.utc),
        )
        session.add(doc)
        session.commit()
        session.refresh(doc)
        session.connection().exec_driver_sql(
            """
            INSERT INTO knowledge_doc_fts(doc_id, title, snippet, body_text, trust_tier, source_domain)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (doc.id, doc.title, doc.snippet, doc.body_text, doc.trust_tier, doc.source_domain),
        )
        session.commit()

    def retrieve(self, session: Session, query: str, limit: int = 5) -> list[KnowledgeDoc]:
        escaped_query = query.replace('"', " ").strip()
        if not escaped_query:
            return []
        docs: list[KnowledgeDoc] = []
        try:
            rows = session.connection().exec_driver_sql(
                """
                SELECT doc_id FROM knowledge_doc_fts
                WHERE knowledge_doc_fts MATCH ?
                LIMIT ?
                """,
                (escaped_query, limit * 3),
            ).fetchall()
            for (doc_id,) in rows:
                doc = session.get(KnowledgeDoc, doc_id)
                if doc:
                    docs.append(doc)
        except Exception:
            docs = session.exec(
                select(KnowledgeDoc)
                .where(KnowledgeDoc.body_text.contains(escaped_query) | KnowledgeDoc.title.contains(escaped_query))
                .limit(limit * 3)
            ).all()
        docs.sort(key=self._trust_sort_key)
        allow_c = not any(doc.trust_tier in {"A", "B"} for doc in docs)
        selected: list[KnowledgeDoc] = []
        for doc in docs:
            if doc.trust_tier == "C" and not allow_c:
                continue
            selected.append(doc)
            if len(selected) >= limit:
                break
        return selected

    def source_stats(self, session: Session) -> tuple[int, dict[str, int], list[KnowledgeDoc]]:
        docs = session.exec(select(KnowledgeDoc).order_by(KnowledgeDoc.crawled_at.desc())).all()
        breakdown = {"A": 0, "B": 0, "C": 0}
        for doc in docs:
            breakdown[doc.trust_tier] = breakdown.get(doc.trust_tier, 0) + 1
        return len(docs), breakdown, docs[:10]

    def _is_chinese(self, text: str) -> bool:
        return len(re.findall(r"[\u4e00-\u9fff]", text)) >= 20

    def _get_trust_tier(self, domain: str) -> str:
        if any(hint in domain for hint in TRUST_A_HINTS):
            return "A"
        if any(hint in domain for hint in TRUST_B_HINTS):
            return "B"
        return "C"

    def _infer_org(self, domain: str) -> str:
        if "gov.cn" in domain:
            return "gov"
        if "edu.cn" in domain:
            return "medical_school"
        if "hospital" in domain or "pkuph.cn" in domain or "pumch.cn" in domain:
            return "public_hospital"
        if domain.endswith("dxy.cn") or "medlive" in domain:
            return "medical_platform"
        return "other"

    def _clean_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", unescape(text)).strip()

    def _trust_sort_key(self, doc: KnowledgeDoc) -> tuple[int, str]:
        order = {"A": 0, "B": 1, "C": 2}
        return (order.get(doc.trust_tier, 3), doc.title)


knowledge_service = KnowledgeService()
