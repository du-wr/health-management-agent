from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from app.core.config import get_settings


logger = logging.getLogger(__name__)


class WHOService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._access_token: str | None = None
        self._token_expiry: datetime | None = None

    def is_configured(self) -> bool:
        return bool(self.settings.who_client_id and self.settings.who_client_secret)

    def _get_token(self) -> str:
        if not self.is_configured():
            raise RuntimeError("WHO ICD-11 API is not configured.")

        now = datetime.now(timezone.utc)
        if self._access_token and self._token_expiry and self._token_expiry > now:
            return self._access_token

        response = httpx.post(
            "https://icdaccessmanagement.who.int/connect/token",
            auth=(self.settings.who_client_id, self.settings.who_client_secret),
            data={"grant_type": "client_credentials", "scope": "icdapi_access"},
            timeout=20.0,
        )
        response.raise_for_status()
        payload = response.json()
        self._access_token = payload["access_token"]
        expires_in = int(payload.get("expires_in", 3600))
        self._token_expiry = now + timedelta(seconds=max(expires_in - 60, 60))
        logger.info("WHO ICD-11 token refreshed")
        return self._access_token

    def search(self, query: str, *, limit: int = 3, language: str = "zh") -> dict[str, Any]:
        if not self.is_configured():
            return {"query": query, "matches": []}

        token = self._get_token()
        response = httpx.get(
            "https://id.who.int/icd/release/11/2025-01/mms/search",
            params={"q": query},
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
                "Accept-Language": language,
                "API-Version": "v2",
            },
            timeout=20.0,
        )
        response.raise_for_status()
        payload = response.json()
        raw_matches = payload.get("destinationEntities", [])[:limit]
        matches = []
        for item in raw_matches:
            uri = str(item.get("id") or "").strip()
            detail = self._fetch_entity(uri, token=token, language=language) if uri else {}
            matches.append(
                {
                    "title": item.get("title") or detail.get("title") or query,
                    "code": item.get("theCode") or item.get("code") or detail.get("code"),
                    "uri": uri,
                    "definition": detail.get("definition") or self._extract_text(item.get("definition")),
                    "synonyms": detail.get("synonyms", []),
                    "source": "WHO ICD-11",
                }
            )
        logger.info("WHO ICD-11 search finished for query=%s matches=%s", query, len(matches))
        return {"query": query, "matches": matches}

    def _fetch_entity(self, uri: str, *, token: str, language: str) -> dict[str, Any]:
        if not uri:
            return {}
        entity_url = uri.replace("http://", "https://")
        response = httpx.get(
            entity_url,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
                "Accept-Language": language,
                "API-Version": "v2",
            },
            timeout=20.0,
        )
        response.raise_for_status()
        payload = response.json()
        return {
            "title": payload.get("title"),
            "code": payload.get("theCode") or payload.get("code"),
            "definition": self._extract_text(payload.get("definition")),
            "synonyms": self._extract_values(payload.get("synonym")),
        }

    def _extract_values(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            cleaned = value.strip()
            return [cleaned] if cleaned else []
        if isinstance(value, dict):
            return [text for text in [self._extract_text(value)] if text]
        if isinstance(value, list):
            results: list[str] = []
            for item in value:
                results.extend(self._extract_values(item))
            return results
        return []

    def _extract_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            parts = [self._extract_text(item) for item in value]
            return "；".join(part for part in parts if part)
        if isinstance(value, dict):
            for key in ("@value", "label", "title", "value", "text"):
                text = self._extract_text(value.get(key))
                if text:
                    return text
            parts = [self._extract_text(item) for item in value.values()]
            return "；".join(part for part in parts if part)
        return str(value).strip()


who_service = WHOService()
