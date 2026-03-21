from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from app.core.config import get_settings


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
        self._token_expiry = now + timedelta(seconds=expires_in - 60)
        return self._access_token

    def search(self, query: str) -> dict[str, Any]:
        if not self.is_configured():
            return {"query": query, "matches": []}
        token = self._get_token()
        response = httpx.get(
            "https://id.who.int/icd/release/11/2025-01/mms/search",
            params={"q": query},
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
                "Accept-Language": "zh",
                "API-Version": "v2",
            },
            timeout=20.0,
        )
        response.raise_for_status()
        payload = response.json()
        return {
            "query": query,
            "matches": [
                {
                    "title": item.get("title"),
                    "code": item.get("theCode") or item.get("code"),
                    "uri": item.get("id"),
                }
                for item in payload.get("destinationEntities", [])[:5]
            ],
        }


who_service = WHOService()
