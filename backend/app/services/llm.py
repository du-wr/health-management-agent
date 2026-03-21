from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from openai import OpenAI

from app.core.config import get_settings


class LLMService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = None
        if self.settings.qwen_api_key:
            self.client = OpenAI(
                api_key=self.settings.qwen_api_key,
                base_url=self.settings.qwen_base_url,
            )

    @property
    def is_configured(self) -> bool:
        return self.client is not None

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        if not self.client:
            raise RuntimeError("Qwen API is not configured.")
        response = self.client.chat.completions.create(
            model=self.settings.qwen_chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        content = response.choices[0].message.content or "{}"
        return json.loads(content)

    def chat_text(self, system_prompt: str, user_prompt: str) -> str:
        if not self.client:
            raise RuntimeError("Qwen API is not configured.")
        response = self.client.chat.completions.create(
            model=self.settings.qwen_chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content or ""

    def image_to_text(self, image_path: Path, prompt: str) -> str:
        if not self.client:
            raise RuntimeError("Qwen API is not configured.")
        encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        suffix = image_path.suffix.lower().lstrip(".") or "png"
        response = self.client.chat.completions.create(
            model=self.settings.qwen_vl_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/{suffix};base64,{encoded}"}},
                    ],
                }
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content or ""


llm_service = LLMService()

