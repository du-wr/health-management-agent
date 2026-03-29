from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Iterator

from openai import OpenAI

from app.core.config import get_settings


class LLMService:
    """统一封装大模型调用。

    项目里约定了三类模型职责：
    - fast: 轻量分析、改写、结构化任务
    - max: 最终高质量回答和润色
    - vl: 图片 OCR
    """

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

    @property
    def fast_model(self) -> str:
        return self.settings.qwen_fast_model or self.settings.qwen_chat_model

    @property
    def max_model(self) -> str:
        return self.settings.qwen_max_model or self.settings.qwen_chat_model

    def _resolve_model(self, model: str | None = None) -> str:
        return model or self.settings.qwen_chat_model

    def chat_json(self, system_prompt: str, user_prompt: str, *, model: str | None = None) -> dict[str, Any]:
        """要求模型输出 JSON 对象。"""
        if not self.client:
            raise RuntimeError("Qwen API is not configured.")
        response = self.client.chat.completions.create(
            model=self._resolve_model(model),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        content = response.choices[0].message.content or "{}"
        return json.loads(content)

    def chat_json_fast(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """快模型版 JSON 调用。"""
        return self.chat_json(system_prompt, user_prompt, model=self.fast_model)

    def chat_json_max(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """强模型版 JSON 调用。"""
        return self.chat_json(system_prompt, user_prompt, model=self.max_model)

    def chat_text(self, system_prompt: str, user_prompt: str, *, model: str | None = None) -> str:
        """普通文本生成。"""
        if not self.client:
            raise RuntimeError("Qwen API is not configured.")
        response = self.client.chat.completions.create(
            model=self._resolve_model(model),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content or ""

    def chat_text_fast(self, system_prompt: str, user_prompt: str) -> str:
        """快模型文本生成。"""
        return self.chat_text(system_prompt, user_prompt, model=self.fast_model)

    def chat_text_max(self, system_prompt: str, user_prompt: str) -> str:
        """强模型文本生成。"""
        return self.chat_text(system_prompt, user_prompt, model=self.max_model)

    def chat_text_stream(self, system_prompt: str, user_prompt: str, *, model: str | None = None) -> Iterator[str]:
        """流式文本生成。前端的打字机效果建立在这里。"""
        if not self.client:
            raise RuntimeError("Qwen API is not configured.")
        stream = self.client.chat.completions.create(
            model=self._resolve_model(model),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield delta

    def chat_text_stream_max(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        """强模型流式文本生成。"""
        return self.chat_text_stream(system_prompt, user_prompt, model=self.max_model)

    def image_to_text(self, image_path: Path, prompt: str) -> str:
        """把图片类报告交给视觉模型做 OCR。"""
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
