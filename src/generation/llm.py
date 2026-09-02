"""LLM client: the OpenAI-compatible API of vLLM (Step 5).

The code is provider-agnostic — the same client works with Ollama (/v1),
llama.cpp server and the like; it is enough to change LLM_BASE_URL / LLM_MODEL.
"""
from __future__ import annotations

import json

from loguru import logger

from src.config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
)


class LLMClient:
    def __init__(
        self,
        *,
        base_url: str = LLM_BASE_URL,
        api_key: str = LLM_API_KEY,
        model: str = LLM_MODEL,
        timeout: float = LLM_TIMEOUT,
    ) -> None:
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model

    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
        json_mode: bool = False,
    ) -> str:
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            if json_mode and "response_format" in str(exc):
                logger.warning("response_format не поддержан бэкендом — падаю на обычный режим")
                kwargs.pop("response_format", None)
                resp = self.client.chat.completions.create(**kwargs)
            else:
                raise
        return resp.choices[0].message.content or ""

    def chat_json(self, messages: list[dict], **kw) -> dict:
        """JSON mode + resilient parsing (strip ``` fences, retry once)."""
        raw = self.chat(messages, json_mode=True, **kw)
        obj = _loads_lenient(raw)
        if obj is not None:
            return obj
        logger.warning("LLM вернул невалидный JSON, повторяю запрос строже")
        retry = messages + [
            {"role": "assistant", "content": raw[:2000]},
            {"role": "user", "content": "Это был невалидный JSON. Верни ТОЛЬКО корректный JSON-объект по схеме."},
        ]
        raw2 = self.chat(retry, json_mode=True, **kw)
        obj = _loads_lenient(raw2)
        if obj is None:
            raise ValueError(f"LLM не вернул валидный JSON: {raw2[:500]!r}")
        return obj

    def health(self) -> bool:
        try:
            self.client.models.list()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("LLM недоступен на {}: {}", self.client.base_url, exc)
            return False


def _loads_lenient(text: str) -> dict | None:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text[:4].lower() == "json":
            text = text[4:]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        a, b = text.find("{"), text.rfind("}")
        if 0 <= a < b:
            try:
                return json.loads(text[a:b + 1])
            except json.JSONDecodeError:
                return None
    return None
