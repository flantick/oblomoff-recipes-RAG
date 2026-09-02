"""Estimating the length of a text in tokens.

By default a Cyrillic heuristic is used (CHARS_PER_TOKEN). If the name of an HF
embedder model is passed, the count is exact and made with its tokenizer
(recommended before Step 3, so that chunk borders match the real limit of the
model)."""
from __future__ import annotations

from src.config import CHARS_PER_TOKEN


class TokenCounter:
    def __init__(self, hf_model: str | None = None) -> None:
        self._tok = None
        if hf_model:
            from transformers import AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(hf_model)

    def count(self, text: str) -> int:
        if self._tok is not None:
            return len(self._tok.encode(text, add_special_tokens=False))
        return max(1, round(len(text) / CHARS_PER_TOKEN))
