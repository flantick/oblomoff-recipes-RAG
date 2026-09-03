"""Tests for src/chunking/tokens.py.

transformers.AutoTokenizer is imported lazily inside TokenCounter.__init__
(`from transformers import AutoTokenizer`), so a real HF model is never
downloaded here: a fake module is installed into sys.modules BEFORE the
counter is constructed, the same pattern tests/retrieval/test_rerank.py uses
for FlagEmbedding.
"""
from __future__ import annotations

import sys
import types

import pytest

from src.chunking import tokens as tokens_mod
from src.chunking.tokens import TokenCounter


# --- fake transformers module --------------------------------------------

class FakeHFTokenizer:
    """Fake of a transformers PreTrainedTokenizer.

    encoded_length controls what encode() returns (as a list of that many
    placeholder ids), so a test can make it differ from both the input text's
    character count and the heuristic estimate. encode_calls records the
    (text, kwargs) pairs so tests can check add_special_tokens is forwarded.
    """

    def __init__(self, model_name: str, encoded_length: int = 3) -> None:
        self.model_name = model_name
        self.encoded_length = encoded_length
        self.encode_calls: list[tuple[str, dict]] = []

    def encode(self, text: str, **kwargs) -> list[int]:
        self.encode_calls.append((text, kwargs))
        return list(range(self.encoded_length))


class FakeAutoTokenizer:
    """Fake of transformers.AutoTokenizer.

    Records the model name it was asked for and hands back a single shared
    FakeHFTokenizer instance so the test can inspect its calls afterwards.
    """

    last_instance: FakeHFTokenizer | None = None
    from_pretrained_calls: list[str] = []

    @classmethod
    def from_pretrained(cls, name: str) -> FakeHFTokenizer:
        cls.from_pretrained_calls.append(name)
        cls.last_instance = FakeHFTokenizer(name)
        return cls.last_instance


@pytest.fixture
def fake_transformers(monkeypatch):
    """Replaces sys.modules["transformers"] with a fake module BEFORE
    TokenCounter.__init__ runs its lazy `from transformers import
    AutoTokenizer`."""
    FakeAutoTokenizer.from_pretrained_calls = []
    FakeAutoTokenizer.last_instance = None
    fake_module = types.ModuleType("transformers")
    fake_module.AutoTokenizer = FakeAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake_module)
    return FakeAutoTokenizer


# --- heuristic branch (hf_model=None) -------------------------------------

def test_count_heuristic_typical_text_rounds_to_nearest_token(monkeypatch):
    """13 chars / 2.5 chars-per-token = 5.2 -> rounds to 5, as a plain int.

    CHARS_PER_TOKEN is pinned rather than read: the constant is a tunable
    heuristic, and retuning it must not turn this test red while the formula
    it checks is still correct.
    """
    monkeypatch.setattr(tokens_mod, "CHARS_PER_TOKEN", 2.5)
    counter = TokenCounter()
    result = counter.count("x" * 13)
    assert result == 5
    assert isinstance(result, int) and not isinstance(result, bool)


def test_count_heuristic_empty_string_returns_one_not_zero():
    """max(1, ...) floors the estimate at 1 token even for an empty string."""
    counter = TokenCounter()
    assert counter.count("") == 1


@pytest.mark.parametrize(
    "text",
    ["x", "xy"],
    ids=["one_char", "two_chars"],
)
def test_count_heuristic_very_short_text_returns_one(monkeypatch, text):
    """1 char (0.4 -> round 0) and 2 chars (0.8 -> round 1) both clamp to 1."""
    monkeypatch.setattr(tokens_mod, "CHARS_PER_TOKEN", 2.5)
    counter = TokenCounter()
    assert counter.count(text) == 1


def test_count_heuristic_uses_banker_rounding_at_half(monkeypatch):
    """Python's round() rounds half-to-even: with CHARS_PER_TOKEN patched to
    2.0, 5 chars / 2.0 = 2.5 rounds to 2 (even), not 3 (as round-half-up
    would give). This is unclamped by max(1, ...), so it isolates round()'s
    actual behaviour rather than the floor."""
    monkeypatch.setattr(tokens_mod, "CHARS_PER_TOKEN", 2.0)
    counter = TokenCounter()
    assert counter.count("x" * 5) == 2


def test_heuristic_mode_never_constructs_a_tokenizer():
    """hf_model=None counts by the character heuristic instead of loading a
    tokenizer: 1000 chars / 2.5 = 400, a number no tokenizer would return for
    a run of identical characters."""
    monkeypatch_free_counter = TokenCounter(hf_model=None)
    assert monkeypatch_free_counter.count("x" * 1000) == 400


def test_empty_hf_model_string_falls_back_to_the_heuristic():
    """hf_model="" is falsy, so the `if hf_model:` guard treats it as "no
    model" and the counter stays on the heuristic rather than asking
    from_pretrained for a model named "". Pinned because the guard tests
    truthiness, not `is not None`."""
    counter = TokenCounter(hf_model="")
    assert counter.count("x" * 1000) == 400


# --- HF branch (hf_model="...") -------------------------------------------

def test_hf_model_name_is_forwarded_to_from_pretrained(fake_transformers):
    """The hf_model string reaches AutoTokenizer.from_pretrained unchanged."""
    TokenCounter(hf_model="BAAI/bge-m3")
    assert fake_transformers.from_pretrained_calls == ["BAAI/bge-m3"]


def test_hf_count_returns_encode_length_not_char_length(fake_transformers):
    """count() uses len(encode(...)), not the heuristic char-based estimate:
    a long text with a short fake encoding must report the short length."""
    counter = TokenCounter(hf_model="BAAI/bge-m3")
    fake_transformers.last_instance.encoded_length = 7
    long_text = "x" * 1000  # heuristic would give 400 tokens, encode gives 7
    assert counter.count(long_text) == 7


def test_hf_count_passes_add_special_tokens_false(fake_transformers):
    """Special tokens would inflate the chunk budget, so encode() must be
    called with add_special_tokens=False."""
    counter = TokenCounter(hf_model="BAAI/bge-m3")
    counter.count("текст рецепта")
    _, kwargs = fake_transformers.last_instance.encode_calls[0]
    assert kwargs["add_special_tokens"] is False


def test_hf_empty_string_returns_zero_without_min_clamp(fake_transformers):
    """Unlike the heuristic branch, the HF branch applies no max(1, ...)
    floor: an empty string that encodes to zero ids reports 0 tokens."""
    counter = TokenCounter(hf_model="BAAI/bge-m3")
    fake_transformers.last_instance.encoded_length = 0
    assert counter.count("") == 0
