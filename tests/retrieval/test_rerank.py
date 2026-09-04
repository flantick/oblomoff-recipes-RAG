"""Tests for src/retrieval/rerank.py: the use_fp16 decision in
Reranker.__init__, pair building in score() and the error handling of
try_load_reranker()."""
from __future__ import annotations

import sys
import types

import pytest

from src.config import RERANK_MODEL
from src.retrieval import rerank as rerank_mod
from src.retrieval.rerank import Reranker, try_load_reranker


class FakeFlagReranker:
    """Fake of FlagEmbedding.FlagReranker.

    Records the arguments it was constructed with (model_name, kwargs) and the
    compute_score calls (pairs, normalize). raw_score is configured AFTER
    construction: Reranker.__init__ only creates the model, it never calls it.
    """

    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self.compute_score_calls: list[dict] = []
        self.raw_score: float | list[float] = []

    def compute_score(self, pairs, normalize=False):
        self.compute_score_calls.append({"pairs": pairs, "normalize": normalize})
        return self.raw_score


@pytest.fixture
def fake_flagembedding(monkeypatch):
    """Replaces sys.modules["FlagEmbedding"] with a fake module BEFORE
    Reranker.__init__ runs its lazy `from FlagEmbedding import FlagReranker`."""
    fake_module = types.ModuleType("FlagEmbedding")
    fake_module.FlagReranker = FakeFlagReranker
    monkeypatch.setitem(sys.modules, "FlagEmbedding", fake_module)
    return fake_module


@pytest.fixture(autouse=True)
def _no_fp16_config(monkeypatch):
    """RERANK_USE_FP16 is unset by default — the tests must not depend on the
    environment variables of the real run."""
    monkeypatch.setattr(rerank_mod, "RERANK_USE_FP16", None)


# --- Reranker.__init__: the use_fp16 decision ---------------------------

def test_fp16_off_on_cpu_by_default(fake_flagembedding):
    """device=cpu with neither use_fp16 nor the config set -> fp16 is off."""
    r = Reranker(device="cpu", use_fp16=None)
    assert r.model.kwargs["use_fp16"] is False


@pytest.mark.parametrize("device", ["cuda", None], ids=["cuda", "none"])
def test_fp16_on_when_not_cpu_by_default(fake_flagembedding, device):
    """device=cuda or unset, with neither use_fp16 nor the config -> fp16 is on."""
    r = Reranker(device=device, use_fp16=None)
    assert r.model.kwargs["use_fp16"] is True


def test_fp16_config_overrides_cpu(fake_flagembedding, monkeypatch):
    """RERANK_USE_FP16="1" turns fp16 on even on the CPU."""
    monkeypatch.setattr(rerank_mod, "RERANK_USE_FP16", "1")
    r = Reranker(device="cpu", use_fp16=None)
    assert r.model.kwargs["use_fp16"] is True


@pytest.mark.parametrize("value", ["0", "true", "yes", ""], ids=["zero", "true", "yes", "empty"])
def test_fp16_config_off_for_any_value_other_than_one(
    fake_flagembedding, monkeypatch, value
):
    """Only the exact string "1" enables fp16: every other value, "true"
    included, silently means off — even on the GPU."""
    monkeypatch.setattr(rerank_mod, "RERANK_USE_FP16", value)
    r = Reranker(device="cuda", use_fp16=None)
    assert r.model.kwargs["use_fp16"] is False


@pytest.mark.parametrize(
    "device, config, explicit",
    [
        ("cpu", "1", True),
        ("cuda", "0", False),
    ],
    ids=[
        "explicit_true_overrides_cpu_and_config_off",
        "explicit_false_overrides_gpu_and_config_on",
    ],
)
def test_fp16_explicit_arg_overrides_device_and_config(
    fake_flagembedding, monkeypatch, device, config, explicit
):
    """An explicit use_fp16 beats both the device and RERANK_USE_FP16."""
    monkeypatch.setattr(rerank_mod, "RERANK_USE_FP16", config)
    r = Reranker(device=device, use_fp16=explicit)
    assert r.model.kwargs["use_fp16"] is explicit


def test_fp16_off_when_device_uppercase_cpu(fake_flagembedding):
    """The device string is case-insensitive: "CPU" counts as cpu too."""
    r = Reranker(device="CPU", use_fp16=None)
    assert r.model.kwargs["use_fp16"] is False


def test_devices_kwarg_present_when_device_given(fake_flagembedding):
    """A given device reaches the model kwargs as devices=<device>."""
    r = Reranker(device="cuda:0", use_fp16=False)
    assert r.model.kwargs["devices"] == "cuda:0"


def test_devices_kwarg_absent_when_device_none(fake_flagembedding):
    """device=None -> the model kwargs carry no devices key at all."""
    r = Reranker(device=None, use_fp16=False)
    assert "devices" not in r.model.kwargs


def test_model_name_passed_to_flagreranker(fake_flagembedding):
    """model_name is forwarded to FlagReranker as a positional argument."""
    r = Reranker(model_name="BAAI/bge-reranker-custom", device="cpu", use_fp16=False)
    assert r.model.model_name == "BAAI/bge-reranker-custom"


# --- Reranker.score -------------------------------------------------------

def test_score_empty_texts_returns_empty_without_calling_model(fake_flagembedding):
    """An empty text list -> [] and compute_score is never reached."""
    r = Reranker(device="cpu", use_fp16=False)
    result = r.score("запрос", [])
    assert result == []
    assert r.model.compute_score_calls == []


def test_score_builds_pairs_in_order(fake_flagembedding):
    """Pairs are built as [query, text], one per text, in the original order."""
    r = Reranker(device="cpu", use_fp16=False)
    r.model.raw_score = [0.1, 0.2, 0.3]
    r.score("запрос", ["первый", "второй", "третий"])
    assert r.model.compute_score_calls[0]["pairs"] == [
        ["запрос", "первый"],
        ["запрос", "второй"],
        ["запрос", "третий"],
    ]


def test_score_wraps_bare_float_in_list(fake_flagembedding):
    """compute_score returning a bare float (not a list) -> [that number]."""
    r = Reranker(device="cpu", use_fp16=False)
    r.model.raw_score = 0.42
    result = r.score("запрос", ["единственный текст"])
    assert result == pytest.approx([0.42])


def test_score_casts_list_elements_to_float(fake_flagembedding):
    """compute_score returning a list -> every element is cast to float."""
    r = Reranker(device="cpu", use_fp16=False)
    r.model.raw_score = [0, 1]
    result = r.score("запрос", ["a", "b"])
    assert result == [0.0, 1.0]
    assert all(isinstance(x, float) for x in result)


def test_score_passes_normalize_true(fake_flagembedding):
    """normalize=True is always forwarded to compute_score."""
    r = Reranker(device="cpu", use_fp16=False)
    r.model.raw_score = [0.5]
    r.score("запрос", ["текст"])
    assert r.model.compute_score_calls[0]["normalize"] is True


# --- try_load_reranker -----------------------------------------------------

def test_try_load_reranker_builds_the_model_named_in_the_config(fake_flagembedding):
    """A constructor that succeeds -> a loaded Reranker comes back, built with
    the model from the config rather than one hardcoded in try_load_reranker."""
    result = try_load_reranker()
    assert result.model.model_name == RERANK_MODEL


def test_try_load_reranker_returns_none_on_exception(monkeypatch):
    """A constructor that raises -> None comes back, the exception is swallowed."""
    fake_module = types.ModuleType("FlagEmbedding")

    class BrokenFlagReranker:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("модель недоступна")

    fake_module.FlagReranker = BrokenFlagReranker
    monkeypatch.setitem(sys.modules, "FlagEmbedding", fake_module)

    result = try_load_reranker()
    assert result is None
