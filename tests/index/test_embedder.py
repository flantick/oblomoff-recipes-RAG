"""Tests for src/index/embedder.py: passage_text() and the device/fp16 logic
plus payload shaping in BGEM3Embedder."""
from __future__ import annotations

import sys
import types

import pytest

from src.config import EMBED_MODEL
from src.index.embedder import BGEM3Embedder, passage_text


class FakeBGEM3FlagModel:
    """Fake of FlagEmbedding.BGEM3FlagModel.

    Records the constructor arguments (model_name, kwargs) and every encode()
    call. The dict returned by encode() is configured AFTER construction via
    encode_result, since BGEM3Embedder.__init__ only builds the model and
    never calls it.
    """

    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self.encode_calls: list[dict] = []
        self.encode_result: dict = {"dense_vecs": [], "lexical_weights": []}

    def encode(self, texts, **kwargs):
        self.encode_calls.append({"texts": texts, **kwargs})
        return self.encode_result


@pytest.fixture
def fake_flagembedding(monkeypatch):
    """Replaces sys.modules["FlagEmbedding"] with a fake module BEFORE
    BGEM3Embedder.__init__ runs its lazy `from FlagEmbedding import
    BGEM3FlagModel`."""
    fake_module = types.ModuleType("FlagEmbedding")
    fake_module.BGEM3FlagModel = FakeBGEM3FlagModel
    monkeypatch.setitem(sys.modules, "FlagEmbedding", fake_module)
    return fake_module


# --- passage_text ----------------------------------------------------------

def test_passage_text_joins_title_and_text_with_blank_line():
    """A chunk with both fields -> "title\\n\\ntext", so the dish name is
    embedded into every chunk of the video, not just the intro."""
    chunk = {"title": "Борщ", "text": "Обжарьте лук и морковь."}
    assert passage_text(chunk) == "Борщ\n\nОбжарьте лук и морковь."


def test_passage_text_no_leading_newlines_when_title_empty():
    """An empty title -> only the text comes back, with no leading blank
    line (the naive f-string would otherwise prefix "\\n\\n")."""
    chunk = {"title": "", "text": "Обжарьте лук и морковь."}
    assert passage_text(chunk) == "Обжарьте лук и морковь."


def test_passage_text_whitespace_only_title_treated_as_absent():
    """A title made only of whitespace is stripped to "" and therefore
    counts as absent, per the explicit .strip() in the source."""
    chunk = {"title": "   \t  ", "text": "Текст рецепта."}
    assert passage_text(chunk) == "Текст рецепта."


def test_passage_text_missing_title_key_does_not_raise():
    """No "title" key at all -> treated the same as an empty title."""
    chunk = {"text": "Текст рецепта."}
    assert passage_text(chunk) == "Текст рецепта."


def test_passage_text_missing_text_key_returns_title_plus_empty_body():
    """No "text" key -> the missing body becomes "", not a KeyError."""
    chunk = {"title": "Борщ"}
    assert passage_text(chunk) == "Борщ\n\n"


def test_passage_text_title_none_does_not_raise():
    """title=None (e.g. from a sparse payload) -> treated as absent, not a
    TypeError from calling .strip() on None."""
    chunk = {"title": None, "text": "Текст рецепта."}
    assert passage_text(chunk) == "Текст рецепта."


# --- BGEM3Embedder.__init__: device / fp16 decision ------------------------

def test_fp16_forced_off_on_cpu_even_if_requested(fake_flagembedding):
    """device="cpu" -> use_fp16 is forced False regardless of the argument,
    because fp16 is not supported on CPU."""
    emb = BGEM3Embedder(device="cpu", use_fp16=True)
    assert emb.model.kwargs["use_fp16"] is False


def test_fp16_kept_as_given_on_cuda(fake_flagembedding):
    """device="cuda" -> the requested use_fp16 value passes through
    unchanged."""
    emb = BGEM3Embedder(device="cuda", use_fp16=True)
    assert emb.model.kwargs["use_fp16"] is True


def test_devices_kwarg_present_when_device_given(fake_flagembedding):
    """A given device reaches the model kwargs as devices=<device>."""
    emb = BGEM3Embedder(device="cuda:0", use_fp16=False)
    assert emb.model.kwargs["devices"] == "cuda:0"


def test_devices_kwarg_absent_when_device_none(fake_flagembedding):
    """device=None -> the model kwargs carry no devices key at all."""
    emb = BGEM3Embedder(device=None, use_fp16=False)
    assert "devices" not in emb.model.kwargs


def test_fp16_off_when_device_uppercase_cpu(fake_flagembedding):
    """The device string is case-insensitive: "CPU" is treated as cpu too."""
    emb = BGEM3Embedder(device="CPU", use_fp16=True)
    assert emb.model.kwargs["use_fp16"] is False


def test_model_name_passed_to_flagmodel(fake_flagembedding):
    """model_name is forwarded to BGEM3FlagModel as a positional argument."""
    emb = BGEM3Embedder(model_name="BAAI/bge-m3-custom", device="cpu", use_fp16=False)
    assert emb.model.model_name == "BAAI/bge-m3-custom"


def test_max_length_and_batch_size_stored_on_instance(fake_flagembedding):
    """max_length and batch_size are kept on the embedder object for reuse in
    _encode()."""
    emb = BGEM3Embedder(device="cpu", use_fp16=False, max_length=512, batch_size=4)
    assert emb.max_length == 512
    assert emb.batch_size == 4


def test_default_model_name_is_config_value(fake_flagembedding):
    """No model_name given -> the model built is the one from src.config, not
    a hardcoded literal."""
    emb = BGEM3Embedder(device="cpu", use_fp16=False)
    assert emb.model.model_name == EMBED_MODEL


# --- BGEM3Embedder._encode / encode_passages / encode_queries --------------

def test_encode_empty_list_returns_empty_without_calling_model(fake_flagembedding):
    """An empty text list -> [] and model.encode is never reached."""
    emb = BGEM3Embedder(device="cpu", use_fp16=False)
    result = emb.encode_passages([])
    assert result == []
    assert emb.model.encode_calls == []


def test_encode_forwards_batch_size_and_max_length(fake_flagembedding):
    """batch_size and max_length stored on the embedder reach model.encode()."""
    emb = BGEM3Embedder(device="cpu", use_fp16=False, max_length=256, batch_size=7)
    emb.model.encode_result = {
        "dense_vecs": [[0.1, 0.2]],
        "lexical_weights": [{}],
    }
    emb.encode_passages(["текст"])
    call = emb.model.encode_calls[0]
    assert call["batch_size"] == 7
    assert call["max_length"] == 256


def test_encode_requests_dense_and_sparse_but_not_colbert(fake_flagembedding):
    """return_dense=True, return_sparse=True, return_colbert_vecs=False are
    forwarded — colbert vectors are unused and would waste memory."""
    emb = BGEM3Embedder(device="cpu", use_fp16=False)
    emb.model.encode_result = {"dense_vecs": [[0.0]], "lexical_weights": [{}]}
    emb.encode_passages(["текст"])
    call = emb.model.encode_calls[0]
    assert call["return_dense"] is True
    assert call["return_sparse"] is True
    assert call["return_colbert_vecs"] is False


def test_encode_casts_dense_vector_to_float_list(fake_flagembedding):
    """A dense vector of ints/numpy-like scalars is cast to a plain list of
    Python floats."""
    emb = BGEM3Embedder(device="cpu", use_fp16=False)
    emb.model.encode_result = {"dense_vecs": [[1, 2, 3]], "lexical_weights": [{}]}
    result = emb.encode_passages(["текст"])
    assert result[0].dense == pytest.approx([1.0, 2.0, 3.0])
    assert all(isinstance(x, float) for x in result[0].dense)


def test_encode_casts_sparse_keys_and_values(fake_flagembedding):
    """Sparse weight keys (token ids) are cast to int, values to float."""
    emb = BGEM3Embedder(device="cpu", use_fp16=False)
    emb.model.encode_result = {
        "dense_vecs": [[0.0]],
        "lexical_weights": [{"5": 0.7}],
    }
    result = emb.encode_passages(["текст"])
    assert result[0].sparse_indices == [5]
    assert result[0].sparse_values == pytest.approx([0.7])
    assert isinstance(result[0].sparse_indices[0], int)
    assert isinstance(result[0].sparse_values[0], float)


@pytest.mark.parametrize(
    "weight, kept",
    [
        pytest.param(0.5, True, id="positive_kept"),
        pytest.param(0.0, False, id="zero_dropped"),
        pytest.param(-0.1, False, id="negative_dropped"),
    ],
)
def test_encode_sparse_weight_threshold(fake_flagembedding, weight, kept):
    """A sparse weight <= 0.0 is dropped; a strictly positive one is kept —
    exercises the `if fv <= 0.0: continue` boundary from both sides."""
    emb = BGEM3Embedder(device="cpu", use_fp16=False)
    emb.model.encode_result = {
        "dense_vecs": [[0.0]],
        "lexical_weights": [{1: weight}],
    }
    result = emb.encode_passages(["текст"])
    if kept:
        assert result[0].sparse_indices == [1]
        assert result[0].sparse_values == pytest.approx([weight])
    else:
        assert result[0].sparse_indices == []
        assert result[0].sparse_values == []


def test_encode_multiple_texts_preserve_order(fake_flagembedding):
    """Several texts -> one Embedding per text, in the same order as the
    model returned them."""
    emb = BGEM3Embedder(device="cpu", use_fp16=False)
    emb.model.encode_result = {
        "dense_vecs": [[1.0], [2.0], [3.0]],
        "lexical_weights": [{}, {}, {}],
    }
    result = emb.encode_passages(["раз", "два", "три"])
    assert [e.dense[0] for e in result] == [1.0, 2.0, 3.0]


def test_encode_passages_and_queries_produce_identical_embeddings(fake_flagembedding):
    """BGE-M3 is symmetric: the same text encodes the same way whether it is
    indexed or queried, because neither side adds a prefix.

    Asserted on the output rather than on the two attributes being one object,
    so splitting them into two thin wrappers stays legal while adding a prefix
    to only one of them does not.
    """
    emb = BGEM3Embedder(device="cpu", use_fp16=False)
    emb.model.encode_result = {
        "dense_vecs": [[1.0, 2.0]],
        "lexical_weights": [{"7": 0.5}],
    }

    assert emb.encode_passages(["стейк"]) == emb.encode_queries(["стейк"])
