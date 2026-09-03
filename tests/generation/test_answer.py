"""Tests for src/generation/answer.py: _sources_from_citations and the
answer() pipeline (retrieval -> min-score gate -> LLM -> RecipeAnswer).

The outside world here is retrieval and the LLM, both injected through
answer()'s own seams (retriever=, llm=); the fakes come from tests/conftest.py.

RETRIEVAL_MIN_SCORE is read from the environment at import time (config.py
calls load_dotenv), so every test in this file runs against a pinned threshold
rather than whatever the developer's .env happens to hold.
"""
from __future__ import annotations

import pytest

from src.generation.answer import _sources_from_citations, answer
from src.generation.schemas import SourceRef
from src.retrieval.schemas import RetrievedVideo
from tests.conftest import FakeLLM, FakeRetriever

MIN_SCORE = 0.5


@pytest.fixture(autouse=True)
def _pinned_min_score(monkeypatch):
    """Pins the relevance gate so the suite does not depend on a local .env."""
    monkeypatch.setattr("src.generation.answer.RETRIEVAL_MIN_SCORE", MIN_SCORE)


def make_video(score: float = 1.0, video_id: str = "v1") -> RetrievedVideo:
    return RetrievedVideo(video_id=video_id, title="Видео", score=score)


def make_citation(n: int = 1, **overrides) -> dict:
    citation = {
        "n": n,
        "video_id": "v1",
        "title": "Стейк",
        "url": "https://youtu.be/v1?t=60",
        "timecode": "01:00",
    }
    citation.update(overrides)
    return citation


# =======================================================================
# _sources_from_citations
# =======================================================================
def test_sources_from_citations_maps_all_fields():
    """A full citation dict is mapped field by field into a SourceRef."""
    result = _sources_from_citations([make_citation()])
    assert result == [
        SourceRef(n=1, video_id="v1", title="Стейк",
                  url="https://youtu.be/v1?t=60", timecode="01:00")
    ]


def test_sources_from_citations_missing_keys_default_to_empty_string():
    """A citation with only "n" gets "" for every other field."""
    result = _sources_from_citations([{"n": 3}])
    assert result == [SourceRef(n=3, video_id="", title="", url="", timecode="")]


def test_sources_from_citations_empty_list_returns_empty_list():
    """An empty citation list -> an empty SourceRef list."""
    assert _sources_from_citations([]) == []


# =======================================================================
# answer(): dependencies default to the module-level singletons
# =======================================================================
def test_answer_falls_back_to_the_module_defaults_when_nothing_is_injected(monkeypatch):
    """Called without retriever/llm — the way ask.py and the FastAPI layer call
    it — answer() takes get_retriever() and LLMClient()."""
    retriever = FakeRetriever(citations=[], videos=[])
    llm = FakeLLM(fail_on_call=True, model="default-model")
    monkeypatch.setattr("src.generation.answer.get_retriever", lambda: retriever)
    monkeypatch.setattr("src.generation.answer.LLMClient", lambda: llm)

    result = answer("запрос")

    assert result.model == "default-model"
    assert retriever.calls[0]["query"] == "запрос"


# =======================================================================
# answer(): the empty-citations early return
# =======================================================================
def test_answer_empty_citations_returns_not_found_without_calling_llm():
    """No citations at all -> found=false, sources=[], and the LLM is never
    reached (FakeLLM(fail_on_call=True) would raise if it were)."""
    retriever = FakeRetriever(citations=[], videos=[])
    llm = FakeLLM(fail_on_call=True, model="my-model")

    result = answer("запрос", retriever=retriever, llm=llm)

    assert result.found is False
    assert result.sources == []
    assert result.model == "my-model"
    assert result.used_reranker is False


def test_answer_echoes_used_reranker_from_retrieval_result():
    """used_reranker in the answer mirrors the RetrievalResult's flag."""
    retriever = FakeRetriever(citations=[], videos=[], used_reranker=True)
    llm = FakeLLM(fail_on_call=True)

    result = answer("запрос", retriever=retriever, llm=llm)

    assert result.used_reranker is True


# =======================================================================
# answer(): the min-score gate
# =======================================================================
def test_answer_skips_llm_below_min_score():
    """top_score below RETRIEVAL_MIN_SCORE -> the LLM is not called, but the
    sources are still filled in and found=false."""
    retriever = FakeRetriever(citations=[make_citation()], videos=[make_video(score=0.4)])
    llm = FakeLLM(fail_on_call=True)

    result = answer("запрос", retriever=retriever, llm=llm)

    assert result.found is False
    assert result.sources == [
        SourceRef(n=1, video_id="v1", title="Стейк",
                  url="https://youtu.be/v1?t=60", timecode="01:00")
    ]


def test_answer_calls_llm_when_score_exactly_at_min_score():
    """The gate is a strict less-than: a score exactly at the threshold still
    reaches the LLM, and its recipe comes back."""
    retriever = FakeRetriever(citations=[make_citation()], videos=[make_video(score=MIN_SCORE)])
    llm = FakeLLM(payload={"found": True, "dish": "Стейк"})

    result = answer("запрос", retriever=retriever, llm=llm)

    assert result.dish == "Стейк"


def test_answer_gate_looks_at_the_best_video_not_the_worst():
    """The gate compares the threshold against the top-scoring video: one
    strong video is enough even when a weak one sits alongside it."""
    retriever = FakeRetriever(
        citations=[make_citation()],
        videos=[make_video(score=0.1, video_id="weak"),
                make_video(score=0.9, video_id="strong")],
    )
    llm = FakeLLM(payload={"found": True, "dish": "Стейк"})

    result = answer("запрос", retriever=retriever, llm=llm)

    assert result.dish == "Стейк"


def test_answer_renders_not_found_when_there_are_citations_but_no_videos():
    """An empty video list with non-empty citations must not raise on
    max(..., default=0.0): the score of 0.0 falls under the gate, so the answer
    comes back not-found with the citations still attached."""
    retriever = FakeRetriever(citations=[make_citation()], videos=[])
    llm = FakeLLM(fail_on_call=True)

    result = answer("запрос", retriever=retriever, llm=llm)

    assert result.found is False
    assert [s.n for s in result.sources] == [1]


# =======================================================================
# answer(): the LLM's reply fails schema validation
# =======================================================================
def test_answer_llm_invalid_schema_returns_not_found():
    """A reply missing the required "found" field raises a pydantic
    ValidationError inside answer(); it must not escape, and the result must
    still carry found=false with sources filled in."""
    retriever = FakeRetriever(citations=[make_citation()], videos=[make_video(score=1.0)])
    llm = FakeLLM(payload={"dish": "стейк"})  # no "found" key

    result = answer("запрос", retriever=retriever, llm=llm)

    assert result.found is False
    assert [s.n for s in result.sources] == [1]


def test_answer_lets_a_failed_llm_call_propagate():
    """A ValueError out of chat_json (two invalid JSON replies in a row) is NOT
    caught by answer(): only a schema mismatch is downgraded to found=false.
    The FastAPI layer is what turns this into a 502."""
    retriever = FakeRetriever(citations=[make_citation()], videos=[make_video(score=1.0)])
    llm = FakeLLM(error=ValueError("LLM did not return valid JSON"))

    with pytest.raises(ValueError):
        answer("запрос", retriever=retriever, llm=llm)


# =======================================================================
# answer(): a successful reply
# =======================================================================
def test_answer_success_maps_llm_recipe_fields():
    """dish/ingredients/steps/notes from the LLM's LLMRecipe are carried over
    into the RecipeAnswer verbatim."""
    retriever = FakeRetriever(citations=[make_citation()], videos=[make_video(score=1.0)])
    llm = FakeLLM(payload={
        "found": True,
        "dish": "Стейк рибай",
        "ingredients": ["стейк 300г", "соль"],
        "steps": ["обжарить с двух сторон"],
        "source_n": 1,
        "notes": "дать отдохнуть 5 минут",
    })

    result = answer("запрос", retriever=retriever, llm=llm)

    assert result.found is True
    assert result.dish == "Стейк рибай"
    assert result.ingredients == ["стейк 300г", "соль"]
    assert result.steps == ["обжарить с двух сторон"]
    assert result.notes == "дать отдохнуть 5 минут"


# =======================================================================
# answer(): selecting the primary source
# =======================================================================
def test_answer_source_n_matches_existing_citation():
    """source_n pointing at a real citation number selects exactly that one."""
    retriever = FakeRetriever(
        citations=[make_citation(1), make_citation(2, title="Другое видео")],
        videos=[make_video(score=1.0)],
    )
    llm = FakeLLM(payload={"found": True, "source_n": 2})

    result = answer("запрос", retriever=retriever, llm=llm)

    assert result.source == SourceRef(
        n=2, video_id="v1", title="Другое видео",
        url="https://youtu.be/v1?t=60", timecode="01:00",
    )


def test_answer_found_true_source_n_none_defaults_to_first_source():
    """found=true with source_n=None falls back to sources[0]."""
    retriever = FakeRetriever(
        citations=[make_citation(1), make_citation(2, title="Другое видео")],
        videos=[make_video(score=1.0)],
    )
    llm = FakeLLM(payload={"found": True, "source_n": None})

    result = answer("запрос", retriever=retriever, llm=llm)

    assert result.source.n == 1


def test_answer_found_true_source_n_unknown_defaults_to_first_source():
    """found=true with a source_n that matches no citation also falls back to
    sources[0], rather than leaving source empty."""
    retriever = FakeRetriever(
        citations=[make_citation(1), make_citation(2, title="Другое видео")],
        videos=[make_video(score=1.0)],
    )
    llm = FakeLLM(payload={"found": True, "source_n": 99})

    result = answer("запрос", retriever=retriever, llm=llm)

    assert result.source.n == 1


def test_answer_found_false_source_stays_none_even_with_sources():
    """found=false with no usable source_n keeps source=None, and the
    sources[0] fallback is not applied since it only fires when found."""
    retriever = FakeRetriever(citations=[make_citation()], videos=[make_video(score=1.0)])
    llm = FakeLLM(payload={"found": False, "source_n": None})

    result = answer("запрос", retriever=retriever, llm=llm)

    assert result.source is None
    assert [s.n for s in result.sources] == [1]


def test_answer_source_n_zero_currently_falls_back_to_first_source():
    """Pins today's behaviour on source_n=0: the truthiness check in
    answer.py:78 skips the lookup, so the fallback wins. Paired with the xfail
    below — that one says what SHOULD happen, this one catches any change to
    what DOES happen."""
    retriever = FakeRetriever(
        citations=[make_citation(1, title="Другое видео"), make_citation(0, title="Нулевой фрагмент")],
        videos=[make_video(score=1.0)],
    )
    llm = FakeLLM(payload={"found": True, "source_n": 0})

    result = answer("запрос", retriever=retriever, llm=llm)

    assert result.source.n == 1
    assert result.source.title == "Другое видео"


@pytest.mark.xfail(
    reason="answer.py:78 - `by_n.get(rec.source_n) if rec.source_n else None` "
    "tests source_n for truthiness instead of `is not None`. source_n=0 is a "
    "valid citation number but is falsy, so the lookup is skipped and the code "
    "falls through to the sources[0] fallback, silently discarding the LLM's "
    "actual choice. Latent rather than live: _build_context numbers citations "
    "with enumerate(chosen, 1), so a real Retriever never emits n=0.",
    strict=True,
)
def test_answer_keeps_source_when_source_n_is_zero():
    """source_n=0 pointing at a real citation numbered 0 should select that
    citation, not fall back to a different one."""
    retriever = FakeRetriever(
        citations=[make_citation(1, title="Другое видео"), make_citation(0, title="Нулевой фрагмент")],
        videos=[make_video(score=1.0)],
    )
    llm = FakeLLM(payload={"found": True, "source_n": 0})

    result = answer("запрос", retriever=retriever, llm=llm)

    assert result.source.n == 0


# =======================================================================
# answer(): temperature forwarding
# =======================================================================
def test_answer_temperature_none_omits_temperature_kwarg():
    """temperature=None (the default) -> chat_json() gets no "temperature"
    keyword at all, not temperature=None."""
    retriever = FakeRetriever(citations=[make_citation()], videos=[make_video(score=1.0)])
    llm = FakeLLM(payload={"found": False})

    answer("запрос", retriever=retriever, llm=llm, temperature=None)

    _, kw = llm.calls[0]
    assert "temperature" not in kw


def test_answer_temperature_set_is_passed_through():
    """An explicit temperature reaches chat_json() as a keyword argument."""
    retriever = FakeRetriever(citations=[make_citation()], videos=[make_video(score=1.0)])
    llm = FakeLLM(payload={"found": False})

    answer("запрос", retriever=retriever, llm=llm, temperature=0.7)

    _, kw = llm.calls[0]
    assert kw["temperature"] == pytest.approx(0.7)


# =======================================================================
# answer(): retrieval parameters are forwarded to retriever.retrieve()
# =======================================================================
def test_answer_forwards_retrieval_params_to_retriever():
    """top_videos/per_video/use_intent_filter reach retriever.retrieve()
    unchanged."""
    retriever = FakeRetriever(citations=[], videos=[])
    llm = FakeLLM(fail_on_call=True)

    answer("рецепт борща", retriever=retriever, llm=llm,
           top_videos=5, per_video=3, use_intent_filter=True)

    assert retriever.calls == [
        {"query": "рецепт борща", "top_videos": 5, "per_video": 3, "use_intent_filter": True}
    ]
