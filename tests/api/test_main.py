"""Tests for src/api/main.py: the FastAPI surface (/health, /search, /ask, /).

The lifespan constructs a real Retriever() + LLMClient(), which would load
bge-m3 and the reranker — several gigabytes and minutes we cannot afford in a
unit test. TestClient(app) called WITHOUT the `with` context manager never
runs the lifespan (confirmed against this project's fastapi/starlette
versions), so every test here fills STATE by hand with fakes instead.

STATE is a module-level dict, i.e. global state; an autouse fixture snapshots
it before each test and restores it after, mirroring the
_reset_retriever_singleton pattern in tests/conftest.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

import src.api.main as main_mod
from src.api.main import app
from src.config import LLM_MODEL, QDRANT_COLLECTION, RETRIEVAL_PER_VIDEO, RETRIEVAL_TOP_VIDEOS
from src.generation.schemas import RecipeAnswer, SourceRef
from tests.conftest import FakeEmbedder, FakeLLM, FakePoint, FakeStore, make_point

client = TestClient(app)


@dataclass
class FakeAppRetriever:
    """Only the two attributes /health and /search touch on a Retriever."""
    store: FakeStore
    embedder: FakeEmbedder


@pytest.fixture(autouse=True)
def _reset_state():
    """STATE is populated by hand in every test; keep it from leaking between
    tests (and from a stray real value some other module might have set)."""
    saved = dict(main_mod.STATE)
    main_mod.STATE.clear()
    yield
    main_mod.STATE.clear()
    main_mod.STATE.update(saved)


def set_state(*, store: FakeStore | None = None, embedder: FakeEmbedder | None = None,
              llm: FakeLLM | None = None) -> None:
    main_mod.STATE["retriever"] = FakeAppRetriever(
        store=store or FakeStore(), embedder=embedder or FakeEmbedder()
    )
    main_mod.STATE["llm"] = llm or FakeLLM()


# =======================================================================
# GET /health
# =======================================================================
def test_health_everything_up_reports_ok():
    """Qdrant reachable with points, LLM healthy -> overall status "ok"."""
    set_state(store=FakeStore(count_value=42), llm=FakeLLM(healthy=True))

    resp = client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["qdrant"] == {
        "ok": True, "collection": QDRANT_COLLECTION, "points": 42, "error": None,
    }
    assert body["llm"] == {"ok": True, "model": LLM_MODEL}


def test_health_llm_unhealthy_is_degraded_but_qdrant_stays_ok():
    """An unhealthy LLM alone flips the overall status, without touching the
    qdrant.ok flag."""
    set_state(store=FakeStore(count_value=10), llm=FakeLLM(healthy=False))

    resp = client.get("/health")

    body = resp.json()
    assert body["status"] == "degraded"
    assert body["qdrant"]["ok"] is True


def test_health_store_count_raises_reports_degraded_with_error_and_no_points():
    """store.count() raising is caught: qdrant.ok False, the error text is
    surfaced, and points stays None rather than some stale value."""
    class BrokenStore(FakeStore):
        def count(self) -> int:
            raise RuntimeError("qdrant is unreachable")

    set_state(store=BrokenStore(), llm=FakeLLM(healthy=True))

    resp = client.get("/health")

    body = resp.json()
    assert body["status"] == "degraded"
    assert body["qdrant"]["ok"] is False
    assert body["qdrant"]["points"] is None
    assert body["qdrant"]["error"] == "qdrant is unreachable"


def test_health_zero_points_is_degraded():
    """An empty collection counts as degraded: there is nothing to search, even
    though the count() call itself succeeded and the LLM is healthy."""
    set_state(store=FakeStore(count_value=0), llm=FakeLLM(healthy=True))

    resp = client.get("/health")

    body = resp.json()
    assert body["status"] == "degraded"
    assert body["qdrant"]["ok"] is True
    assert body["qdrant"]["points"] == 0


# =======================================================================
# GET /search
# =======================================================================
def test_search_hit_fields_come_from_the_point_payload():
    """Every SearchHit field is read out of the matching point's payload."""
    point = make_point(
        score=0.87, title="Стейк рибай", url="https://youtu.be/v1?t=60",
        timecode="01:00", section="steps", text="Обжарьте стейк.",
    )
    set_state(store=FakeStore(points=[point]))

    resp = client.get("/search", params={"q": "стейк"})

    assert resp.status_code == 200
    assert resp.json() == [{
        "title": "Стейк рибай", "url": "https://youtu.be/v1?t=60",
        "timecode": "01:00", "section": "steps",
        "score": pytest.approx(0.87), "text": "Обжарьте стейк.",
    }]


def test_search_missing_payload_keys_default_to_empty_strings():
    """A point whose payload lacks title/url/timecode/section/text does not
    raise: every missing field renders as an empty string."""
    point = FakePoint(id="x", score=0.5, payload={})
    set_state(store=FakeStore(points=[point]))

    resp = client.get("/search", params={"q": "стейк"})

    hit = resp.json()[0]
    assert hit["title"] == ""
    assert hit["url"] == ""
    assert hit["timecode"] == ""
    assert hit["section"] == ""
    assert hit["text"] == ""
    assert hit["score"] == pytest.approx(0.5)


def test_search_forwards_k_and_mode_to_store_search():
    """The k and mode query params reach store.search() unchanged."""
    store = FakeStore(points=[make_point()])
    set_state(store=store)

    client.get("/search", params={"q": "стейк", "k": 3, "mode": "sparse"})

    assert store.search_calls[0]["k"] == 3
    assert store.search_calls[0]["mode"] == "sparse"


def test_search_empty_result_returns_empty_list():
    set_state(store=FakeStore(points=[]))

    resp = client.get("/search", params={"q": "стейк"})

    assert resp.json() == []


def test_search_query_is_encoded_through_the_retriever_embedder():
    """The raw query string reaches embedder.encode_queries()."""
    embedder = FakeEmbedder()
    set_state(store=FakeStore(points=[]), embedder=embedder)

    client.get("/search", params={"q": "борщ рецепт"})

    assert embedder.encoded == [["борщ рецепт"]]


@pytest.mark.parametrize(
    "params",
    [
        pytest.param({"q": "а"}, id="q-shorter-than-2-chars"),
        pytest.param({"q": "стейк", "k": 0}, id="k-below-1"),
        pytest.param({"q": "стейк", "k": 21}, id="k-above-20"),
        pytest.param({"q": "стейк", "mode": "fulltext"}, id="mode-not-in-allowed-set"),
    ],
)
def test_search_validation_rejects_out_of_range_params(params):
    set_state(store=FakeStore(points=[]))

    resp = client.get("/search", params=params)

    assert resp.status_code == 422


@pytest.mark.parametrize("k", [1, 20], ids=["k-at-lower-bound", "k-at-upper-bound"])
def test_search_validation_accepts_boundary_k(k):
    """k=1 and k=20 are the inclusive edges of the allowed range."""
    set_state(store=FakeStore(points=[]))

    resp = client.get("/search", params={"q": "стейк", "k": k})

    assert resp.status_code == 200


# =======================================================================
# POST /ask
# =======================================================================
def make_recipe_answer(**overrides) -> RecipeAnswer:
    defaults = dict(
        query="как приготовить стейк",
        found=True,
        dish="Стейк рибай",
        ingredients=["стейк 300г"],
        steps=["обжарить"],
        notes=None,
        source=SourceRef(n=1, video_id="v1", title="Стейк", url="https://youtu.be/v1", timecode="01:00"),
        sources=[],
        model="fake-model",
        used_reranker=True,
    )
    defaults.update(overrides)
    return RecipeAnswer(**defaults)


def test_ask_success_body_matches_answer_result(monkeypatch):
    """The /ask response body is exactly what answer() returned, serialized."""
    result = make_recipe_answer()
    monkeypatch.setattr(main_mod, "answer", lambda *a, **kw: result)
    set_state()

    resp = client.post("/ask", json={"query": "как приготовить стейк"})

    assert resp.status_code == 200
    assert resp.json() == result.model_dump()


def test_ask_forwards_query_retriever_llm_and_body_params(monkeypatch):
    """answer() gets the query positionally plus retriever/llm from STATE and
    top_videos/per_video/use_intent_filter/temperature from the request body."""
    calls = []

    def fake_answer(query, **kw):
        calls.append((query, kw))
        return make_recipe_answer(query=query)

    monkeypatch.setattr(main_mod, "answer", fake_answer)
    retriever = FakeAppRetriever(store=FakeStore(), embedder=FakeEmbedder())
    llm = FakeLLM()
    main_mod.STATE["retriever"] = retriever
    main_mod.STATE["llm"] = llm

    client.post("/ask", json={
        "query": "борщ", "top_videos": 4, "per_video": 3,
        "use_intent_filter": True, "temperature": 0.6,
    })

    query, kw = calls[0]
    assert query == "борщ"
    assert kw["retriever"] is retriever
    assert kw["llm"] is llm
    assert kw["top_videos"] == 4
    assert kw["per_video"] == 3
    assert kw["use_intent_filter"] is True
    assert kw["temperature"] == pytest.approx(0.6)


def test_ask_answer_raising_returns_502_not_500(monkeypatch):
    """An exception out of answer() is turned into a 502 with the error text in
    detail, not an unhandled 500."""
    def fake_answer(*a, **kw):
        raise ValueError("LLM did not return valid JSON")

    monkeypatch.setattr(main_mod, "answer", fake_answer)
    set_state()

    resp = client.post("/ask", json={"query": "как приготовить стейк"})

    assert resp.status_code == 502
    assert "LLM did not return valid JSON" in resp.json()["detail"]


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({"query": "а"}, id="query-shorter-than-2-chars"),
        pytest.param({"query": "а" * 501}, id="query-longer-than-500-chars"),
        pytest.param({"query": "стейк", "top_videos": 0}, id="top_videos-below-1"),
        pytest.param({"query": "стейк", "top_videos": 9}, id="top_videos-above-8"),
        pytest.param({"query": "стейк", "per_video": 0}, id="per_video-below-1"),
        pytest.param({"query": "стейк", "per_video": 13}, id="per_video-above-12"),
        pytest.param({"query": "стейк", "temperature": -0.1}, id="temperature-below-0"),
        pytest.param({"query": "стейк", "temperature": 1.6}, id="temperature-above-1.5"),
    ],
)
def test_ask_validation_rejects_out_of_range_body(monkeypatch, body):
    monkeypatch.setattr(main_mod, "answer", lambda *a, **kw: make_recipe_answer())
    set_state()

    resp = client.post("/ask", json=body)

    assert resp.status_code == 422


def test_ask_defaults_come_from_config_not_hardcoded_values(monkeypatch):
    """Omitting top_videos/per_video in the body must forward the CONFIGURED
    RETRIEVAL_TOP_VIDEOS/RETRIEVAL_PER_VIDEO, not hardcoded numbers — a
    previous version hardcoded 3/2 here and silently overrode server settings
    on every call."""
    captured = {}

    def fake_answer(query, **kw):
        captured.update(kw)
        return make_recipe_answer()

    # The expectation and the source read the same constants, so the test can
    # only tell them apart while the configured values differ from the old
    # hardcode. Say so out loud instead of passing on a coincidence.
    if (RETRIEVAL_TOP_VIDEOS, RETRIEVAL_PER_VIDEO) == (3, 2):
        pytest.skip(
            "RETRIEVAL_TOP_VIDEOS/PER_VIDEO happen to equal the historical "
            "hardcode (3/2); this test cannot distinguish the two here"
        )

    monkeypatch.setattr(main_mod, "answer", fake_answer)
    set_state()

    client.post("/ask", json={"query": "как приготовить стейк"})

    assert captured["top_videos"] == RETRIEVAL_TOP_VIDEOS
    assert captured["per_video"] == RETRIEVAL_PER_VIDEO


def test_ask_temperature_defaults_to_none(monkeypatch):
    captured = {}

    def fake_answer(query, **kw):
        captured.update(kw)
        return make_recipe_answer()

    monkeypatch.setattr(main_mod, "answer", fake_answer)
    set_state()

    client.post("/ask", json={"query": "как приготовить стейк"})

    assert captured["temperature"] is None


# =======================================================================
# GET /
# =======================================================================
def test_root_reports_service_docs_and_health():
    resp = client.get("/")

    assert resp.json() == {"service": "oblomoff RAG", "docs": "/docs", "health": "/health"}


# =======================================================================
# lifespan
# =======================================================================
# The lifespan is what wires production: nothing else ever puts anything into
# STATE. Running it is safe once the two heavy constructors are replaced —
# without that substitution `with TestClient(app)` would load bge-m3 and the
# reranker for real.

def test_lifespan_populates_state_on_startup_and_clears_it_on_shutdown(monkeypatch):
    """Startup builds the retriever and the LLM client once and puts both in
    STATE; shutdown empties it.

    Nothing else populates STATE, so a lifespan that skipped either one would
    leave every endpoint raising KeyError on the first request — in
    production only, since the tests seed STATE by hand.
    """
    built: list[str] = []
    the_retriever = FakeAppRetriever(store=FakeStore(count_value=7), embedder=FakeEmbedder())
    the_llm = FakeLLM(healthy=True)

    def fake_retriever():
        built.append("retriever")
        return the_retriever

    def fake_llm_client():
        built.append("llm")
        return the_llm

    monkeypatch.setattr(main_mod, "Retriever", fake_retriever)
    monkeypatch.setattr(main_mod, "LLMClient", fake_llm_client)

    with TestClient(app) as started_client:
        # Identity, not just the set of keys: swapping the two objects keeps
        # both keys present and would only surface as an AttributeError on the
        # first real request.
        assert main_mod.STATE["retriever"] is the_retriever
        assert main_mod.STATE["llm"] is the_llm
        assert sorted(built) == ["llm", "retriever"]
        # /health is the endpoint that actually reads STATE, so this proves the
        # wiring rather than merely that the app starts.
        assert started_client.get("/health").json()["status"] == "ok"

    assert main_mod.STATE == {}


# =======================================================================
# validation: the accepted side of every boundary
# =======================================================================
# The rejection tests above sit outside each range; these sit exactly on it.
# Without them a narrowed bound (say top_videos le=8 -> le=5) would pass
# unnoticed, since nothing would be testing that 8 is still allowed.

@pytest.mark.parametrize(
    "body",
    [
        pytest.param({"query": "ст"}, id="query_at_min_length"),
        pytest.param({"query": "с" * 500}, id="query_at_max_length"),
        pytest.param({"query": "стейк", "top_videos": 1}, id="top_videos_at_min"),
        pytest.param({"query": "стейк", "top_videos": 8}, id="top_videos_at_max"),
        pytest.param({"query": "стейк", "per_video": 1}, id="per_video_at_min"),
        pytest.param({"query": "стейк", "per_video": 12}, id="per_video_at_max"),
        pytest.param({"query": "стейк", "temperature": 0.0}, id="temperature_at_min"),
        pytest.param({"query": "стейк", "temperature": 1.5}, id="temperature_at_max"),
    ],
)
def test_ask_accepts_values_exactly_on_the_boundary(monkeypatch, body):
    """Every AskRequest bound is inclusive."""
    monkeypatch.setattr(main_mod, "answer", lambda *a, **kw: make_recipe_answer())
    set_state()

    assert client.post("/ask", json=body).status_code == 200


@pytest.mark.parametrize(
    "params",
    [
        pytest.param({"q": "ст"}, id="q_at_min_length"),
        pytest.param({"q": "с" * 500}, id="q_at_max_length"),
        pytest.param({"q": "стейк", "k": 1}, id="k_at_min"),
        pytest.param({"q": "стейк", "k": 20}, id="k_at_max"),
        pytest.param({"q": "стейк", "mode": "hybrid"}, id="mode_hybrid"),
        pytest.param({"q": "стейк", "mode": "dense"}, id="mode_dense"),
        pytest.param({"q": "стейк", "mode": "sparse"}, id="mode_sparse"),
    ],
)
def test_search_accepts_values_exactly_on_the_boundary(params):
    """Every /search bound is inclusive, and all three modes are allowed."""
    set_state()

    assert client.get("/search", params=params).status_code == 200


def test_ask_temperature_zero_is_forwarded_rather_than_dropped(monkeypatch):
    """temperature=0.0 is a deliberate setting, not "unset".

    Anything falsy-based here (`req.temperature or None`) would silently swap
    a request for greedy decoding back to the model's default — the same trap
    that already bit source_n=0 in answer().
    """
    captured: dict = {}

    def fake_answer(query, **kw):
        captured.update(kw)
        return make_recipe_answer()

    monkeypatch.setattr(main_mod, "answer", fake_answer)
    set_state()

    client.post("/ask", json={"query": "стейк", "temperature": 0.0})

    assert captured["temperature"] == pytest.approx(0.0)


def test_ask_found_false_is_served_as_a_normal_200(monkeypatch):
    """A "no recipe in the transcripts" answer is a successful response, not
    an error, and a null source must survive the response_model."""
    result = make_recipe_answer(found=False, dish="", ingredients=[], steps=[], source=None)
    monkeypatch.setattr(main_mod, "answer", lambda *a, **kw: result)
    set_state()

    resp = client.post("/ask", json={"query": "борщ"})

    assert resp.status_code == 200
    assert resp.json()["found"] is False
    assert resp.json()["source"] is None


def test_ask_answer_returning_a_non_recipe_object_is_not_downgraded_to_502(monkeypatch):
    """The 502 is for a generation that raised, not for one that returned
    something unusable: a malformed return value fails later, as a 500.

    Pinned so the boundary of the error handling stays visible — widening the
    try/except to cover this would hide a programming error behind a gateway
    status.
    """
    monkeypatch.setattr(main_mod, "answer", lambda *a, **kw: "не RecipeAnswer")
    set_state()

    with pytest.raises(AttributeError):
        client.post("/ask", json={"query": "стейк"})


# =======================================================================
# /search: defaults, missing payload, CORS
# =======================================================================
def test_search_defaults_to_five_hits_of_hybrid_search():
    """Called with only q, /search asks the store for 5 hybrid hits — the
    defaults are what the UI and curl actually get."""
    store = FakeStore()
    set_state(store=store)

    client.get("/search", params={"q": "стейк"})

    assert store.search_calls[0]["k"] == 5
    assert store.search_calls[0]["mode"] == "hybrid"


def test_search_point_without_payload_yields_empty_fields():
    """Qdrant can return a point with no payload at all; the endpoint fills
    the hit with empty strings instead of raising."""
    store = FakeStore(points=[FakePoint(id="pt-1", score=0.5, payload=None)])
    set_state(store=store)

    [hit] = client.get("/search", params={"q": "стейк"}).json()

    assert hit == {
        "title": "", "url": "", "timecode": "", "section": "", "score": 0.5, "text": "",
    }


def test_cors_allows_any_origin():
    """The Streamlit UI calls this API from a browser on another port, so the
    permissive CORS middleware is load bearing, not decoration."""
    set_state()

    resp = client.get("/health", headers={"Origin": "http://localhost:8501"})

    assert resp.headers["access-control-allow-origin"] == "*"
