"""Tests for src/index/store.py: point_id determinism, VectorStore client
construction, collection/payload-index bootstrapping, upsert batching, and the
three search modes (dense/sparse/hybrid).

The real QdrantClient is never touched: src.index.store.QdrantClient is
monkeypatched to a small stateful fake that tracks which collections exist,
what their payload_schema and points are, and what each write/read call
received — so assertions are made on that state, not on "was it called".
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import Any

import pytest
from qdrant_client import models

import src.index.store as store_mod
from src.index.embedder import Embedding
from src.index.store import VectorStore, point_id

# Ground truth for ensure_payload_indexes, kept independent of the module's
# own _PAYLOAD_INDEXES constant so a typo in that constant would still show up.
EXPECTED_PAYLOAD_INDEXES: dict[str, models.PayloadSchemaType] = {
    "video_id": models.PayloadSchemaType.KEYWORD,
    "chunk_index": models.PayloadSchemaType.INTEGER,
    "section": models.PayloadSchemaType.KEYWORD,
    "has_ingredients": models.PayloadSchemaType.BOOL,
    "has_steps": models.PayloadSchemaType.BOOL,
    "playlist_titles": models.PayloadSchemaType.KEYWORD,
    "title": models.PayloadSchemaType.TEXT,
}


# --- a stateful fake one level below FakeStore (fakes QdrantClient itself) --
class FakeQdrantClient:
    """A stand-in for qdrant_client.QdrantClient.

    Keeps real collection state (existence, payload_schema, points) instead of
    just logging calls, so tests assert on the resulting state. `*_error`/
    `*_error_fields` attributes let a test inject exceptions from specific
    calls to exercise the except branches in store.py.
    """

    def __init__(self, *, url: str | None = None, path: str | None = None,
                 timeout: float | None = None, check_compatibility: bool | None = None) -> None:
        self.init_kwargs = {
            "url": url, "path": path, "timeout": timeout,
            "check_compatibility": check_compatibility,
        }
        self.collections: dict[str, dict[str, Any]] = {}
        self.create_collection_calls: list[dict] = []
        self.upsert_calls: list[list[models.PointStruct]] = []
        self.scroll_calls: list[dict] = []
        self.query_points_calls: list[dict] = []
        self.get_collection_error: Exception | None = None
        self.payload_schema_is_none = False
        self.create_payload_index_error_fields: set[str] = set()
        self.create_payload_index_calls: list[str] = []
        self.scroll_response: list = []
        self.query_points_response: list = []

    # --- schema ---------------------------------------------------
    def collection_exists(self, name: str) -> bool:
        return name in self.collections

    def create_collection(self, name: str, *, vectors_config=None, sparse_vectors_config=None) -> None:
        self.create_collection_calls.append(
            {"name": name, "vectors_config": vectors_config, "sparse_vectors_config": sparse_vectors_config}
        )
        self.collections[name] = {
            "vectors_config": vectors_config,
            "sparse_vectors_config": sparse_vectors_config,
            "payload_schema": {},
            "points": {},
        }

    def delete_collection(self, name: str) -> None:
        self.collections.pop(name, None)

    def get_collection(self, name: str):
        if self.get_collection_error is not None:
            raise self.get_collection_error
        if self.payload_schema_is_none:
            # what a real Qdrant returns for a freshly created collection
            return SimpleNamespace(payload_schema=None)
        return SimpleNamespace(payload_schema=dict(self.collections[name]["payload_schema"]))

    def create_payload_index(self, name: str, *, field_name: str, field_schema) -> None:
        self.create_payload_index_calls.append(field_name)
        if field_name in self.create_payload_index_error_fields:
            raise RuntimeError(f"boom: {field_name}")
        self.collections[name]["payload_schema"][field_name] = field_schema

    # --- writing ----------------------------------------------
    def upsert(self, name: str, *, points, wait: bool = True) -> None:
        self.upsert_calls.append(list(points))
        for p in points:
            self.collections[name]["points"][p.id] = p

    def count(self, name: str, *, exact: bool = True):
        return SimpleNamespace(count=len(self.collections[name]["points"]))

    def scroll(self, name: str, *, scroll_filter=None, limit=None, with_payload=True, with_vectors=True):
        self.scroll_calls.append(
            {"name": name, "scroll_filter": scroll_filter, "limit": limit,
             "with_payload": with_payload, "with_vectors": with_vectors}
        )
        return self.scroll_response, None

    def query_points(self, name: str, *, query=None, using=None, limit=None,
                      query_filter=None, with_payload=True, prefetch=None):
        self.query_points_calls.append(
            {"name": name, "query": query, "using": using, "limit": limit,
             "query_filter": query_filter, "with_payload": with_payload, "prefetch": prefetch}
        )
        return SimpleNamespace(points=self.query_points_response)


@pytest.fixture
def patch_client(monkeypatch) -> None:
    """Redirects src.index.store.QdrantClient (imported at module load) to the fake."""
    monkeypatch.setattr(store_mod, "QdrantClient", FakeQdrantClient)


def make_store(*, url: str = "http://fake-qdrant:6333", collection: str = "test_col",
                path: str | None = None) -> VectorStore:
    return VectorStore(url=url, collection=collection, path=path)


def make_embedding(dense: list[float] | None = None, sparse_indices: list[int] | None = None,
                     sparse_values: list[float] | None = None) -> Embedding:
    return Embedding(
        dense=dense if dense is not None else [0.1, 0.2, 0.3],
        sparse_indices=sparse_indices if sparse_indices is not None else [1, 2],
        sparse_values=sparse_values if sparse_values is not None else [0.5, 0.75],
    )


# --- point_id -----------------------------------------------------
def test_point_id_is_stable_across_processes():
    """A chunk id always maps to the same point id, in this run and the next.

    The whole idempotency of the build rests on this: re-running
    `python -m src.index.build` without --recreate has to upsert over the same
    points. Comparing two calls in one process would not catch it — changing
    the uuid5 namespace keeps them equal to each other while duplicating the
    entire index on the next run. Hence a golden value.
    """
    assert point_id("vid1::001") == "6d692762-19b1-568d-939a-d0b2c8ff5853"


def test_point_id_differs_for_different_chunk_id():
    assert point_id("vid1::001") != point_id("vid1::002")


def test_point_id_returns_a_uuid_string_qdrant_accepts():
    """Qdrant only takes an unsigned integer or a UUID as a point id, so the
    value has to parse as a UUID."""
    result = point_id("vid1::001")
    assert isinstance(result, str)
    assert str(uuid.UUID(result)) == result


# --- VectorStore.__init__ ------------------------------------------
def test_init_with_path_uses_path_and_ignores_url(patch_client):
    vs = make_store(url="http://should-be-ignored:6333", collection="c", path="/tmp/qdrant-data")
    assert vs.client.init_kwargs == {
        "url": None, "path": "/tmp/qdrant-data", "timeout": None, "check_compatibility": None,
    }


def test_init_without_path_uses_url_timeout_and_check_compatibility_false(patch_client):
    vs = make_store(url="http://fake-qdrant:6333", collection="c", path=None)
    assert vs.client.init_kwargs == {
        "url": "http://fake-qdrant:6333", "path": None, "timeout": 60, "check_compatibility": False,
    }


def test_init_stores_collection_name(patch_client):
    vs = make_store(collection="my_collection")
    assert vs.collection == "my_collection"


# --- ensure_collection ----------------------------------------------
def test_ensure_collection_creates_when_missing(patch_client):
    vs = make_store()
    vs.ensure_collection(dense_size=256)

    col = vs.client.collections[vs.collection]
    dense_cfg = col["vectors_config"]["dense"]
    assert dense_cfg.size == 256
    assert dense_cfg.distance == models.Distance.COSINE
    assert "lexical" in col["sparse_vectors_config"]


def test_ensure_collection_missing_also_creates_payload_indexes(patch_client):
    vs = make_store()
    vs.ensure_collection(dense_size=256)

    schema = vs.client.collections[vs.collection]["payload_schema"]
    assert schema == EXPECTED_PAYLOAD_INDEXES


def test_ensure_collection_existing_no_recreate_keeps_points(patch_client):
    vs = make_store()
    vs.ensure_collection(dense_size=256)
    vs.upsert([{"chunk_id": "a"}], [make_embedding()])

    vs.ensure_collection(dense_size=256, recreate=False)

    assert len(vs.client.collections[vs.collection]["points"]) == 1
    assert len(vs.client.create_collection_calls) == 1  # not recreated


def test_ensure_collection_existing_no_recreate_still_creates_payload_indexes(patch_client):
    vs = make_store()
    vs.ensure_collection(dense_size=256)
    # simulate a collection that predates one of the indexes
    del vs.client.collections[vs.collection]["payload_schema"]["title"]

    vs.ensure_collection(dense_size=256, recreate=False)

    assert vs.client.collections[vs.collection]["payload_schema"] == EXPECTED_PAYLOAD_INDEXES


def test_ensure_collection_recreate_true_drops_existing_points(patch_client):
    vs = make_store()
    vs.ensure_collection(dense_size=256)
    vs.upsert([{"chunk_id": "a"}], [make_embedding()])
    assert len(vs.client.collections[vs.collection]["points"]) == 1

    vs.ensure_collection(dense_size=256, recreate=True)

    assert vs.client.collections[vs.collection]["points"] == {}
    assert len(vs.client.create_collection_calls) == 2  # deleted, then recreated


def test_ensure_collection_recreate_true_also_creates_payload_indexes(patch_client):
    vs = make_store()
    vs.ensure_collection(dense_size=256)

    vs.ensure_collection(dense_size=256, recreate=True)

    assert vs.client.collections[vs.collection]["payload_schema"] == EXPECTED_PAYLOAD_INDEXES


# --- ensure_payload_indexes -------------------------------------------
def test_ensure_payload_indexes_skips_fields_already_present(patch_client):
    vs = make_store()
    vs.client.collections[vs.collection] = {
        "vectors_config": None, "sparse_vectors_config": None,
        "payload_schema": {"video_id": models.PayloadSchemaType.KEYWORD}, "points": {},
    }

    vs.ensure_payload_indexes()

    # video_id was already there: create_payload_index must not be re-attempted for it.
    assert "video_id" not in vs.client.create_payload_index_calls
    expected_missing = set(EXPECTED_PAYLOAD_INDEXES) - {"video_id"}
    assert set(vs.client.create_payload_index_calls) == expected_missing


def test_ensure_payload_indexes_handles_a_null_payload_schema(patch_client):
    """A real Qdrant reports payload_schema=None on a fresh collection, and
    every index still gets created.

    The `or {}` in the source is belt-and-braces rather than load bearing: it
    sits inside a try/except that swallows everything and falls back to an
    empty set, so dropping it turns None.keys() into a caught AttributeError
    with the same outcome plus a log line. This test pins the outcome, which
    is what matters, not the guard.
    """
    vs = make_store()
    vs.ensure_collection(dense_size=3)
    vs.client.collections[vs.collection]["payload_schema"] = {}
    vs.client.payload_schema_is_none = True

    vs.ensure_payload_indexes()

    assert set(vs.client.collections[vs.collection]["payload_schema"]) == set(EXPECTED_PAYLOAD_INDEXES)


def test_ensure_payload_indexes_get_collection_error_falls_back_to_creating_all(patch_client):
    vs = make_store()
    vs.client.collections[vs.collection] = {
        "vectors_config": None, "sparse_vectors_config": None, "payload_schema": {}, "points": {},
    }
    vs.client.get_collection_error = RuntimeError("qdrant is down")

    vs.ensure_payload_indexes()  # must not raise

    assert vs.client.collections[vs.collection]["payload_schema"] == EXPECTED_PAYLOAD_INDEXES


def test_ensure_payload_indexes_one_field_failing_does_not_block_the_rest(patch_client):
    vs = make_store()
    vs.client.collections[vs.collection] = {
        "vectors_config": None, "sparse_vectors_config": None, "payload_schema": {}, "points": {},
    }
    vs.client.create_payload_index_error_fields = {"video_id"}

    vs.ensure_payload_indexes()  # must not raise

    schema = vs.client.collections[vs.collection]["payload_schema"]
    assert "video_id" not in schema
    expected_rest = {k: v for k, v in EXPECTED_PAYLOAD_INDEXES.items() if k != "video_id"}
    assert schema == expected_rest


# --- upsert -------------------------------------------------------
def test_upsert_point_id_and_payload(patch_client):
    vs = make_store()
    vs.ensure_collection(dense_size=3)
    chunk = {"chunk_id": "vid1::001", "video_id": "vid1", "text": "текст"}

    vs.upsert([chunk], [make_embedding(dense=[1.0, 2.0, 3.0])])

    stored = vs.client.collections[vs.collection]["points"]
    point = stored[point_id("vid1::001")]
    assert point.payload == chunk


def test_upsert_stores_dense_and_sparse_vectors_under_correct_names(patch_client):
    vs = make_store()
    vs.ensure_collection(dense_size=3)
    chunk = {"chunk_id": "vid1::001"}
    emb = make_embedding(dense=[1.0, 2.0, 3.0], sparse_indices=[7, 9], sparse_values=[0.4, 0.6])

    vs.upsert([chunk], [emb])

    point = vs.client.collections[vs.collection]["points"][point_id("vid1::001")]
    assert point.vector["dense"] == [1.0, 2.0, 3.0]
    assert point.vector["lexical"].indices == [7, 9]
    assert point.vector["lexical"].values == [0.4, 0.6]


def test_upsert_splits_into_batches(patch_client):
    vs = make_store()
    vs.ensure_collection(dense_size=3)
    chunks = [{"chunk_id": f"vid1::{i:03d}"} for i in range(5)]
    embs = [make_embedding() for _ in range(5)]

    vs.upsert(chunks, embs, batch=2)

    sizes = [len(batch) for batch in vs.client.upsert_calls]
    assert sizes == [2, 2, 1]


def test_upsert_returns_total_point_count(patch_client):
    vs = make_store()
    vs.ensure_collection(dense_size=3)
    chunks = [{"chunk_id": f"vid1::{i:03d}"} for i in range(5)]
    embs = [make_embedding() for _ in range(5)]

    result = vs.upsert(chunks, embs, batch=2)

    assert result == 5


def test_upsert_empty_list_returns_zero_without_calling_client(patch_client):
    vs = make_store()
    vs.ensure_collection(dense_size=3)

    result = vs.upsert([], [])

    assert result == 0
    assert vs.client.upsert_calls == []


def test_upsert_silently_stops_at_the_shorter_of_chunks_and_embeddings(patch_client):
    """Pins what zip() does when the two lists disagree: the extra chunks are
    dropped without a word.

    build.py pairs the batches by hand, so a mismatch is possible; when it
    happens the run still logs "Готово" with a count that merely looks small.
    Recorded as current behaviour, not endorsed — the returned count at least
    reflects what was really written.
    """
    vs = make_store()
    vs.ensure_collection(dense_size=3)
    chunks = [{"chunk_id": f"vid1::{i:03d}"} for i in range(3)]

    result = vs.upsert(chunks, [make_embedding()])

    assert result == 1
    assert list(vs.client.collections[vs.collection]["points"]) == [point_id("vid1::000")]


# --- count ----------------------------------------------------------
def test_count_returns_count_field_from_client(patch_client):
    vs = make_store()
    vs.ensure_collection(dense_size=3)
    vs.upsert([{"chunk_id": "a"}, {"chunk_id": "b"}], [make_embedding(), make_embedding()])

    assert vs.count() == 2


# --- fetch_chunks -----------------------------------------------------
def test_fetch_chunks_empty_indices_returns_empty_without_calling_client(patch_client):
    vs = make_store()
    vs.ensure_collection(dense_size=3)

    result = vs.fetch_chunks("vid1", [])

    assert result == []
    assert vs.client.scroll_calls == []


def test_fetch_chunks_builds_filter_on_video_id_and_sorted_chunk_index(patch_client):
    vs = make_store()
    vs.ensure_collection(dense_size=3)

    vs.fetch_chunks("vid1", [3, 1, 2])

    call = vs.client.scroll_calls[0]
    expected_filter = models.Filter(
        must=[
            models.FieldCondition(key="video_id", match=models.MatchValue(value="vid1")),
            models.FieldCondition(key="chunk_index", match=models.MatchAny(any=[1, 2, 3])),
        ]
    )
    assert call["scroll_filter"] == expected_filter


def test_fetch_chunks_limit_equals_number_of_requested_indices(patch_client):
    vs = make_store()
    vs.ensure_collection(dense_size=3)

    vs.fetch_chunks("vid1", [5, 6, 7, 8])

    assert vs.client.scroll_calls[0]["limit"] == 4


def test_fetch_chunks_asks_for_payloads_but_not_vectors(patch_client):
    """Neighbours are fetched for their TEXT, so the payload is required and
    the vectors are pure waste.

    with_payload=False would return points with payload=None, which
    _with_neighbors turns into empty chunks — an emptier context, and no error
    anywhere to explain why.
    """
    vs = make_store()
    vs.ensure_collection(dense_size=3)

    vs.fetch_chunks("vid1", [1])

    assert vs.client.scroll_calls[0]["with_payload"] is True
    assert vs.client.scroll_calls[0]["with_vectors"] is False


def test_fetch_chunks_none_payload_becomes_empty_dict(patch_client):
    vs = make_store()
    vs.ensure_collection(dense_size=3)
    vs.client.scroll_response = [SimpleNamespace(payload=None), SimpleNamespace(payload={"chunk_id": "a"})]

    result = vs.fetch_chunks("vid1", [1, 2])

    assert result == [{}, {"chunk_id": "a"}]


# --- search -----------------------------------------------------------
def test_search_mode_dense_uses_dense_vector_without_prefetch(patch_client):
    vs = make_store()
    vs.ensure_collection(dense_size=3)
    emb = make_embedding(dense=[1.0, 2.0, 3.0])

    vs.search(emb, k=5, mode="dense")

    call = vs.client.query_points_calls[0]
    assert call["using"] == "dense"
    assert call["query"] == [1.0, 2.0, 3.0]
    assert call["prefetch"] is None


def test_search_mode_sparse_uses_lexical_vector(patch_client):
    vs = make_store()
    vs.ensure_collection(dense_size=3)
    emb = make_embedding(sparse_indices=[7, 9], sparse_values=[0.4, 0.6])

    vs.search(emb, k=5, mode="sparse")

    call = vs.client.query_points_calls[0]
    assert call["using"] == "lexical"
    assert call["query"] == models.SparseVector(indices=[7, 9], values=[0.4, 0.6])
    assert call["prefetch"] is None


@pytest.mark.parametrize("mode", ["hybrid", "anything-else"], ids=["hybrid", "unknown-mode-falls-back-to-hybrid"])
def test_search_non_dense_non_sparse_mode_uses_rrf_fusion_with_two_prefetches(patch_client, mode):
    vs = make_store()
    vs.ensure_collection(dense_size=3)
    emb = make_embedding(dense=[1.0, 2.0, 3.0], sparse_indices=[7], sparse_values=[0.4])

    vs.search(emb, k=4, mode=mode, overfetch=5)

    call = vs.client.query_points_calls[0]
    assert isinstance(call["query"], models.FusionQuery)
    assert call["query"].fusion == models.Fusion.RRF
    prefetch = call["prefetch"]
    assert len(prefetch) == 2
    dense_pf = next(p for p in prefetch if p.using == "dense")
    lexical_pf = next(p for p in prefetch if p.using == "lexical")
    assert dense_pf.limit == 20  # k * overfetch = 4 * 5
    assert lexical_pf.limit == 20
    # Each branch must carry ITS OWN query. Swapping the two is accepted by
    # pydantic without a murmur, and both halves of the fusion would then
    # search the wrong vector — silently, with no error anywhere.
    assert dense_pf.query == [1.0, 2.0, 3.0]
    assert lexical_pf.query == models.SparseVector(indices=[7], values=[0.4])


def test_search_defaults_to_hybrid_with_fivefold_overfetch(patch_client):
    """Called with nothing but k, search runs the hybrid fusion and prefetches
    five times as deep.

    No caller anywhere passes overfetch — retriever.py, api/main.py and
    index/search.py all pass only k and mode — so the prefetch depth used in
    production IS this default.
    """
    vs = make_store()
    vs.ensure_collection(dense_size=3)

    vs.search(make_embedding(), k=4)

    call = vs.client.query_points_calls[0]
    assert isinstance(call["query"], models.FusionQuery)
    assert [p.limit for p in call["prefetch"]] == [20, 20]


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
def test_search_query_filter_is_forwarded_in_every_mode(patch_client, mode):
    vs = make_store()
    vs.ensure_collection(dense_size=3)
    emb = make_embedding()
    qfilter = models.Filter(
        must=[models.FieldCondition(key="video_id", match=models.MatchValue(value="vid1"))]
    )

    vs.search(emb, k=3, mode=mode, query_filter=qfilter)

    assert vs.client.query_points_calls[0]["query_filter"] == qfilter


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
def test_search_k_is_forwarded_as_limit(patch_client, mode):
    vs = make_store()
    vs.ensure_collection(dense_size=3)
    emb = make_embedding()

    vs.search(emb, k=7, mode=mode)

    assert vs.client.query_points_calls[0]["limit"] == 7


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
def test_search_always_asks_for_payloads(patch_client, mode):
    """Every mode requests the payload: it carries the chunk text, and without
    it retrieval builds an empty context and answers found=false on anything."""
    vs = make_store()
    vs.ensure_collection(dense_size=3)

    vs.search(make_embedding(), k=3, mode=mode)

    assert vs.client.query_points_calls[0]["with_payload"] is True


def test_search_returns_the_points_of_the_response_not_the_response_itself(patch_client):
    vs = make_store()
    vs.ensure_collection(dense_size=3)
    sentinel_points = ["p1", "p2"]
    vs.client.query_points_response = sentinel_points

    result = vs.search(make_embedding(), k=2, mode="dense")

    assert result is sentinel_points
