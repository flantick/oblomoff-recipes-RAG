"""Tests for src/index/search.py: make_filter (building the Qdrant filter)
and search (embedding the query and delegating to the store). The CLI
wrapper _main() is out of scope (by project convention)."""
from __future__ import annotations

from qdrant_client import models

from src.index.embedder import Embedding
from src.index.search import make_filter, search


# --- make_filter: no arguments ------------------------------------------
def test_make_filter_returns_none_when_no_arguments_given():
    """An empty Filter in Qdrant means 'match nothing', which is a different
    thing from 'no restriction' — make_filter must return None, not
    models.Filter(must=[])."""
    assert make_filter() is None


# --- make_filter: single condition ---------------------------------------
def test_make_filter_builds_single_condition_for_video_id_only():
    flt = make_filter(video_id="vid1")
    assert flt == models.Filter(
        must=[models.FieldCondition(key="video_id", match=models.MatchValue(value="vid1"))]
    )


def test_make_filter_builds_single_condition_for_section_only():
    flt = make_filter(section="steps")
    assert flt == models.Filter(
        must=[models.FieldCondition(key="section", match=models.MatchValue(value="steps"))]
    )


def test_make_filter_builds_single_condition_for_has_ingredients_true():
    flt = make_filter(has_ingredients=True)
    assert flt == models.Filter(
        must=[models.FieldCondition(key="has_ingredients", match=models.MatchValue(value=True))]
    )


def test_make_filter_builds_condition_for_has_ingredients_false():
    """has_ingredients=False must still produce a condition: the check in the
    source is `if has_ingredients is not None`, not a truthiness check. A
    naive `if has_ingredients:` would silently drop the ability to search for
    chunks WITHOUT ingredients (False is falsy)."""
    flt = make_filter(has_ingredients=False)
    assert flt == models.Filter(
        must=[models.FieldCondition(key="has_ingredients", match=models.MatchValue(value=False))]
    )


# --- make_filter: combined ------------------------------------------------
def test_make_filter_combines_all_three_in_declaration_order():
    flt = make_filter(video_id="vid1", section="steps", has_ingredients=True)
    assert flt == models.Filter(
        must=[
            models.FieldCondition(key="video_id", match=models.MatchValue(value="vid1")),
            models.FieldCondition(key="section", match=models.MatchValue(value="steps")),
            models.FieldCondition(key="has_ingredients", match=models.MatchValue(value=True)),
        ]
    )


# --- search: embedding and delegation -------------------------------------
def test_search_encodes_query_and_forwards_first_embedding(fake_store, fake_embedder):
    """The query text is encoded via encode_queries, and the store receives
    the first (and only) resulting embedding, not the whole list."""
    search(fake_store, fake_embedder, "стейк")

    assert fake_embedder.encoded == [["стейк"]]
    [call] = fake_store.search_calls
    # Spelled out rather than recomputed with the fake: calling the fake again
    # would compare it against itself, and would also append a second entry to
    # `encoded`, making the order of these two asserts significant.
    assert call["emb"] == Embedding(
        dense=[0.1, 0.2, 0.3], sparse_indices=[7, 42], sparse_values=[0.5, 0.25]
    )


def test_search_uses_default_k_mode_and_filter(fake_store, fake_embedder):
    """Defaults: k=5, mode='hybrid', query_filter=None."""
    search(fake_store, fake_embedder, "стейк")

    [call] = fake_store.search_calls
    assert call["k"] == 5
    assert call["mode"] == "hybrid"
    assert call["query_filter"] is None


def test_search_forwards_k_mode_and_filter_unchanged(fake_store, fake_embedder):
    """Explicit k, mode and query_filter reach store.search unmodified."""
    flt = make_filter(video_id="vid1")
    search(fake_store, fake_embedder, "стейк", k=3, mode="dense", query_filter=flt)

    [call] = fake_store.search_calls
    assert call["k"] == 3
    assert call["mode"] == "dense"
    assert call["query_filter"] is flt


def test_search_returns_store_result_unchanged(fake_store, fake_embedder):
    """search() returns exactly what store.search returned, in the same order."""
    from tests.conftest import make_point

    p1, p2 = make_point(score=0.9), make_point(score=0.5)
    fake_store.points = [p1, p2]

    result = search(fake_store, fake_embedder, "стейк", k=2)

    assert result == [p1, p2]
