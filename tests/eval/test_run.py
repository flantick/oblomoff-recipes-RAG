"""Tests for src/eval/run.py: the golden-set quality metrics.

Both eval_retrieval() and eval_answer() call retriever.retrieve()/answer()
through their own injection seams; nothing here touches Qdrant, an embedder or
an LLM. FakeRetriever from tests/conftest.py always returns the same
RetrievalResult regardless of the call, but eval_retrieval() calls retrieve()
twice per item with different kwargs (a wide ranking pass, then a production
pass) and needs the two calls to differ — so this file defines its own
two-pass fake, built the same way FakeRetriever is.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from src.eval.dataset import GoldenItem
from src.eval.run import _chunk_index, _mean, eval_answer, eval_retrieval
from src.generation.schemas import RecipeAnswer, SourceRef
from src.retrieval.schemas import RetrievalResult, RetrievedPassage, RetrievedVideo


# =======================================================================
# fakes and factories local to this file
# =======================================================================
@dataclass
class TwoPassRetriever:
    """Stands in for src.retrieval.retriever.Retriever inside eval_retrieval().

    Unlike FakeRetriever (one canned result for every call), eval_retrieval()
    calls retrieve() twice per item with different kwargs and needs the two
    passes to differ:
        wide — the ranking pass: retrieve(query, top_videos=..., rerank=True)
        prod — the production pass: retrieve(query) or retrieve(query, token_budget=...)
    Both maps are keyed by query text so several golden items can be run at once.
    """
    wide: dict[str, RetrievalResult] = field(default_factory=dict)
    prod: dict[str, RetrievalResult] = field(default_factory=dict)
    calls: list[dict] = field(default_factory=list)

    def retrieve(self, query: str, **kw):
        self.calls.append({"query": query, **kw})
        table = self.wide if kw.get("rerank") else self.prod
        return table[query]


def make_item(*, id: str = "q1", query: str = "запрос", relevant=None,
              kind: str = "exact", expect_found: bool = True) -> GoldenItem:
    return GoldenItem(
        id=id, query=query,
        relevant=["v1"] if relevant is None else relevant,
        kind=kind, expect_found=expect_found,
    )


def make_video(video_id: str = "v1", score: float = 1.0, passages=None) -> RetrievedVideo:
    return RetrievedVideo(video_id=video_id, title=f"Видео {video_id}", score=score,
                           passages=list(passages) if passages else [])


def make_passage(video_id: str = "v1", chunk_ids=("v1::001",)) -> RetrievedPassage:
    return RetrievedPassage(
        video_id=video_id, title="Видео", url="https://youtu.be/x?t=10",
        timecode="00:10", start=10.0, end=20.0, text="текст фрагмента",
        chunk_ids=list(chunk_ids), score=1.0,
    )


def make_result(query: str, videos) -> RetrievalResult:
    return RetrievalResult(query=query, videos=list(videos), context="",
                            citations=[], used_reranker=True, mode="hybrid")


def make_recipe_answer(*, query: str = "запрос", found: bool = True, dish: str = "",
                        ingredients=None, steps=None, source: SourceRef | None = None) -> RecipeAnswer:
    return RecipeAnswer(
        query=query, found=found, dish=dish,
        ingredients=list(ingredients) if ingredients else [],
        steps=list(steps) if steps else [], source=source,
    )


def make_answer_fn(table: dict[str, object]):
    """table maps query -> RecipeAnswer to return, or an Exception to raise."""
    def _fn(query, **kw):
        outcome = table[query]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
    return _fn


# =======================================================================
# _chunk_index
# =======================================================================
def test_chunk_index_parses_numeric_suffix():
    """A well-formed '<video_id>::NNN' chunk_id yields its integer index."""
    assert _chunk_index("vid1::007") == 7


def test_chunk_index_parses_zero_suffix():
    """Chunk index 0 (the clip's edge chunk) parses to plain 0, not falsy -1."""
    assert _chunk_index("vid1::000") == 0


@pytest.mark.parametrize(
    "chunk_id",
    ["no-separator-at-all", "", "vid1::abc"],
    ids=["missing-separator", "empty-string", "non-numeric-suffix"],
)
def test_chunk_index_returns_minus_one_for_invalid_input(chunk_id):
    """Anything that is not '<id>::<int>' falls back to the sentinel -1."""
    assert _chunk_index(chunk_id) == -1


# =======================================================================
# _mean
# =======================================================================
def test_mean_of_empty_list_is_zero():
    """An empty metric list must not raise statistics.StatisticsError."""
    assert _mean([]) == 0.0


def test_mean_rounds_to_four_decimal_places():
    """A repeating decimal is rounded to exactly 4 digits."""
    assert _mean([1 / 3, 1 / 3, 1 / 3]) == pytest.approx(0.3333)


def test_mean_of_ordinary_values():
    assert _mean([1.0, 3.0]) == pytest.approx(2.0)


# =======================================================================
# eval_retrieval: hit@k / MRR from the wide pass
# =======================================================================
def test_eval_retrieval_relevant_video_at_rank_one_hits_every_k():
    """The relevant video first in the wide order -> hit@1/3/5=1, MRR=1."""
    item = make_item(query="Q1", relevant=["v1"])
    wide = make_result("Q1", [make_video("v1", score=0.9)])
    prod = make_result("Q1", [make_video("v1", score=0.9)])
    retriever = TwoPassRetriever(wide={"Q1": wide}, prod={"Q1": prod})

    summary = eval_retrieval([item], retriever, rank_k=5, budget=None)["summary"]

    assert summary["hit@1"] == pytest.approx(1.0)
    assert summary["hit@3"] == pytest.approx(1.0)
    assert summary["hit@5"] == pytest.approx(1.0)
    assert summary["mrr"] == pytest.approx(1.0)


def test_eval_retrieval_relevant_video_at_rank_three_is_boundary_hit_for_hit_at_3():
    """rank=3 counts for hit@3/hit@5 but not hit@1; MRR=1/3."""
    item = make_item(query="Q1", relevant=["v1"])
    wide = make_result("Q1", [make_video("other1"), make_video("other2"), make_video("v1")])
    prod = make_result("Q1", [make_video("v1")])
    retriever = TwoPassRetriever(wide={"Q1": wide}, prod={"Q1": prod})

    summary = eval_retrieval([item], retriever, rank_k=5, budget=None)["summary"]

    assert summary["hit@1"] == pytest.approx(0.0)
    assert summary["hit@3"] == pytest.approx(1.0)
    assert summary["hit@5"] == pytest.approx(1.0)
    assert summary["mrr"] == pytest.approx(1 / 3, abs=1e-4)


def test_eval_retrieval_miss_gives_zero_hits_and_zero_mrr():
    """The relevant video absent from the wide list -> all hit@k=0, MRR=0."""
    item = make_item(query="Q1", relevant=["v1"])
    wide = make_result("Q1", [make_video("other")])
    prod = make_result("Q1", [make_video("other")])
    retriever = TwoPassRetriever(wide={"Q1": wide}, prod={"Q1": prod})

    summary = eval_retrieval([item], retriever, rank_k=5, budget=None)["summary"]

    assert summary["hit@1"] == pytest.approx(0.0)
    assert summary["hit@3"] == pytest.approx(0.0)
    assert summary["hit@5"] == pytest.approx(0.0)
    assert summary["mrr"] == pytest.approx(0.0)


# =======================================================================
# eval_retrieval: negative queries
# =======================================================================
def test_eval_retrieval_negative_query_excluded_from_n_and_hit_rate():
    """A negative item (empty relevant) does not count toward n and does not
    dilute hit@1: if it counted as a miss, hit@1 would drop to 0.5."""
    pos = make_item(id="pos1", query="Q1", relevant=["v1"])
    neg = make_item(id="neg1", query="Q2", relevant=[], kind="negative", expect_found=False)
    retriever = TwoPassRetriever(
        wide={"Q1": make_result("Q1", [make_video("v1")]),
              "Q2": make_result("Q2", [make_video("other", score=0.7)])},
        prod={"Q1": make_result("Q1", [make_video("v1")]),
              "Q2": make_result("Q2", [make_video("other2", score=0.6)])},
    )

    summary = eval_retrieval([pos, neg], retriever, rank_k=5, budget=None)["summary"]

    assert summary["n"] == 1
    assert summary["hit@1"] == pytest.approx(1.0)


def test_eval_retrieval_negative_prod_score_feeds_neg_score_max():
    """A negative item's production top score is collected separately and
    surfaces as neg_score_max once a positive is present too."""
    pos = make_item(id="pos1", query="Q1", relevant=["v1"])
    neg = make_item(id="neg1", query="Q2", relevant=[], kind="negative", expect_found=False)
    retriever = TwoPassRetriever(
        wide={"Q1": make_result("Q1", [make_video("v1", score=0.8)]),
              "Q2": make_result("Q2", [make_video("other", score=0.7)])},
        prod={"Q1": make_result("Q1", [make_video("v1", score=0.8)]),
              "Q2": make_result("Q2", [make_video("other2", score=0.3)])},
    )

    summary = eval_retrieval([pos, neg], retriever, rank_k=5, budget=None)["summary"]

    assert summary["neg_score_max"] == pytest.approx(0.3)
    assert summary["neg_score_mean"] == pytest.approx(0.3)


# =======================================================================
# eval_retrieval: ctx_precision / edge_rate
# =======================================================================
def test_eval_retrieval_ctx_precision_is_share_of_relevant_passages():
    """One of two production passages comes from the relevant video -> 0.5."""
    item = make_item(query="Q1", relevant=["v1"])
    prod_video = make_video("v1", passages=[
        make_passage("v1", chunk_ids=["v1::003"]),
        make_passage("v2", chunk_ids=["v2::003"]),
    ])
    retriever = TwoPassRetriever(
        wide={"Q1": make_result("Q1", [make_video("v1")])},
        prod={"Q1": make_result("Q1", [prod_video])},
    )

    result = eval_retrieval([item], retriever, rank_k=5, budget=None)

    assert result["summary"]["ctx_precision"] == pytest.approx(0.5)
    assert result["rows"][0]["ctx_precision"] == pytest.approx(0.5)


def test_eval_retrieval_edge_rate_counts_passages_starting_at_chunk_zero():
    """One of two passages starts at chunk index 0 (the clip's edge) -> 0.5."""
    item = make_item(query="Q1", relevant=["v1"])
    prod_video = make_video("v1", passages=[
        make_passage("v1", chunk_ids=["v1::000", "v1::001"]),
        make_passage("v1", chunk_ids=["v1::005"]),
    ])
    retriever = TwoPassRetriever(
        wide={"Q1": make_result("Q1", [make_video("v1")])},
        prod={"Q1": make_result("Q1", [prod_video])},
    )

    summary = eval_retrieval([item], retriever, rank_k=5, budget=None)["summary"]

    assert summary["edge_rate"] == pytest.approx(0.5)


def test_eval_retrieval_video_without_passages_does_not_skew_summary_ctx_precision():
    """An item whose production video has zero passages contributes nothing
    to ctx_precision: the summary equals the OTHER item's ratio (1.0), not the
    average of 1.0 and an implicit 0.0 for the empty one."""
    item1 = make_item(id="q1", query="Q1", relevant=["v1"])
    item2 = make_item(id="q2", query="Q2", relevant=["v2"])
    prod1 = make_video("v1", passages=[make_passage("v1", chunk_ids=["v1::003"])])
    prod2 = make_video("v2", passages=[])  # no passages at all
    retriever = TwoPassRetriever(
        wide={"Q1": make_result("Q1", [make_video("v1")]),
              "Q2": make_result("Q2", [make_video("v2")])},
        prod={"Q1": make_result("Q1", [prod1]),
              "Q2": make_result("Q2", [prod2])},
    )

    summary = eval_retrieval([item1, item2], retriever, rank_k=5, budget=None)["summary"]

    assert summary["ctx_precision"] == pytest.approx(1.0)



@pytest.mark.xfail(
    reason="src/eval/run.py:108 sets rows[i]['ctx_precision'] from ctx_prec[-1] "
    "whenever ctx_prec is non-empty, instead of checking whether THIS item's "
    "own `passages` list was non-empty. An item whose video has no passages "
    "silently inherits the previous item's ctx_precision in the per-row report "
    "instead of getting None, which is misleading when reading the JSON report "
    "row by row (the aggregate summary metric is unaffected).",
    strict=True,
)
def test_eval_retrieval_row_ctx_precision_is_none_when_item_has_no_passages():
    item1 = make_item(id="q1", query="Q1", relevant=["v1"])
    item2 = make_item(id="q2", query="Q2", relevant=["v2"])
    prod1 = make_video("v1", passages=[make_passage("v1", chunk_ids=["v1::003"])])
    prod2 = make_video("v2", passages=[])
    retriever = TwoPassRetriever(
        wide={"Q1": make_result("Q1", [make_video("v1")]),
              "Q2": make_result("Q2", [make_video("v2")])},
        prod={"Q1": make_result("Q1", [prod1]),
              "Q2": make_result("Q2", [prod2])},
    )

    rows = eval_retrieval([item1, item2], retriever, rank_k=5, budget=None)["rows"]

    assert rows[1]["ctx_precision"] is None


# =======================================================================
# eval_retrieval: by_kind
# =======================================================================
def test_eval_retrieval_by_kind_splits_hit_rates_per_kind():
    """by_kind only covers positives and buckets hit@1/hit@3 per kind."""
    exact = make_item(id="q1", query="Q1", relevant=["v1"], kind="exact")
    para = make_item(id="q2", query="Q2", relevant=["v2"], kind="paraphrase")
    retriever = TwoPassRetriever(
        wide={"Q1": make_result("Q1", [make_video("v1")]),  # rank 1
              "Q2": make_result("Q2", [make_video("x1"), make_video("x2"),
                                        make_video("x3"), make_video("v2")])},  # rank 4
        prod={"Q1": make_result("Q1", [make_video("v1")]),
              "Q2": make_result("Q2", [make_video("v2")])},
    )

    by_kind = eval_retrieval([exact, para], retriever, rank_k=5, budget=None)["by_kind"]

    assert by_kind["exact"] == {"n": 1, "hit@1": pytest.approx(1.0), "hit@3": pytest.approx(1.0)}
    assert by_kind["paraphrase"] == {"n": 1, "hit@1": pytest.approx(0.0), "hit@3": pytest.approx(0.0)}


# =======================================================================
# eval_retrieval: pos_score/neg_score block
# =======================================================================
def test_eval_retrieval_score_gap_block_absent_with_only_positives():
    """No negatives at all -> the separability block is not computed."""
    item = make_item(query="Q1", relevant=["v1"])
    retriever = TwoPassRetriever(
        wide={"Q1": make_result("Q1", [make_video("v1")])},
        prod={"Q1": make_result("Q1", [make_video("v1")])},
    )

    summary = eval_retrieval([item], retriever, rank_k=5, budget=None)["summary"]

    assert "pos_score_min" not in summary
    assert "neg_score_max" not in summary


def test_eval_retrieval_score_gap_block_absent_with_only_negatives():
    """No positives at all -> the separability block is not computed."""
    item = make_item(query="Q1", relevant=[], kind="negative", expect_found=False)
    retriever = TwoPassRetriever(
        wide={"Q1": make_result("Q1", [make_video("v1")])},
        prod={"Q1": make_result("Q1", [make_video("v1")])},
    )

    summary = eval_retrieval([item], retriever, rank_k=5, budget=None)["summary"]

    assert "pos_score_min" not in summary


def test_eval_retrieval_score_gap_block_present_with_both():
    """With one positive and one negative both scored, the gap block reports
    the positive's production score and the negative's."""
    pos = make_item(id="pos1", query="Q1", relevant=["v1"])
    neg = make_item(id="neg1", query="Q2", relevant=[], kind="negative", expect_found=False)
    retriever = TwoPassRetriever(
        wide={"Q1": make_result("Q1", [make_video("v1", score=0.8)]),
              "Q2": make_result("Q2", [make_video("other", score=0.5)])},
        prod={"Q1": make_result("Q1", [make_video("v1", score=0.8)]),
              "Q2": make_result("Q2", [make_video("other2", score=0.3)])},
    )

    summary = eval_retrieval([pos, neg], retriever, rank_k=5, budget=None)["summary"]

    assert summary["pos_score_min"] == pytest.approx(0.8)
    assert summary["pos_score_p10"] == pytest.approx(0.8)
    assert summary["neg_score_max"] == pytest.approx(0.3)


# =======================================================================
# eval_retrieval: rank_k / budget forwarding
# =======================================================================
def test_eval_retrieval_rank_k_forwarded_as_top_videos_to_wide_pass():
    item = make_item(query="Q1", relevant=["v1"])
    retriever = TwoPassRetriever(
        wide={"Q1": make_result("Q1", [make_video("v1")])},
        prod={"Q1": make_result("Q1", [make_video("v1")])},
    )

    eval_retrieval([item], retriever, rank_k=7, budget=None)

    wide_calls = [c for c in retriever.calls if c.get("rerank")]
    assert wide_calls[0]["top_videos"] == 7


def test_eval_retrieval_budget_none_omits_token_budget_from_prod_call():
    item = make_item(query="Q1", relevant=["v1"])
    retriever = TwoPassRetriever(
        wide={"Q1": make_result("Q1", [make_video("v1")])},
        prod={"Q1": make_result("Q1", [make_video("v1")])},
    )

    eval_retrieval([item], retriever, rank_k=5, budget=None)

    prod_calls = [c for c in retriever.calls if not c.get("rerank")]
    assert "token_budget" not in prod_calls[0]


def test_eval_retrieval_budget_set_is_forwarded_to_prod_call():
    item = make_item(query="Q1", relevant=["v1"])
    retriever = TwoPassRetriever(
        wide={"Q1": make_result("Q1", [make_video("v1")])},
        prod={"Q1": make_result("Q1", [make_video("v1")])},
    )

    eval_retrieval([item], retriever, rank_k=5, budget=1500)

    prod_calls = [c for c in retriever.calls if not c.get("rerank")]
    assert prod_calls[0]["token_budget"] == 1500


# =======================================================================
# eval_answer: found_acc
# =======================================================================
def test_eval_answer_found_acc_true_when_matching_expectation(monkeypatch):
    """A positive item where found matches expect_found scores 1.0."""
    item = make_item(query="Q1", relevant=["v1"], expect_found=True)
    fn = make_answer_fn({"Q1": make_recipe_answer(query="Q1", found=True)})
    monkeypatch.setattr("src.generation.answer.answer", fn)

    summary = eval_answer([item], retriever=None, llm=None)["summary"]

    assert summary["found_acc"] == pytest.approx(1.0)


def test_eval_answer_found_acc_accounts_for_negative_expectation(monkeypatch):
    """A negative item expecting found=false scores 1.0 when the pipeline
    honestly reports not-found."""
    item = make_item(query="Q1", relevant=[], kind="negative", expect_found=False)
    fn = make_answer_fn({"Q1": make_recipe_answer(query="Q1", found=False)})
    monkeypatch.setattr("src.generation.answer.answer", fn)

    summary = eval_answer([item], retriever=None, llm=None)["summary"]

    assert summary["found_acc"] == pytest.approx(1.0)


def test_eval_answer_found_acc_zero_on_mismatch(monkeypatch):
    """found=false when expect_found=true scores 0.0."""
    item = make_item(query="Q1", relevant=["v1"], expect_found=True)
    fn = make_answer_fn({"Q1": make_recipe_answer(query="Q1", found=False)})
    monkeypatch.setattr("src.generation.answer.answer", fn)

    summary = eval_answer([item], retriever=None, llm=None)["summary"]

    assert summary["found_acc"] == pytest.approx(0.0)


# =======================================================================
# eval_answer: attribution
# =======================================================================
def test_eval_answer_attribution_true_when_source_video_is_relevant(monkeypatch):
    item = make_item(query="Q1", relevant=["v1"])
    source = SourceRef(n=1, video_id="v1", title="t", url="u", timecode="00:10")
    fn = make_answer_fn({"Q1": make_recipe_answer(query="Q1", found=True, source=source)})
    monkeypatch.setattr("src.generation.answer.answer", fn)

    summary = eval_answer([item], retriever=None, llm=None)["summary"]

    assert summary["attribution"] == pytest.approx(1.0)


def test_eval_answer_attribution_excludes_negative_queries(monkeypatch):
    """A negative item that (incorrectly) comes back found=true is excluded
    from attribution: if it counted as a 0, attribution would drop to 0.5
    instead of staying 1.0."""
    pos = make_item(id="pos1", query="Q1", relevant=["v1"])
    neg = make_item(id="neg1", query="Q2", relevant=[], kind="negative", expect_found=False)
    pos_source = SourceRef(n=1, video_id="v1", title="t", url="u", timecode="00:10")
    neg_source = SourceRef(n=1, video_id="unrelated", title="t", url="u", timecode="00:10")
    fn = make_answer_fn({
        "Q1": make_recipe_answer(query="Q1", found=True, source=pos_source),
        "Q2": make_recipe_answer(query="Q2", found=True, source=neg_source),
    })
    monkeypatch.setattr("src.generation.answer.answer", fn)

    summary = eval_answer([pos, neg], retriever=None, llm=None)["summary"]

    assert summary["attribution"] == pytest.approx(1.0)


# =======================================================================
# eval_answer: steps/ingredients averaged only over found answers
# =======================================================================
def test_eval_answer_steps_and_ingredients_averaged_only_over_found(monkeypatch):
    """A not-found item's steps/ingredients must not enter the average: if it
    did, avg_steps would be 1.5 instead of 2.0."""
    found_item = make_item(id="q1", query="Q1", relevant=["v1"])
    not_found_item = make_item(id="q2", query="Q2", relevant=["v2"])
    fn = make_answer_fn({
        "Q1": make_recipe_answer(query="Q1", found=True, steps=["a", "b"], ingredients=["x"]),
        "Q2": make_recipe_answer(query="Q2", found=False, steps=["c"], ingredients=["y"]),
    })
    monkeypatch.setattr("src.generation.answer.answer", fn)

    summary = eval_answer([found_item, not_found_item], retriever=None, llm=None)["summary"]

    assert summary["avg_steps"] == pytest.approx(2.0)
    assert summary["avg_ingredients"] == pytest.approx(1.0)


# =======================================================================
# eval_answer: with_amounts
# =======================================================================
def test_eval_answer_with_amounts_is_share_of_ingredients_with_a_digit(monkeypatch):
    item = make_item(query="Q1", relevant=["v1"])
    fn = make_answer_fn({
        "Q1": make_recipe_answer(query="Q1", found=True, ingredients=["соль", "сахар 200 г"]),
    })
    monkeypatch.setattr("src.generation.answer.answer", fn)

    summary = eval_answer([item], retriever=None, llm=None)["summary"]

    assert summary["with_amounts"] == pytest.approx(0.5)


def test_eval_answer_with_amounts_no_ingredients_does_not_divide_by_zero(monkeypatch):
    """found=true with an empty ingredients list must not raise
    ZeroDivisionError, and the metric falls back to 0.0."""
    item = make_item(query="Q1", relevant=["v1"])
    fn = make_answer_fn({"Q1": make_recipe_answer(query="Q1", found=True, ingredients=[])})
    monkeypatch.setattr("src.generation.answer.answer", fn)

    summary = eval_answer([item], retriever=None, llm=None)["summary"]

    assert summary["with_amounts"] == pytest.approx(0.0)


# =======================================================================
# eval_answer: exceptions from answer() are caught and the run continues
# =======================================================================
def test_eval_answer_exception_produces_error_row_and_continues(monkeypatch):
    """An exception on one item does not abort the run: the next item is
    still processed and its metrics contribute to the summary."""
    failing = make_item(id="q1", query="Q1", relevant=["v1"])
    ok = make_item(id="q2", query="Q2", relevant=["v2"])
    fn = make_answer_fn({
        "Q1": RuntimeError("LLM timed out"),
        "Q2": make_recipe_answer(query="Q2", found=True),
    })
    monkeypatch.setattr("src.generation.answer.answer", fn)

    result = eval_answer([failing, ok], retriever=None, llm=None)

    assert result["rows"][0] == {"id": "q1", "query": "Q1", "error": "LLM timed out"}
    assert result["summary"]["n"] == 2
    assert result["summary"]["found_acc"] == pytest.approx(1.0)


def test_eval_answer_error_message_is_truncated_to_200_chars(monkeypatch):
    long_message = "x" * 250
    item = make_item(query="Q1", relevant=["v1"])
    fn = make_answer_fn({"Q1": RuntimeError(long_message)})
    monkeypatch.setattr("src.generation.answer.answer", fn)

    rows = eval_answer([item], retriever=None, llm=None)["rows"]

    assert len(rows[0]["error"]) == 200
    assert rows[0]["error"] == long_message[:200]


# =======================================================================
# eval_answer: row shape
# =======================================================================
def test_eval_answer_row_contains_expected_fields(monkeypatch):
    source = SourceRef(n=1, video_id="v1", title="t", url="u", timecode="00:10")
    item = make_item(id="q1", query="Q1", relevant=["v1"], kind="descriptive", expect_found=True)
    fn = make_answer_fn({
        "Q1": make_recipe_answer(query="Q1", found=True, steps=["a"], ingredients=["b", "c"], source=source),
    })
    monkeypatch.setattr("src.generation.answer.answer", fn)

    counter = iter([100.0, 100.5])
    monkeypatch.setattr("src.eval.run.time.time", lambda: next(counter))

    rows = eval_answer([item], retriever=None, llm=None)["rows"]

    assert rows[0] == {
        "id": "q1", "query": "Q1", "kind": "descriptive",
        "found": True, "expect_found": True,
        "steps": 1, "ingredients": 2,
        "source": "v1", "seconds": pytest.approx(0.5),
    }


def test_eval_answer_row_source_is_none_without_a_source(monkeypatch):
    item = make_item(query="Q1", relevant=["v1"])
    fn = make_answer_fn({"Q1": make_recipe_answer(query="Q1", found=False, source=None)})
    monkeypatch.setattr("src.generation.answer.answer", fn)

    rows = eval_answer([item], retriever=None, llm=None)["rows"]

    assert rows[0]["source"] is None


# =======================================================================
# eval_answer: attribution has to be able to come out wrong
# =======================================================================
def test_eval_answer_attribution_zero_when_source_video_is_not_relevant(monkeypatch):
    """A confident answer citing the WRONG video scores zero attribution.

    Every existing attribution test expects 1.0, so the metric could be a
    constant and no one would notice — this is the case that makes it a
    measurement.
    """
    item = make_item(id="q1", query="стейк", relevant=["v1"])
    answer_fn = make_answer_fn({
        "стейк": make_recipe_answer(
            found=True,
            source=SourceRef(n=1, video_id="v999", title="Другое", url="u", timecode="00:01"),
        )
    })
    monkeypatch.setattr("src.generation.answer.answer", answer_fn)

    report = eval_answer([item], retriever=None, llm=None)

    assert report["summary"]["attribution"] == pytest.approx(0.0)


def test_eval_answer_attribution_zero_when_the_answer_cites_no_source(monkeypatch):
    """found=true with source=None also scores zero: the guard is
    `res.source and ...`, and dropping it would raise instead of scoring."""
    item = make_item(id="q1", query="стейк", relevant=["v1"])
    answer_fn = make_answer_fn({"стейк": make_recipe_answer(found=True, source=None)})
    monkeypatch.setattr("src.generation.answer.answer", answer_fn)

    report = eval_answer([item], retriever=None, llm=None)

    assert report["summary"]["attribution"] == pytest.approx(0.0)


def test_eval_answer_with_amounts_counts_only_ingredients_carrying_a_number(monkeypatch):
    """The share is computed over an asymmetric list, so inverting the digit
    test changes the answer instead of leaving it at one half."""
    item = make_item(id="q1", query="стейк", relevant=["v1"])
    answer_fn = make_answer_fn({
        "стейк": make_recipe_answer(
            found=True,
            ingredients=["соль", "перец", "стейк 300 г"],
            source=SourceRef(n=1, video_id="v1", title="Видео", url="u", timecode="00:01"),
        )
    })
    monkeypatch.setattr("src.generation.answer.answer", answer_fn)

    report = eval_answer([item], retriever=None, llm=None)

    # one of three, rounded to four places by _mean
    assert report["summary"]["with_amounts"] == pytest.approx(0.3333)


def test_eval_answer_empty_item_list_returns_zeroed_summary(monkeypatch):
    """No golden items at all -> every metric is 0.0 rather than a crash on an
    empty mean."""
    monkeypatch.setattr("src.generation.answer.answer", make_answer_fn({}))

    report = eval_answer([], retriever=None, llm=None)

    assert report["rows"] == []
    assert report["summary"]["n"] == 0
    assert report["summary"]["found_acc"] == 0.0
    assert report["summary"]["attribution"] == 0.0


# =======================================================================
# eval_retrieval: empty inputs and empty retrieval results
# =======================================================================
def test_eval_retrieval_empty_item_list_returns_zeroed_summary():
    """An empty golden set produces an empty report, not a division by zero."""
    report = eval_retrieval([], TwoPassRetriever(), rank_k=10, budget=None)

    assert report["rows"] == []
    assert report["by_kind"] == {}
    assert report["summary"]["n"] == 0
    assert report["summary"]["mrr"] == 0.0
    assert report["summary"]["hit@1"] == 0.0
    # the score-separability block only exists when both classes are present
    assert "pos_score_min" not in report["summary"]


def test_eval_retrieval_handles_a_query_that_retrieved_nothing():
    """Qdrant returning no videos at all (an over-eager intent filter, say) is
    a miss with zero scores, not a crash on videos[0]."""
    item = make_item(id="q1", query="ничего", relevant=["v1"])
    retriever = TwoPassRetriever(
        wide={"ничего": make_result("ничего", [])},
        prod={"ничего": make_result("ничего", [])},
    )

    report = eval_retrieval([item], retriever, rank_k=10, budget=None)

    row = report["rows"][0]
    assert row["rank"] is None
    assert row["top_score"] == 0.0
    assert row["prod_score"] == 0.0
    assert report["summary"]["mrr"] == 0.0
    assert report["summary"]["ctx_precision"] == 0.0


def test_eval_retrieval_edge_rate_ignores_an_unparsable_chunk_id():
    """A chunk id the parser cannot read yields -1, which must NOT count as the
    opening chunk: the comparison is `== 0`, and a `<= 0` there would report
    junk ids as intros."""
    item = make_item(id="q1", query="стейк", relevant=["v1"])
    passages = [make_passage(video_id="v1", chunk_ids=["v1::мусор"])]
    retriever = TwoPassRetriever(
        wide={"стейк": make_result("стейк", [make_video("v1")])},
        prod={"стейк": make_result("стейк", [make_video("v1", passages=passages)])},
    )

    report = eval_retrieval([item], retriever, rank_k=10, budget=None)

    assert report["summary"]["edge_rate"] == pytest.approx(0.0)


def test_eval_retrieval_by_kind_ignores_negatives_sharing_a_kind_with_positives():
    """A negative labelled with the same kind as a positive must not inflate
    that kind's n: the rows are filtered by the positive ids, not by kind
    alone."""
    positive = make_item(id="q1", query="стейк", relevant=["v1"], kind="exact")
    negative = make_item(id="q2", query="ничего", relevant=[], kind="exact")
    retriever = TwoPassRetriever(
        wide={
            "стейк": make_result("стейк", [make_video("v1")]),
            "ничего": make_result("ничего", [make_video("v9", score=0.01)]),
        },
        prod={
            "стейк": make_result("стейк", [make_video("v1", passages=[make_passage()])]),
            "ничего": make_result("ничего", [make_video("v9", score=0.01)]),
        },
    )

    report = eval_retrieval([positive, negative], retriever, rank_k=10, budget=None)

    assert report["by_kind"]["exact"]["n"] == 1
    assert report["summary"]["n"] == 1
