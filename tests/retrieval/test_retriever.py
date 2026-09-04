"""Tests for src/retrieval/retriever.py: the pure helpers (_truncate_words,
_stitch, _position_weight, _merge_adjacent), grouping by video with the
neighbour fetch (_group/_with_neighbors), context assembly within a budget
(_build_context), retrieve() end to end on fakes, and the get_retriever()
singleton.

Note: the RETRIEVAL_* values are the defaults of retrieve()'s parameters,
evaluated when the config is imported. Wherever a test depends on them they are
passed as explicit arguments rather than patched.
"""
from __future__ import annotations

import pytest
from qdrant_client import models

from src.config import (
    RETRIEVAL_FIRST_CHUNK_WEIGHT,
    RETRIEVAL_LAST_CHUNK_WEIGHT,
    RETRIEVAL_SECTION_WEIGHTS,
)
from src.retrieval.retriever import Retriever, _stitch, _truncate_words
from src.retrieval.schemas import RetrievedChunk, RetrievedPassage, RetrievedVideo
from tests.conftest import (
    FakeEmbedder,
    FakePoint,
    FakeReranker,
    FakeStore,
    make_payload,
    make_point,
)


# --- helpers local to this file ---------------------------------------
def make_chunk(
    *,
    chunk_id: str = "c1",
    video_id: str = "vid1",
    title: str = "Видео",
    url: str = "https://youtu.be/vid1",
    timecode: str = "00:00",
    start: float = 0.0,
    end: float = 10.0,
    chunk_index: int = 1,
    n_chunks: int = 5,
    section: str = "steps",
    text: str = "текст чанка",
    score: float = 1.0,
    retriever_score: float | None = None,
) -> RetrievedChunk:
    """Builds a RetrievedChunk directly, without going through retrieve()."""
    return RetrievedChunk(
        chunk_id=chunk_id,
        video_id=video_id,
        title=title,
        url=url,
        timecode=timecode,
        start=start,
        end=end,
        chunk_index=chunk_index,
        n_chunks=n_chunks,
        section=section,
        text=text,
        score=score,
        retriever_score=retriever_score if retriever_score is not None else score,
    )


class WordCounter:
    """A predictable token counter: 1 token = 1 word.

    The real TokenCounter counts by CHARS_PER_TOKEN, which is awkward for the
    exact budget tests of _build_context — here we want a count that is trivial
    to do in one's head.
    """

    def count(self, text: str) -> int:
        return len(text.split())


def make_retriever(
    *, store: FakeStore | None = None, reranker=None, counter=None
) -> Retriever:
    return Retriever(
        store=store or FakeStore(),
        embedder=FakeEmbedder(),
        reranker=reranker,
        token_counter=counter or WordCounter(),
    )


# =======================================================================
# _truncate_words
# =======================================================================
def test_truncate_words_returns_unchanged_when_shorter_than_limit():
    """Text shorter than the limit comes back as is, with no ellipsis."""
    text = "раз два три"
    assert _truncate_words(text, 5) == text


def test_truncate_words_returns_unchanged_when_exactly_at_limit():
    """Text exactly at the word limit is not truncated."""
    text = "раз два три"
    assert _truncate_words(text, 3) == text


def test_truncate_words_adds_ellipsis_when_longer_than_limit():
    """Text longer than the limit is cut on a word and gets an ellipsis."""
    text = "раз два три четыре"
    assert _truncate_words(text, 2) == "раз два …"


# =======================================================================
# _stitch
# =======================================================================
def test_stitch_removes_full_overlap_of_tail_and_head():
    """A tail of a matching the head of b collapses without duplication."""
    a = "один два три"
    b = "два три четыре"
    assert _stitch(a, b) == "один два три четыре"


def test_stitch_concatenates_with_space_when_no_overlap():
    """No shared words -> a plain concatenation with a space."""
    assert _stitch("привет", "мир") == "привет мир"


def test_stitch_overlap_comparison_is_case_insensitive():
    """The overlap is found regardless of the case of the words."""
    a = "Обжарьте Стейк"
    b = "стейк на огне"
    # A one-word overlap is found ("Стейк"/"стейк"); the result keeps the
    # casing from a, not from b.
    assert _stitch(a, b) == "Обжарьте Стейк на огне"


def test_stitch_detects_single_word_overlap():
    """An overlap of a single word is detected and collapsed too."""
    assert _stitch("вода соль сахар", "сахар мука") == "вода соль сахар мука"


def test_stitch_max_overlap_words_caps_detection():
    """max_overlap_words caps how long an overlap may be: a real two-word
    overlap is not found when the cap is one."""
    a = "один два три четыре пять"
    b = "четыре пять шесть семь"
    # The real overlap is "четыре пять" (2 words), but the cap is 1: at n=1 the
    # last word of a ("пять") differs from the first word of b ("четыре"), so
    # no overlap is found and the texts are glued with the duplication intact.
    assert _stitch(a, b, max_overlap_words=1) == a + " " + b


# =======================================================================
# _position_weight
# =======================================================================
def test_position_weight_first_chunk_gets_first_chunk_multiplier():
    """chunk_index=0 -> the section weight is scaled by the first-chunk weight."""
    c = make_chunk(chunk_index=0, n_chunks=5, section="other")
    assert Retriever._position_weight(c) == pytest.approx(RETRIEVAL_FIRST_CHUNK_WEIGHT)


def test_position_weight_last_chunk_gets_last_chunk_multiplier():
    """The last chunk of a video is scaled by the last-chunk weight."""
    c = make_chunk(chunk_index=4, n_chunks=5, section="other")
    assert Retriever._position_weight(c) == pytest.approx(RETRIEVAL_LAST_CHUNK_WEIGHT)


def test_position_weight_middle_chunk_uses_only_section_weight():
    """A middle chunk gets its section weight and no positional multiplier."""
    c = make_chunk(chunk_index=2, n_chunks=5, section="ingredients")
    # 1.15 is RETRIEVAL_SECTION_WEIGHTS["ingredients"]; spelled out as a literal
    # so that a change of the config value shows up here as a failure.
    assert Retriever._position_weight(c) == pytest.approx(1.15)
    assert RETRIEVAL_SECTION_WEIGHTS["ingredients"] == pytest.approx(1.15)


def test_position_weight_unknown_section_defaults_to_one():
    """A section absent from RETRIEVAL_SECTION_WEIGHTS yields a 1.0 multiplier."""
    c = make_chunk(chunk_index=2, n_chunks=5, section="совсем незнакомая секция")
    assert Retriever._position_weight(c) == pytest.approx(1.0)


def test_position_weight_zero_n_chunks_is_not_treated_as_last():
    """n_chunks=0 is falsy for the `and`, so the chunk is NOT counted as last
    even though chunk_index >= n_chunks - 1 holds."""
    c = make_chunk(chunk_index=3, n_chunks=0, section="other")
    assert Retriever._position_weight(c) == pytest.approx(1.0)


# =======================================================================
# _merge_adjacent
# =======================================================================
def test_merge_adjacent_stitches_consecutive_chunk_indices():
    """Chunks adjacent by chunk_index are stitched into one passage."""
    chunks = [
        make_chunk(chunk_id="a", chunk_index=1, text="один два три", start=0.0, end=5.0, score=0.4),
        make_chunk(chunk_id="b", chunk_index=2, text="три четыре", start=5.0, end=10.0, score=0.9),
    ]
    passages = Retriever._merge_adjacent(chunks)
    assert len(passages) == 1
    assert passages[0].text == "один два три четыре"
    assert passages[0].chunk_ids == ["a", "b"]


def test_merge_adjacent_breaks_on_index_gap():
    """A gap in chunk_index keeps the chunks in separate passages."""
    chunks = [
        make_chunk(chunk_id="a", chunk_index=1, text="раз", start=0.0, end=5.0),
        make_chunk(chunk_id="b", chunk_index=3, text="два", start=10.0, end=15.0),
    ]
    passages = Retriever._merge_adjacent(chunks)
    assert len(passages) == 2
    assert [p.chunk_ids for p in passages] == [["a"], ["b"]]


def test_merge_adjacent_caps_group_by_max_words():
    """A group is cut off when the next adjacent chunk would not fit into
    max_words, even though it follows by chunk_index."""
    chunks = [
        make_chunk(chunk_id="a", chunk_index=1, text="одно два три", start=0.0, end=1.0),
        make_chunk(chunk_id="b", chunk_index=2, text="четыре пять шесть", start=1.0, end=2.0),
        make_chunk(chunk_id="c", chunk_index=3, text="семь восемь девять", start=2.0, end=3.0),
    ]
    # A 7-word cap: "a"+"b" = 6 words fit, "a"+"b"+"c" = 9 do not, so "c" starts
    # a new group despite being adjacent by chunk_index.
    passages = Retriever._merge_adjacent(chunks, max_words=7)
    assert [p.chunk_ids for p in passages] == [["a", "b"], ["c"]]


def test_merge_adjacent_passage_score_is_max_of_group():
    """A passage scores the maximum over its chunks, not the first or the last."""
    chunks = [
        make_chunk(chunk_id="a", chunk_index=1, text="раз", start=0.0, end=1.0, score=0.2),
        make_chunk(chunk_id="b", chunk_index=2, text="два", start=1.0, end=2.0, score=0.9),
        make_chunk(chunk_id="c", chunk_index=3, text="три", start=2.0, end=3.0, score=0.1),
    ]
    passages = Retriever._merge_adjacent(chunks)
    assert passages[0].score == pytest.approx(0.9)


def test_merge_adjacent_sorts_result_by_start():
    """The resulting passages are sorted by start, not by input order."""
    chunks = [
        make_chunk(chunk_id="late", chunk_index=5, text="поздний", start=100.0, end=105.0),
        make_chunk(chunk_id="early", chunk_index=1, text="ранний", start=10.0, end=15.0),
    ]
    passages = Retriever._merge_adjacent(chunks)
    assert [p.chunk_ids for p in passages] == [["early"], ["late"]]


def test_merge_adjacent_empty_input_returns_empty_list():
    """An empty chunk list -> an empty passage list."""
    assert Retriever._merge_adjacent([]) == []


# =======================================================================
# _group
# =======================================================================
def test_group_ranks_video_by_its_best_chunk_not_its_worst_or_average():
    """A video is ranked by the MAXIMUM score among its chunks.

    vidA holds one excellent chunk and one poor one; vidB holds two mediocre
    ones. By the maximum vidA wins; by the minimum, the average or the last
    chunk vidB would win instead.
    """
    cands = [
        make_chunk(chunk_id="a1", video_id="vidA", chunk_index=1, score=10.0, start=0.0, end=1.0),
        make_chunk(chunk_id="a2", video_id="vidA", chunk_index=9, score=1.0, start=90.0, end=91.0),
        make_chunk(chunk_id="b1", video_id="vidB", chunk_index=1, score=6.0, start=0.0, end=1.0),
        make_chunk(chunk_id="b2", video_id="vidB", chunk_index=9, score=6.0, start=90.0, end=91.0),
    ]
    r = make_retriever()
    videos = r._group(cands, top_videos=2, per_video=2, neighbor_radius=0)

    assert [v.video_id for v in videos] == ["vidA", "vidB"]
    assert videos[0].score == pytest.approx(10.0)
    assert videos[1].score == pytest.approx(6.0)


def test_group_ranks_video_before_fetching_neighbors():
    """The video rank is computed BEFORE the neighbours are fetched: a losing
    video never enters top_videos, so store.fetch_chunks is not called for it."""
    cands = [
        make_chunk(chunk_id="winner", video_id="vidA", chunk_index=5, score=10.0),
        make_chunk(chunk_id="loser", video_id="vidB", chunk_index=5, score=1.0),
    ]
    store = FakeStore(corpus=[make_payload(video_id="vidB", chunk_index=4)])
    r = make_retriever(store=store)

    videos = r._group(cands, top_videos=1, per_video=1, neighbor_radius=1)

    assert [v.video_id for v in videos] == ["vidA"]
    assert store.fetch_calls == [("vidA", [4, 6])]


def test_group_video_score_ignores_zero_scored_neighbors():
    """Neighbours arrive with score=0 as filler: they join the passages but do
    not dilute the score the video was ranked with."""
    cands = [make_chunk(chunk_id="hit", video_id="v1", chunk_index=5, score=7.0, start=50.0, end=55.0)]
    store = FakeStore(
        corpus=[
            make_payload(video_id="v1", chunk_index=4, start=40.0, end=50.0),
            make_payload(video_id="v1", chunk_index=6, start=55.0, end=60.0),
        ]
    )
    r = make_retriever(store=store)

    videos = r._group(cands, top_videos=1, per_video=1, neighbor_radius=1)

    kept = {cid for p in videos[0].passages for cid in p.chunk_ids}
    assert kept == {"hit", "v1::004", "v1::006"}
    assert videos[0].score == pytest.approx(7.0)


def test_group_limits_to_top_videos_by_max_score():
    """top_videos trims the result to N videos ranked by their best chunk."""
    cands = [
        make_chunk(chunk_id="a", video_id="v1", chunk_index=1, score=5.0),
        make_chunk(chunk_id="b", video_id="v2", chunk_index=1, score=9.0),
        make_chunk(chunk_id="c", video_id="v3", chunk_index=1, score=1.0),
    ]
    r = make_retriever()
    videos = r._group(cands, top_videos=2, per_video=1, neighbor_radius=0)
    assert [v.video_id for v in videos] == ["v2", "v1"]


def test_group_limits_chunks_to_per_video_by_score():
    """per_video keeps only the top-N chunks of a video by score."""
    cands = [
        make_chunk(chunk_id="low", video_id="v1", chunk_index=1, text="низкий", score=1.0, start=0.0, end=1.0),
        make_chunk(chunk_id="high", video_id="v1", chunk_index=10, text="высокий", score=9.0, start=100.0, end=101.0),
        make_chunk(chunk_id="mid", video_id="v1", chunk_index=20, text="средний", score=5.0, start=200.0, end=201.0),
    ]
    r = make_retriever()
    videos = r._group(cands, top_videos=1, per_video=2, neighbor_radius=0)
    kept = {cid for p in videos[0].passages for cid in p.chunk_ids}
    assert kept == {"high", "mid"}


def test_group_orders_chunks_by_chunk_index_before_merging():
    """Chunks are sorted by chunk_index before stitching, not left in score
    order — otherwise time-adjacent chunks would never merge."""
    cands = [
        make_chunk(chunk_id="second", video_id="v1", chunk_index=2, text="два", score=9.0, start=5.0, end=10.0),
        make_chunk(chunk_id="first", video_id="v1", chunk_index=1, text="один", score=1.0, start=0.0, end=5.0),
    ]
    r = make_retriever()
    videos = r._group(cands, top_videos=1, per_video=2, neighbor_radius=0)
    assert len(videos[0].passages) == 1
    assert videos[0].passages[0].chunk_ids == ["first", "second"]


# =======================================================================
# _with_neighbors
# =======================================================================
def test_with_neighbors_requests_indices_within_radius():
    """The wanted index set is built within the radius around every chunk."""
    store = FakeStore(corpus=[])
    r = make_retriever(store=store)
    chunks = [make_chunk(chunk_id="a", video_id="v1", chunk_index=5)]
    r._with_neighbors("v1", chunks, radius=2)
    assert store.fetch_calls == [("v1", [3, 4, 6, 7])]


def test_with_neighbors_drops_negative_and_existing_indices():
    """Negative indices and indices already held are never requested."""
    store = FakeStore(corpus=[])
    r = make_retriever(store=store)
    chunks = [
        make_chunk(chunk_id="a", video_id="v1", chunk_index=0),
        make_chunk(chunk_id="b", video_id="v1", chunk_index=1),
    ]
    r._with_neighbors("v1", chunks, radius=1)
    # from idx=0: -1 (dropped) and 1 (already held); from idx=1: 0 (held), 2 (new)
    assert store.fetch_calls == [("v1", [2])]


def test_with_neighbors_empty_wanted_set_does_not_call_store():
    """An empty wanted set (an empty chunk list) -> the store is not touched."""
    store = FakeStore(corpus=[])
    r = make_retriever(store=store)
    result = r._with_neighbors("v1", [], radius=1)
    assert result == []
    assert store.fetch_calls == []


def test_with_neighbors_adds_fetched_chunks_with_zero_score():
    """Successfully fetched neighbours are appended with score=0.0."""
    store = FakeStore(corpus=[make_payload(video_id="v1", chunk_index=6, text="сосед")])
    r = make_retriever(store=store)
    chunks = [make_chunk(chunk_id="a", video_id="v1", chunk_index=5, score=7.0)]
    result = r._with_neighbors("v1", chunks, radius=1)
    assert len(result) == 2
    assert result[1].score == 0.0
    assert result[1].retriever_score == 0.0
    assert result[1].text == "сосед"


def test_with_neighbors_returns_original_chunks_on_store_exception():
    """The except branch: when store.fetch_chunks raises, the original chunks
    come back without neighbours and the exception does not escape."""
    store = FakeStore(fetch_error=RuntimeError("qdrant недоступен"))
    r = make_retriever(store=store)
    chunks = [make_chunk(chunk_id="a", video_id="v1", chunk_index=5)]
    result = r._with_neighbors("v1", chunks, radius=1)
    assert result == chunks


# =======================================================================
# _build_context
# =======================================================================
def make_passage(
    *,
    video_id: str = "v1",
    title: str = "Видео",
    url: str = "https://youtu.be/v1",
    timecode: str = "00:00",
    start: float = 0.0,
    end: float = 10.0,
    text: str = "текст",
    chunk_ids: list[str] | None = None,
    score: float = 1.0,
) -> RetrievedPassage:
    return RetrievedPassage(
        video_id=video_id,
        title=title,
        url=url,
        timecode=timecode,
        start=start,
        end=end,
        text=text,
        chunk_ids=chunk_ids or ["c1"],
        score=score,
    )


def test_build_context_empty_videos_returns_empty_context():
    """An empty video list -> an empty context string and no citations."""
    r = make_retriever()
    context, citations = r._build_context([], budget=1000)
    assert context == ""
    assert citations == []


def test_build_context_always_includes_first_selected_passage_even_over_budget():
    """The first (most relevant) passage is always included, even when its cost
    alone exceeds the budget."""
    passage = make_passage(text=" ".join(["слово"] * 30), score=1.0)  # cost = 30 + 24 = 54
    video = RetrievedVideo(video_id="v1", title="Видео", score=1.0, passages=[passage])
    r = make_retriever()
    context, citations = r._build_context([video], budget=1)
    assert len(citations) == 1
    assert "слово" in context


def test_build_context_continues_past_a_skipped_passage_to_fit_a_smaller_one():
    """continue, not break: a passage that does not fit is skipped, and a
    smaller one with a lower score is still examined and added."""
    big = make_passage(video_id="v1", text=" ".join(["big"] * 40), score=10.0, start=0.0)   # cost 64
    mid = make_passage(video_id="v1", text=" ".join(["mid"] * 40), score=8.0, start=1.0)    # cost 64
    small = make_passage(video_id="v1", text=" ".join(["small"] * 16), score=5.0, start=2.0)  # cost 40
    video = RetrievedVideo(video_id="v1", title="Видео", score=10.0, passages=[big, mid, small])
    r = make_retriever()
    # budget=110: big (64) is taken first; mid would make 128 > 110 and is
    # skipped; small still fits at 64 + 40 = 104.
    context, citations = r._build_context([video], budget=110)
    assert len(citations) == 2
    assert "big" in context
    assert "small" in context
    assert "mid" not in context


def test_build_context_citation_numbers_match_block_numbers_in_text():
    """The [n] marker in a block matches the "n" field of the citation."""
    p1 = make_passage(video_id="v1", title="Первое", timecode="00:01", score=5.0, start=0.0)
    p2 = make_passage(video_id="v1", title="Второе", timecode="00:02", score=1.0, start=10.0)
    video = RetrievedVideo(video_id="v1", title="Видео", score=5.0, passages=[p1, p2])
    r = make_retriever()
    context, citations = r._build_context([video], budget=10_000)
    assert citations[0]["n"] == 1 and citations[1]["n"] == 2
    assert "[1] Первое (00:01)" in context
    assert "[2] Второе (00:02)" in context


def test_build_context_orders_passages_of_one_video_by_time_not_by_score():
    """Selection goes by relevance, the order in the prompt goes by time.

    Inside one video the more relevant passage comes LATER in the recording, so
    a prompt ordered by score would hand the LLM the end of the recipe before
    its beginning.
    """
    early = make_passage(video_id="v1", title="Ранний", timecode="00:10", score=1.0, start=10.0)
    late = make_passage(video_id="v1", title="Поздний", timecode="01:40", score=9.0, start=100.0)
    video = RetrievedVideo(video_id="v1", title="Видео", score=9.0, passages=[early, late])
    r = make_retriever()
    context, citations = r._build_context([video], budget=10_000)

    assert [c["timecode"] for c in citations] == ["00:10", "01:40"]
    assert context.index("Ранний") < context.index("Поздний")


def test_build_context_orders_by_video_rank_then_passage_start():
    """Videos come in rank order (their position in the videos list), and the
    passages inside a video by time, regardless of score."""
    low_score_first_video = make_passage(video_id="v0", title="Ноль", timecode="00:05", score=1.0, start=5.0)
    high_score_second_video = make_passage(video_id="v1", title="Один", timecode="00:01", score=9.0, start=1.0)
    v0 = RetrievedVideo(video_id="v0", title="Видео0", score=1.0, passages=[low_score_first_video])
    v1 = RetrievedVideo(video_id="v1", title="Видео1", score=9.0, passages=[high_score_second_video])
    r = make_retriever()
    context, citations = r._build_context([v0, v1], budget=10_000)
    assert citations[0]["video_id"] == "v0"
    assert citations[1]["video_id"] == "v1"
    assert context.index("Ноль") < context.index("Один")


# =======================================================================
# retrieve() end to end, on fakes
# =======================================================================
def test_retrieve_zeroes_score_for_tail_beyond_rerank_top():
    """Candidates past rerank_top are zeroed so they cannot overtake the
    reranked head."""
    points = [
        make_point(score=0.1, video_id="v0", chunk_id="p0", chunk_index=1, n_chunks=5, section="steps"),
        make_point(score=0.2, video_id="v1", chunk_id="p1", chunk_index=1, n_chunks=5, section="steps"),
        make_point(score=0.3, video_id="v2", chunk_id="p2", chunk_index=1, n_chunks=5, section="steps"),
        make_point(score=0.4, video_id="v3", chunk_id="p3", chunk_index=1, n_chunks=5, section="steps"),
    ]
    store = FakeStore(points=points)
    reranker = FakeReranker(scores=[0.9, 0.5])  # scores only for the first rerank_top=2
    r = make_retriever(store=store, reranker=reranker)

    result = r.retrieve(
        "запрос",
        top_videos=4,
        per_video=1,
        overfetch=1,
        rerank=True,
        rerank_top=2,
        neighbor_radius=0,
        section_weights=False,
    )

    by_video = {v.video_id: v for v in result.videos}
    assert result.used_reranker is True
    assert by_video["v0"].score == pytest.approx(0.9)
    assert by_video["v1"].score == pytest.approx(0.5)
    assert by_video["v2"].score == 0.0
    assert by_video["v3"].score == 0.0


def test_retrieve_applies_section_weights_by_default():
    """section_weights=True (the default) rescales the score by the positional
    multiplier and can change the winner: a chunk with a higher raw score but
    sitting at position zero (the greeting) loses to the middle of a video."""
    points = [
        make_point(score=1.0, video_id="v_intro", chunk_id="a", chunk_index=0, n_chunks=5, section="other"),
        make_point(score=0.5, video_id="v_middle", chunk_id="b", chunk_index=2, n_chunks=5, section="other"),
    ]
    store = FakeStore(points=points)
    r = make_retriever(store=store, reranker=None)

    result = r.retrieve("запрос", top_videos=1, per_video=1, overfetch=1, rerank_top=2, neighbor_radius=0)

    assert result.videos[0].video_id == "v_middle"
    assert result.videos[0].score == pytest.approx(0.5)


def test_retrieve_used_reranker_false_when_rerank_flag_is_false():
    """rerank=False -> the reranker never touches the scores."""
    points = [make_point(score=0.42, video_id="v0", chunk_index=1, n_chunks=5, section="steps")]
    store = FakeStore(points=points)
    reranker = FakeReranker(scores=[0.99])
    r = make_retriever(store=store, reranker=reranker)

    result = r.retrieve("запрос", rerank=False, neighbor_radius=0, section_weights=False)

    assert result.used_reranker is False
    assert result.videos[0].score == pytest.approx(0.42)


def test_retrieve_used_reranker_false_when_no_reranker_object():
    """reranker=None on the Retriever -> no reranking even with rerank=True."""
    points = [make_point(score=0.42, video_id="v0", chunk_index=1, n_chunks=5, section="steps")]
    store = FakeStore(points=points)
    r = make_retriever(store=store, reranker=None)

    result = r.retrieve("запрос", rerank=True, neighbor_radius=0, section_weights=False)

    assert result.used_reranker is False
    assert result.videos[0].score == pytest.approx(0.42)


def test_retrieve_pool_uses_rerank_top_when_it_exceeds_the_product():
    """The candidate pool is the larger of top_videos*per_video*overfetch and
    rerank_top: here the product is 12, so rerank_top=24 wins."""
    store = FakeStore(points=[])
    r = make_retriever(store=store, reranker=None)

    result = r.retrieve(
        "запрос", top_videos=2, per_video=3, overfetch=2, rerank_top=24,
        neighbor_radius=0, section_weights=False,
    )

    assert result.debug["pool"] == 24
    assert store.search_calls[0]["k"] == 24


def test_retrieve_pool_uses_the_product_when_it_exceeds_rerank_top():
    """The same rule the other way round: the product is 60 and beats
    rerank_top=5."""
    store = FakeStore(points=[])
    r = make_retriever(store=store, reranker=None)

    result = r.retrieve(
        "запрос", top_videos=2, per_video=3, overfetch=10, rerank_top=5,
        neighbor_radius=0, section_weights=False,
    )

    assert result.debug["pool"] == 60
    assert store.search_calls[0]["k"] == 60


def test_retrieve_debug_reports_candidate_and_passage_counts():
    """debug counts the candidates that came back and the passages that were
    assembled out of them."""
    points = [
        make_point(score=0.5, video_id="v0", chunk_id="a", chunk_index=1, n_chunks=5),
        make_point(score=0.4, video_id="v0", chunk_id="b", chunk_index=7, n_chunks=9),
    ]
    store = FakeStore(points=points)
    r = make_retriever(store=store, reranker=None)

    result = r.retrieve(
        "запрос", top_videos=1, per_video=2, neighbor_radius=0, section_weights=False,
    )

    assert result.debug["candidates"] == 2
    # the chunks are not adjacent (1 and 7), so they stay two passages
    assert result.debug["passages"] == 2


def test_retrieve_debug_echoes_neighbor_radius():
    """debug echoes back the neighbor_radius the call was made with."""
    store = FakeStore(points=[])
    r = make_retriever(store=store, reranker=None)

    result = r.retrieve("запрос", neighbor_radius=3, section_weights=False)

    assert result.debug["neighbor_radius"] == 3


def test_retrieve_use_intent_filter_true_passes_the_built_filter_to_store():
    """use_intent_filter=True passes the filter built from the query into
    store.search and reports it in debug."""
    store = FakeStore(points=[])
    r = make_retriever(store=store, reranker=None)

    result = r.retrieve("хочу борщ", use_intent_filter=True, section_weights=False)

    assert store.search_calls[0]["query_filter"] == models.Filter(
        should=[
            models.FieldCondition(
                key="playlist_titles", match=models.MatchValue(value="Супы")
            )
        ]
    )
    assert result.debug["intent_filter"] == ["Супы"]
    assert result.debug["filtered"] is True


def test_retrieve_use_intent_filter_false_does_not_pass_filter():
    """use_intent_filter=False (the default) -> query_filter=None reaches
    store.search even for a query that would have matched an intent."""
    store = FakeStore(points=[])
    r = make_retriever(store=store, reranker=None)

    result = r.retrieve("хочу борщ", use_intent_filter=False, section_weights=False)

    assert store.search_calls[0]["query_filter"] is None
    assert result.debug["intent_filter"] == []
    assert result.debug["filtered"] is False


def test_retrieve_passes_mode_to_store():
    """The mode argument reaches store.search."""
    store = FakeStore(points=[])
    r = make_retriever(store=store, reranker=None)

    r.retrieve("запрос", mode="dense", section_weights=False)

    assert store.search_calls[0]["mode"] == "dense"


def test_retrieve_echoes_mode_in_the_result():
    """The mode argument is echoed back in the RetrievalResult."""
    store = FakeStore(points=[])
    r = make_retriever(store=store, reranker=None)

    result = r.retrieve("запрос", mode="sparse", section_weights=False)

    assert result.mode == "sparse"


# =======================================================================
# retrieve(): reading a payload that is incomplete or oddly typed
# =======================================================================
def test_retrieve_falls_back_to_point_id_when_payload_is_missing():
    """A point with no payload at all: chunk_id falls back to the point id and
    the text fields come back empty instead of raising."""
    store = FakeStore(points=[FakePoint(id="pt-42", score=0.5, payload=None)])
    r = make_retriever(store=store, reranker=None)

    result = r.retrieve("запрос", neighbor_radius=0, section_weights=False)

    passage = result.videos[0].passages[0]
    assert passage.chunk_ids == ["pt-42"]
    assert passage.title == ""
    assert passage.text == ""
    assert passage.timecode == ""


def test_retrieve_missing_numeric_payload_keys_fall_back_to_defaults():
    """A payload without chunk_index/section: chunk_index defaults to 0 and the
    section to "other", so the chunk is penalised as the first one of a video.
    """
    store = FakeStore(
        points=[FakePoint(id="p", score=1.0, payload={"video_id": "v1", "text": "мясо"})]
    )
    r = make_retriever(store=store, reranker=None)

    result = r.retrieve("запрос", neighbor_radius=0, section_weights=True)

    # "other" is absent from RETRIEVAL_SECTION_WEIGHTS -> 1.0; chunk_index 0 ->
    # the first-chunk penalty is the only multiplier left.
    assert result.videos[0].score == pytest.approx(RETRIEVAL_FIRST_CHUNK_WEIGHT)


def test_retrieve_casts_string_timestamps_from_payload_to_float():
    """start/end arriving as strings are cast to float rather than propagated
    as text into the passage."""
    payload = {"video_id": "v1", "text": "текст", "start": "60.5", "end": "70.5"}
    store = FakeStore(points=[FakePoint(id="p", score=1.0, payload=payload)])
    r = make_retriever(store=store, reranker=None)

    result = r.retrieve("запрос", neighbor_radius=0, section_weights=False)

    passage = result.videos[0].passages[0]
    assert passage.start == pytest.approx(60.5)
    assert passage.end == pytest.approx(70.5)


def test_retrieve_passes_negative_start_through_unchanged():
    """A negative start is carried through as is: this layer does no timecode
    validation, and the test pins that down so a silent clamp would show up."""
    payload = make_payload(video_id="v1", start=-5.0, end=10.0)
    store = FakeStore(points=[FakePoint(id="p", score=1.0, payload=payload)])
    r = make_retriever(store=store, reranker=None)

    result = r.retrieve("запрос", neighbor_radius=0, section_weights=False)

    assert result.videos[0].passages[0].start == pytest.approx(-5.0)


# =======================================================================
# get_retriever()
# =======================================================================
def test_get_retriever_returns_same_instance_on_repeated_calls(monkeypatch):
    """A repeated get_retriever() call returns the same object (a singleton)
    instead of building a new Retriever."""
    import src.retrieval.retriever as mod

    # The heavy constructors are replaced by light fakes: a unit test must not
    # load real HF models or connect to Qdrant.
    monkeypatch.setattr(mod, "VectorStore", lambda: object())
    monkeypatch.setattr(mod, "BGEM3Embedder", lambda: object())
    monkeypatch.setattr(mod, "try_load_reranker", lambda: None)

    first = mod.get_retriever()
    second = mod.get_retriever()

    assert first is second
