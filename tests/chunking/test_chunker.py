"""Tests for src/chunking/chunker.py: the timecode formatter (_fmt_timecode),
the section-label aggregator (_dominant_label), the break decision
(_should_break), the overlap tail picker (_overlap_tail), and the end-to-end
chunk_transcript() on real VideoTranscript objects built by a local factory.
"""
from __future__ import annotations

import pytest

from src.chunking.chunker import (
    ChunkConfig,
    _Seg,
    _dominant_label,
    _fmt_timecode,
    _overlap_tail,
    _should_break,
    chunk_transcript,
)
from src.chunking.sections import label_segments
from src.etl.schemas import CleanSegment, VideoMeta, VideoTranscript


# --- helpers local to this file ---------------------------------------
class WordCounter:
    """A predictable token counter: 1 token = 1 word.

    The real TokenCounter counts by CHARS_PER_TOKEN, which makes it awkward to
    hit an exact token boundary in a test; here the count is trivial to work
    out by hand.
    """

    def count(self, text: str) -> int:
        return len(text.split())


def make_segment(text: str, start: float = 0.0, end: float = 1.0) -> CleanSegment:
    return CleanSegment(text=text, start=start, end=end, timecode="00:00", url="")


def make_transcript(
    texts_with_times: list[tuple[str, float, float]],
    *,
    video_id: str = "vid1",
    title: str = "Название",
    playlist_ids: list[str] | None = None,
    playlist_titles: list[str] | None = None,
) -> VideoTranscript:
    meta = VideoMeta(
        video_id=video_id,
        title=title,
        url=f"https://youtu.be/{video_id}",
        playlist_ids=playlist_ids or [],
        playlist_titles=playlist_titles or [],
    )
    segments = [make_segment(t, s, e) for t, s, e in texts_with_times]
    return VideoTranscript(
        meta=meta,
        language="ru",
        is_generated=False,
        source="ytapi",
        raw_cues_count=len(segments),
        segments=segments,
    )


def seg(text: str = "текст", start: float = 0.0, end: float = 1.0, label: str = "other", tokens: int = 1) -> _Seg:
    return _Seg(text=text, start=start, end=end, label=label, tokens=tokens)


# =======================================================================
# _fmt_timecode
# =======================================================================
@pytest.mark.parametrize(
    "seconds, expected",
    [
        pytest.param(65, "01:05", id="under_an_hour"),
        pytest.param(3600, "1:00:00", id="exactly_one_hour"),
        pytest.param(3725, "1:02:05", id="over_an_hour"),
        pytest.param(0, "00:00", id="zero"),
        pytest.param(65.9, "01:05", id="fractional_seconds_truncated"),
    ],
)
def test_fmt_timecode(seconds, expected):
    assert _fmt_timecode(seconds) == expected


# =======================================================================
# _dominant_label
# =======================================================================
def test_dominant_label_single_label_is_returned_as_is():
    assert _dominant_label(["steps"]) == "steps"


def test_dominant_label_explicit_majority_wins():
    assert _dominant_label(["steps", "steps", "other"]) == "steps"


def test_dominant_label_tie_between_two_labels_is_mixed():
    """A 1-vs-1 tie between two distinct non-'other' labels resolves to 'mixed'."""
    assert _dominant_label(["steps", "ingredients"]) == "mixed"


def test_dominant_label_all_other_falls_back_to_other():
    """When every label is 'other', the `core = [...] or labels` fallback kicks
    in and 'other' itself is returned instead of an empty-counts crash."""
    assert _dominant_label(["other", "other"]) == "other"


def test_dominant_label_other_mixed_with_one_content_label_ignores_other():
    """'other' entries are filtered out of the vote when a real label is present."""
    assert _dominant_label(["other", "steps", "other"]) == "steps"


# =======================================================================
# _should_break
# =======================================================================
def test_should_break_true_when_over_max_even_with_matching_labels():
    """cur_tokens >= cfg.max forces a break unconditionally."""
    cfg = ChunkConfig(target=10, max=10, min=0, overlap=0)
    cur_last = seg(label="steps")
    nxt = seg(text="Обжарьте лук", label="steps")
    assert _should_break(10, cur_last, nxt, cfg) is True


def test_should_break_false_below_target_even_with_a_label_change():
    """cur_tokens < cfg.target never breaks, even across a section change."""
    cfg = ChunkConfig(target=10, max=1000, min=0, overlap=0)
    cur_last = seg(label="steps")
    nxt = seg(text="500 г муки", label="ingredients")
    assert _should_break(5, cur_last, nxt, cfg) is False


def test_should_break_true_at_target_on_label_change():
    """Exactly at the target, a section change is a good place to break."""
    cfg = ChunkConfig(target=10, max=1000, min=0, overlap=0)
    cur_last = seg(label="steps")
    nxt = seg(text="500 г муки", label="ingredients")
    assert _should_break(10, cur_last, nxt, cfg) is True


def test_should_break_true_at_target_when_next_starts_new_step():
    """A new-step marker ('затем', …) is a good boundary even without a label
    change."""
    cfg = ChunkConfig(target=10, max=1000, min=0, overlap=0)
    cur_last = seg(label="steps")
    nxt = seg(text="Затем добавьте соль", label="steps")
    assert _should_break(10, cur_last, nxt, cfg) is True


def test_should_break_false_at_target_inside_an_ingredient_list():
    """A dense ingredient list is stretched past the target rather than split."""
    cfg = ChunkConfig(target=10, max=1000, min=0, overlap=0)
    cur_last = seg(label="ingredients")
    nxt = seg(text="100 г сахара", label="ingredients")
    assert _should_break(10, cur_last, nxt, cfg) is False


def test_should_break_true_at_max_even_inside_an_ingredient_list():
    """The hard limit outranks the ingredient-list exemption.

    The max check comes first for a reason: an ingredient list is stretched
    past the TARGET, but not past the MAX, or a long enumeration would grow a
    chunk beyond the embedding model's context.
    """
    cfg = ChunkConfig(target=10, max=1000, min=0, overlap=0)
    cur_last = seg(label="ingredients")
    nxt = seg(text="100 г сахара", label="ingredients")
    assert _should_break(1000, cur_last, nxt, cfg) is True


def test_should_break_true_at_target_same_non_ingredient_label():
    """Matching labels that are not 'ingredients' still break once the target
    is reached and no new-step marker leads the next segment."""
    cfg = ChunkConfig(target=10, max=1000, min=0, overlap=0)
    cur_last = seg(label="steps")
    nxt = seg(text="Обжарьте лук", label="steps")
    assert _should_break(10, cur_last, nxt, cfg) is True


# =======================================================================
# _overlap_tail
# =======================================================================
def test_overlap_tail_returns_empty_when_overlap_not_positive():
    segs = [seg(tokens=10), seg(tokens=10)]
    assert _overlap_tail(segs, 0) == []


def test_overlap_tail_never_returns_more_than_two_segments():
    """Even with a huge overlap budget, at most the last two segments are kept."""
    s1, s2, s3 = seg(text="s1", tokens=10), seg(text="s2", tokens=10), seg(text="s3", tokens=10)
    result = _overlap_tail([s1, s2, s3], 1000)
    assert result == [s2, s3]


def test_overlap_tail_stops_once_accumulated_tokens_reach_overlap():
    """A single trailing segment already meeting the overlap budget is enough:
    the loop does not reach for a second one."""
    s1, s2 = seg(text="s1", tokens=10), seg(text="s2", tokens=10)
    result = _overlap_tail([s1, s2], 5)
    assert result == [s2]


def test_overlap_tail_empty_segments_returns_empty_list():
    assert _overlap_tail([], 10) == []


# =======================================================================
# chunk_transcript
# =======================================================================
def test_chunk_transcript_empty_segments_returns_empty_list():
    vt = make_transcript([])
    assert chunk_transcript(vt, WordCounter()) == []


def test_chunk_transcript_single_segment_produces_one_chunk():
    vt = make_transcript(
        [("Всем привет сегодня будем готовить борщ", 0.0, 5.0)],
        video_id="abc123",
        title="Борщ рецепт",
        playlist_ids=["pl1"],
        playlist_titles=["Супы"],
    )
    chunks = chunk_transcript(vt, WordCounter())

    assert len(chunks) == 1
    c = chunks[0]
    assert c.chunk_id == "abc123::000"
    assert c.video_id == "abc123"
    assert c.title == "Борщ рецепт"
    assert c.chunk_index == 0
    assert c.n_chunks == 1
    assert c.playlist_ids == ["pl1"]
    assert c.playlist_titles == ["Супы"]
    assert "t=0" in c.url
    # timecode and token_len are derived, not copied: both are what retrieval
    # later shows the user and spends its budget on.
    assert c.timecode == "00:00"
    assert c.token_len == 6  # WordCounter: six words in the segment


def test_chunk_transcript_chunk_id_has_zero_padded_index_and_shared_n_chunks():
    """With a tiny target/max every one-word segment becomes its own chunk;
    the fourth one must be '...::003' and every chunk reports n_chunks=4."""
    cfg = ChunkConfig(target=1, max=1, min=0, overlap=0)
    texts = [("Раз", 0.0, 1.0), ("Два", 1.0, 2.0), ("Три", 2.0, 3.0), ("Четыре", 3.0, 4.0)]
    vt = make_transcript(texts, video_id="vid1")

    chunks = chunk_transcript(vt, WordCounter(), cfg)

    assert len(chunks) == 4
    assert chunks[3].chunk_id == "vid1::003"
    assert [c.n_chunks for c in chunks] == [4, 4, 4, 4]


def test_chunk_transcript_start_end_are_rounded_and_url_uses_integer_seconds():
    cfg = ChunkConfig(target=1000, max=1000, min=0, overlap=0)
    vt = make_transcript([("Один сегмент текста", 61.987654, 130.111111)])

    chunks = chunk_transcript(vt, WordCounter(), cfg)

    c = chunks[0]
    assert c.start == pytest.approx(61.99)
    assert c.end == pytest.approx(130.11)
    assert "t=61" in c.url


def test_chunk_transcript_char_len_is_length_of_the_joined_text():
    """Two segments merged into one group produce text joined by a single
    space, and char_len matches that exact joined string."""
    cfg = ChunkConfig(target=1000, max=1000, min=0, overlap=0)
    vt = make_transcript([("Раз", 0.0, 1.0), ("Два", 1.0, 2.0)])

    chunks = chunk_transcript(vt, WordCounter(), cfg)

    assert len(chunks) == 1
    assert chunks[0].text == "Раз Два"
    assert chunks[0].char_len == len("Раз Два")


def test_chunk_transcript_group_spans_from_first_start_to_last_end():
    """A chunk covering several segments starts at the first one and ends at
    the last — this pair is the timecode link and the fragment length the user
    sees, so both ends of the group matter."""
    cfg = ChunkConfig(target=1000, max=1000, min=0, overlap=0)
    vt = make_transcript([("Раз", 0.0, 1.0), ("Два", 5.0, 9.0)])

    chunks = chunk_transcript(vt, WordCounter(), cfg)

    assert len(chunks) == 1
    assert chunks[0].start == pytest.approx(0.0)
    assert chunks[0].end == pytest.approx(9.0)


def test_chunk_transcript_section_is_mixed_when_two_labels_tie_in_a_group():
    """The chunk's section is derived from the labels of its segments: an
    ingredient segment and a step segment in one group tie, and a tie is
    reported as 'mixed' rather than an arbitrary winner.

    section is a payload index in Qdrant and a scoring weight in retrieval, so
    it has to survive the trip through chunking intact.
    """
    ingredients = "200 г муки и 100 г сахара"
    steps = "затем нарежьте и обжарьте лук"
    # Precondition: this test speaks about chunker's aggregation, but it needs
    # the section heuristic to label these two texts as expected. Asserting it
    # here means a change to the heuristic fails on this line, not further down.
    assert label_segments([ingredients, steps]) == ["ingredients", "steps"]

    cfg = ChunkConfig(target=1000, max=1000, min=0, overlap=0)
    vt = make_transcript([(ingredients, 0.0, 5.0), (steps, 5.0, 10.0)])

    chunks = chunk_transcript(vt, WordCounter(), cfg)

    assert len(chunks) == 1
    assert chunks[0].section == "mixed"
    assert chunks[0].has_ingredients is True
    assert chunks[0].has_steps is True


@pytest.mark.parametrize(
    "texts, expect_ingredients, expect_steps",
    [
        pytest.param(
            [("500 г муки и 100 г сахара", 0.0, 1.0), ("200 мл молока и 50 г масла", 1.0, 2.0)],
            True,
            False,
            id="ingredients_only",
        ),
        pytest.param(
            [("Обжарьте лук и посолите блюдо", 0.0, 1.0), ("Потушите мясо и добавьте лук", 1.0, 2.0)],
            False,
            True,
            id="steps_only",
        ),
        pytest.param(
            [("500 г муки и 100 г сахара", 0.0, 1.0), ("Обжарьте лук и посолите блюдо", 1.0, 2.0)],
            True,
            True,
            id="mixed_ingredients_and_steps",
        ),
    ],
)
def test_chunk_transcript_has_ingredients_and_has_steps_flags(texts, expect_ingredients, expect_steps):
    """The flags reflect which section labels are present in the chunk's group
    of source segments."""
    cfg = ChunkConfig(target=1000, max=1000, min=0, overlap=0)
    vt = make_transcript(texts)

    chunks = chunk_transcript(vt, WordCounter(), cfg)

    assert len(chunks) == 1
    assert chunks[0].has_ingredients is expect_ingredients
    assert chunks[0].has_steps is expect_steps


def test_chunk_transcript_section_change_has_no_overlap():
    """When a break happens because the section label changes, the new group
    starts clean: it does not repeat the tail of the previous section."""
    cfg = ChunkConfig(target=10, max=1000, min=0, overlap=5)
    ingredient_1 = "500 г муки и 100 г сахара"
    ingredient_2 = "200 мл молока и 50 г масла"
    step = "Нарежьте лук и обжарьте на сковороде"
    # Precondition: the case needs a real section change, which comes from the
    # heuristic in sections.py. Asserting it here means a retuned heuristic
    # fails on this line instead of looking like a chunking bug below.
    assert label_segments([ingredient_1, ingredient_2, step]) == [
        "ingredients", "ingredients", "steps",
    ]
    vt = make_transcript([(ingredient_1, 0.0, 1.0), (ingredient_2, 1.0, 2.0), (step, 2.0, 3.0)])

    chunks = chunk_transcript(vt, WordCounter(), cfg)

    assert len(chunks) == 2
    assert chunks[0].text == f"{ingredient_1} {ingredient_2}"
    assert chunks[1].text == step


def test_chunk_transcript_same_section_break_carries_overlap():
    """When a break happens within one section (label unchanged), the previous
    segment is repeated at the head of the next group as overlap."""
    cfg = ChunkConfig(target=10, max=1000, min=0, overlap=3)
    step_1 = "Обжарьте лук и посолите блюдо"
    step_2 = "Потушите мясо и добавьте лук"
    step_3 = "Посыпьте зеленью и полейте соусом"
    # Precondition: the label must stay the same across all three, or the break
    # would be a section change and carry no overlap.
    assert label_segments([step_1, step_2, step_3]) == ["steps", "steps", "steps"]
    vt = make_transcript([(step_1, 0.0, 1.0), (step_2, 1.0, 2.0), (step_3, 2.0, 3.0)])

    chunks = chunk_transcript(vt, WordCounter(), cfg)

    assert len(chunks) == 2
    assert chunks[0].text == f"{step_1} {step_2}"
    # step_2 is repeated at the head of the second chunk as the overlap tail.
    assert chunks[1].text == f"{step_2} {step_3}"


def test_chunk_transcript_short_tail_is_glued_without_duplicating_overlap():
    """A trailing group shorter than cfg.min is glued to the previous group.

    The two groups already share a segment because of the overlap carried by
    the earlier break, and that shared segment must be deduplicated by
    identity (`prev_ids` in the source) rather than appended a second time
    through a naive extend.
    """
    cfg = ChunkConfig(target=10, max=1000, min=15, overlap=3)
    step_1 = "Обжарьте лук и посолите блюдо"
    step_2 = "Потушите мясо и добавьте лук"
    step_3 = "Посыпьте зеленью и полейте соусом"
    vt = make_transcript([(step_1, 0.0, 1.0), (step_2, 1.0, 2.0), (step_3, 2.0, 3.0)])

    chunks = chunk_transcript(vt, WordCounter(), cfg)

    assert len(chunks) == 1
    # A naive `groups[-2].extend(groups[-1])` without id-based dedup would
    # produce "step_1 step_2 step_2 step_3" (step_2 duplicated).
    assert chunks[0].text == f"{step_1} {step_2} {step_3}"


def test_chunk_transcript_config_changes_the_number_of_chunks():
    """The same transcript sliced with two different ChunkConfig.max values
    yields a different number of chunks, proving the config actually drives
    the split rather than being ignored."""
    texts = [("Раз", 0.0, 1.0), ("Два", 1.0, 2.0), ("Три", 2.0, 3.0), ("Четыре", 3.0, 4.0)]
    vt = make_transcript(texts)

    tight_cfg = ChunkConfig(target=1, max=1, min=0, overlap=0)
    loose_cfg = ChunkConfig(target=2, max=2, min=0, overlap=0)

    tight_chunks = chunk_transcript(vt, WordCounter(), tight_cfg)
    loose_chunks = chunk_transcript(vt, WordCounter(), loose_cfg)

    assert len(tight_chunks) == 4
    assert len(loose_chunks) == 2
