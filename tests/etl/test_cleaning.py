"""Tests for src/etl/cleaning.py: the single-cue normaliser (clean_line), the
non-recipe classifier (classify_recipe), the ad-window cutter
(strip_ad_windows), the timecode/sentence-end helpers, the two block-merging
strategies (merge_into_blocks, merge_sentences_into_blocks) and the
clean_transcript() entry point that picks between them.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

import src.etl.cleaning as cleaning
from src.etl.cleaning import (
    _ends_sentence,
    _fmt_timecode,
    classify_recipe,
    clean_line,
    clean_transcript,
    merge_into_blocks,
    merge_sentences_into_blocks,
    strip_ad_windows,
)
from src.etl.punctuation import Sentence as RealSentence
from src.etl.schemas import RawCue, VideoMeta


# --- tunables pinned for the whole file --------------------------------
@pytest.fixture(autouse=True)
def _pinned_tunables(monkeypatch):
    """Pins the window and block sizes to the values the expectations below
    were written against.

    AD_WINDOW_* live in the module and BLOCK_* come from the config; all five
    are tunable heuristics. Without pinning, retuning any of them turns this
    file red while the logic it checks is still correct — and the failure
    reads as "expected 125.0, got 130.0", which says nothing about which of
    the two actually moved.
    """
    monkeypatch.setattr(cleaning, "AD_WINDOW_BEFORE", 8.0)
    monkeypatch.setattr(cleaning, "AD_WINDOW_AFTER", 25.0)
    monkeypatch.setattr(cleaning, "BLOCK_GAP_SECONDS", 3.0)
    monkeypatch.setattr(cleaning, "BLOCK_TARGET_CHARS", 400)
    monkeypatch.setattr(cleaning, "BLOCK_MAX_CHARS", 750)


# --- helpers local to this file ---------------------------------------
def make_cue(text: str, start: float, duration: float = 1.0) -> RawCue:
    return RawCue(text=text, start=start, duration=duration)


def make_meta(title: str = "Борщ классический", description: str | None = None) -> VideoMeta:
    return VideoMeta(video_id="v1", title=title, url="https://youtu.be/v1", description=description)


@dataclass
class Sentence:
    """A punctuated sentence as produced by the restorer, NOT a RawCue: it
    carries `end` instead of `duration`."""
    text: str
    start: float
    end: float


# =======================================================================
# clean_line
# =======================================================================
@pytest.mark.parametrize(
    "text, expected",
    [
        pytest.param("[музыка] (смех) Привет", "Привет", id="brackets_removed"),
        pytest.param(">> Привет всем", "Привет всем", id="speaker_prefix_gt_gt"),
        pytest.param("- Привет всем", "Привет всем", id="speaker_prefix_dash"),
        pytest.param("ээ мм привет", "привет", id="interjections_removed"),
        pytest.param("лук ЛУК на сковороде", "лук на сковороде", id="repeat_word_collapsed_case_insensitive"),
        pytest.param("Привет ,   мир !", "Привет, мир!", id="space_before_punct_removed"),
        pytest.param(", , Привет", "Привет", id="leading_punctuation_stripped"),
        pytest.param("Привет\nмир", "Привет мир", id="newline_becomes_space"),
        pytest.param("   ну вот   ", "", id="only_junk_becomes_empty"),
    ],
)
def test_clean_line(text, expected):
    """Each individual cleaning rule produces the expected normalised text."""
    assert clean_line(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        pytest.param("ну нужно", "нужно", id="filler_word_boundary_nu"),
        pytest.param("вот вотчина", "вотчина", id="filler_word_boundary_vot"),
    ],
)
def test_clean_line_filler_word_boundary(text, expected):
    """A filler is removed only as a standalone token: 'ну' inside 'нужно' and
    'вот' inside 'вотчина' must NOT be touched (the \\b guard in _FILLER_RE)."""
    assert clean_line(text) == expected


def test_clean_line_multiword_filler_removed_entirely():
    """A multi-word filler ('как бы', 'так сказать') is removed as one unit."""
    assert clean_line("как бы так сказать надо") == "надо"


# =======================================================================
# classify_recipe
# =======================================================================
def test_classify_recipe_marker_in_title_marks_not_recipe():
    """A delivery marker in the title flips is_recipe and records which
    substring matched."""
    meta = make_meta(title="Обзор доставки суши")
    result = classify_recipe(meta)
    assert result.is_recipe is False
    assert "Обзор доставки" in result.skip_reason


def test_classify_recipe_marker_in_description_with_clean_title():
    """A delivery marker in the description alone is enough to disqualify a
    video even when the title looks like a normal recipe."""
    meta = make_meta(title="Борщ классический", description="Сегодня заказали доставку")
    result = classify_recipe(meta)
    assert result.is_recipe is False
    assert "заказали доставку" in result.skip_reason


def test_classify_recipe_no_marker_stays_recipe():
    """Without any delivery marker, is_recipe/skip_reason are left untouched."""
    meta = make_meta(title="Борщ классический")
    result = classify_recipe(meta)
    assert result.is_recipe is True
    assert result.skip_reason is None


def test_classify_recipe_none_description_does_not_raise():
    """description=None must not blow up the f-string/regex search."""
    meta = make_meta(title="Борщ классический", description=None)
    result = classify_recipe(meta)
    assert result.is_recipe is True


def test_classify_recipe_mutates_and_returns_same_object():
    """classify_recipe mutates its input in place and returns that same
    object rather than a copy."""
    meta = make_meta(title="Обзор доставки суши")
    result = classify_recipe(meta)
    assert result is meta


# =======================================================================
# strip_ad_windows
# =======================================================================
def test_strip_ad_windows_no_markers_returns_input_unchanged():
    """With no ad markers present, the original list object is returned and
    nothing is reported as removed."""
    cues = [make_cue("обычный текст", 1.0)]
    kept, removed = strip_ad_windows(cues)
    assert kept is cues
    assert removed == []


def test_strip_ad_windows_removes_window_around_single_marker():
    """A single ad marker cuts out everything within [center-8, center+25];
    unrelated cues survive."""
    cues = [
        make_cue("до рекламы", 50.0),
        make_cue("у нас промокод CODE", 100.0),
        make_cue("после рекламы", 200.0),
    ]
    kept, removed = strip_ad_windows(cues)
    assert [c.text for c in kept] == ["до рекламы", "после рекламы"]
    assert removed == [{"start": 92.0, "end": 125.0}]


@pytest.mark.parametrize(
    "start, kept_expected",
    [
        pytest.param(91.9, True, id="just_before_window_kept"),
        pytest.param(92.0, False, id="left_edge_inclusive_removed"),
        pytest.param(125.0, False, id="right_edge_inclusive_removed"),
        pytest.param(125.1, True, id="just_after_window_kept"),
    ],
)
def test_strip_ad_windows_boundaries_are_inclusive(start, kept_expected):
    """The window edges (center-8 and center+25) are cut, not just the
    interior."""
    cues = [make_cue("у нас промокод CODE", 100.0), make_cue("marker", start)]
    kept, _ = strip_ad_windows(cues)
    is_kept = any(c.text == "marker" for c in kept)
    assert is_kept is kept_expected


def test_strip_ad_windows_merges_overlapping_windows():
    """Two ad markers whose windows overlap collapse into a single removed
    span instead of two."""
    cues = [make_cue("промокод A", 100.0), make_cue("промокод B", 130.0)]
    _, removed = strip_ad_windows(cues)
    assert removed == [{"start": 92.0, "end": 155.0}]


def test_strip_ad_windows_keeps_non_overlapping_windows_separate():
    """Two ad markers far apart produce two independent removed spans, and a
    cue is dropped when it falls in ANY of them — not only when it falls in
    all of them."""
    cues = [make_cue("промокод A", 100.0), make_cue("промокод B", 200.0)]
    kept, removed = strip_ad_windows(cues)
    assert kept == []
    assert removed == [
        {"start": 92.0, "end": 125.0},
        {"start": 192.0, "end": 225.0},
    ]


def test_strip_ad_windows_rounds_bounds_to_one_decimal():
    """start/end in the removed-span report are rounded to 1 decimal place."""
    cues = [make_cue("промокод", 100.16)]
    _, removed = strip_ad_windows(cues)
    assert removed == [{"start": pytest.approx(92.2), "end": pytest.approx(125.2)}]


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
# _ends_sentence
# =======================================================================
@pytest.mark.parametrize(
    "text, expected",
    [
        pytest.param("Привет.", True, id="period"),
        pytest.param("Привет!", True, id="exclamation"),
        pytest.param("Привет?", True, id="question_mark"),
        pytest.param("Привет...", True, id="ellipsis"),
        pytest.param("Привет,", False, id="comma"),
        pytest.param("Привет", False, id="bare_letter"),
        pytest.param("Привет.   ", True, id="trailing_whitespace_ignored"),
    ],
)
def test_ends_sentence(text, expected):
    assert _ends_sentence(text) is expected


# =======================================================================
# merge_into_blocks
# =======================================================================
def test_merge_into_blocks_empty_input_returns_empty_list():
    assert merge_into_blocks([], "vid1") == []


def test_merge_into_blocks_skips_cues_that_clean_to_empty():
    """A cue that is pure junk after clean_line (e.g. only a [music] tag)
    contributes no block at all."""
    cues = [make_cue("[музыка]", 0.0)]
    assert merge_into_blocks(cues, "vid1") == []


def test_merge_into_blocks_pause_breaks_the_block():
    """A gap of BLOCK_GAP_SECONDS or more between two cues starts a new
    block even though neither size limit was hit."""
    cues = [make_cue("Привет всем", 0.0, duration=1.0), make_cue("Готовим борщ", 10.0, duration=1.0)]
    segments = merge_into_blocks(cues, "vid1")
    assert [s.text for s in segments] == ["Привет всем", "Готовим борщ"]


def test_merge_into_blocks_exceeding_max_chars_forces_break(monkeypatch):
    """Once the accumulated buffer reaches BLOCK_MAX_CHARS, the next cue
    starts a fresh block even without a pause or a finished sentence.

    BLOCK_TARGET_CHARS/BLOCK_GAP_SECONDS are pushed far out of reach here so
    that only the BLOCK_MAX_CHARS branch can fire.
    """
    monkeypatch.setattr(cleaning, "BLOCK_TARGET_CHARS", 1000)
    monkeypatch.setattr(cleaning, "BLOCK_MAX_CHARS", 5)
    monkeypatch.setattr(cleaning, "BLOCK_GAP_SECONDS", 1000.0)
    cues = [make_cue("привет", 0.0, duration=1.0), make_cue("еще", 1.0, duration=1.0)]
    segments = merge_into_blocks(cues, "vid1")
    assert [s.text for s in segments] == ["Привет", "Еще"]


def test_merge_into_blocks_flushes_on_target_reached_at_sentence_end(monkeypatch):
    """Reaching BLOCK_TARGET_CHARS on a cue that finishes a sentence flushes
    right away."""
    monkeypatch.setattr(cleaning, "BLOCK_TARGET_CHARS", 5)
    monkeypatch.setattr(cleaning, "BLOCK_MAX_CHARS", 1000)
    monkeypatch.setattr(cleaning, "BLOCK_GAP_SECONDS", 1000.0)
    cues = [make_cue("Привет.", 0.0, duration=1.0), make_cue("Мир.", 1.0, duration=1.0)]
    segments = merge_into_blocks(cues, "vid1")
    assert [s.text for s in segments] == ["Привет.", "Мир."]


def test_merge_into_blocks_keeps_accumulating_past_target_without_sentence_end(monkeypatch):
    """Reaching BLOCK_TARGET_CHARS on a cue that does NOT finish a sentence
    does not flush: the buffer keeps growing until a sentence actually ends."""
    monkeypatch.setattr(cleaning, "BLOCK_TARGET_CHARS", 5)
    monkeypatch.setattr(cleaning, "BLOCK_MAX_CHARS", 1000)
    monkeypatch.setattr(cleaning, "BLOCK_GAP_SECONDS", 1000.0)
    cues = [make_cue("Привет", 0.0, duration=1.0), make_cue("мир.", 1.0, duration=1.0)]
    segments = merge_into_blocks(cues, "vid1")
    assert [s.text for s in segments] == ["Привет мир."]


def test_merge_into_blocks_start_and_end_span_the_whole_group():
    """A block's start is the first cue's start; its end is the last cue's
    start+duration."""
    cues = [make_cue("Привет", 0.0, duration=1.0), make_cue("мир.", 1.0, duration=2.5)]
    segments = merge_into_blocks(cues, "vid1")
    assert len(segments) == 1
    assert segments[0].start == pytest.approx(0.0)
    assert segments[0].end == pytest.approx(3.5)


def test_merge_into_blocks_timecode_and_url_from_start():
    """timecode is formatted from the block start, and the url embeds the
    start truncated to whole seconds."""
    cues = [make_cue("Привет", 65.7, duration=2.0)]
    segments = merge_into_blocks(cues, "vid1")
    assert segments[0].timecode == "01:05"
    assert segments[0].url == "https://youtu.be/vid1?t=65"


def test_merge_into_blocks_rounds_start_and_end_to_two_decimals():
    """The stored timestamps are rounded, so the artifact on disk does not
    carry the full float noise of the ASR output."""
    cues = [make_cue("Привет.", 0.123456, duration=1.111111)]
    segments = merge_into_blocks(cues, "vid1")
    assert segments[0].start == pytest.approx(0.12)
    assert segments[0].end == pytest.approx(1.23)


def test_merge_into_blocks_capitalises_first_letter():
    """The first character of the merged text is upper-cased even if the
    source cue was all lower case."""
    cues = [make_cue("привет всем", 0.0, duration=1.0)]
    segments = merge_into_blocks(cues, "vid1")
    assert segments[0].text == "Привет всем"


# =======================================================================
# merge_sentences_into_blocks
# =======================================================================
def test_merge_sentences_into_blocks_empty_input_returns_empty_list():
    assert merge_sentences_into_blocks([], "vid1") == []


def test_merge_sentences_into_blocks_flushes_on_target_without_sentence_end(monkeypatch):
    """Unlike merge_into_blocks, reaching BLOCK_TARGET_CHARS flushes here
    regardless of whether the text ends a sentence."""
    monkeypatch.setattr(cleaning, "BLOCK_TARGET_CHARS", 5)
    monkeypatch.setattr(cleaning, "BLOCK_MAX_CHARS", 1000)
    monkeypatch.setattr(cleaning, "BLOCK_GAP_SECONDS", 1000.0)
    sentences = [Sentence("привет", 0.0, 1.0), Sentence("мир", 1.0, 2.0)]
    segments = merge_sentences_into_blocks(sentences, "vid1")
    assert [s.text for s in segments] == ["привет", "мир"]


def test_merge_sentences_into_blocks_does_not_capitalise_or_reformat():
    """Unlike merge_into_blocks, the text is joined as-is: no case
    normalisation and no razdel re-segmentation (the input is already
    punctuated by the restorer)."""
    sentences = [Sentence("привет", 0.0, 1.0)]
    segments = merge_sentences_into_blocks(sentences, "vid1")
    assert segments[0].text == "привет"


def test_merge_sentences_into_blocks_pause_breaks_the_block():
    sentences = [Sentence("Привет.", 0.0, 1.0), Sentence("Мир.", 10.0, 11.0)]
    segments = merge_sentences_into_blocks(sentences, "vid1")
    assert [s.text for s in segments] == ["Привет.", "Мир."]


def test_merge_sentences_into_blocks_max_chars_forces_hard_cut(monkeypatch):
    monkeypatch.setattr(cleaning, "BLOCK_TARGET_CHARS", 1000)
    monkeypatch.setattr(cleaning, "BLOCK_MAX_CHARS", 5)
    monkeypatch.setattr(cleaning, "BLOCK_GAP_SECONDS", 1000.0)
    sentences = [Sentence("Привет.", 0.0, 1.0), Sentence("Мир.", 1.0, 2.0)]
    segments = merge_sentences_into_blocks(sentences, "vid1")
    assert [s.text for s in segments] == ["Привет.", "Мир."]


# =======================================================================
# clean_transcript
# =======================================================================
def _ad_cues() -> list[RawCue]:
    """Two normal cues far enough apart to land in separate blocks (pause
    beats BLOCK_GAP_SECONDS), plus an ad cue sitting between them."""
    return [
        make_cue("Привет всем", 0.0, duration=1.0),
        make_cue("у нас промокод CODE", 100.0, duration=1.0),
        make_cue("Готовим борщ", 300.0, duration=1.0),
    ]


class _NoopRestorer:
    """backend='none' must route through merge_into_blocks just like
    restorer=None; restore_cues must not even be called."""
    backend = "none"

    def restore_cues(self, cues):
        raise AssertionError("restore_cues should not be called when backend='none'")


class _WorkingRestorer:
    """A duck-typed stand-in for the real punctuation restorer: it just
    echoes each kept cue back as an already-punctuated sentence, so the
    resulting text lets us tell which cues actually reached it.

    This one hands back the REAL punctuation.Sentence rather than the local
    stand-in: this is the only place where the two modules actually meet, so
    renaming a field there has to fail here.
    """
    backend = "rupunct"

    def restore_cues(self, cues):
        return [RealSentence(c.text.strip(), c.start, c.start + c.duration) for c in cues]


class _RestorerWithoutBackendAttribute:
    """A restorer object that never got a `backend` attribute at all.

    clean_transcript reads it with getattr(..., "none"), so such an object must
    be treated as "no punctuation" instead of raising AttributeError.
    """

    def restore_cues(self, cues):
        raise AssertionError("restore_cues should not be called without a backend")


def test_clean_transcript_no_restorer_uses_merge_into_blocks_and_none_backend():
    segments, removed, backend = clean_transcript(_ad_cues(), "vid1", restorer=None)
    assert [s.text for s in segments] == ["Привет всем", "Готовим борщ"]
    assert backend is None
    assert removed == [{"start": 92.0, "end": 125.0}]


def test_clean_transcript_backend_none_uses_merge_into_blocks_and_none_backend():
    segments, removed, backend = clean_transcript(_ad_cues(), "vid1", restorer=_NoopRestorer())
    assert [s.text for s in segments] == ["Привет всем", "Готовим борщ"]
    assert backend is None
    assert removed == [{"start": 92.0, "end": 125.0}]


def test_clean_transcript_working_restorer_uses_merge_sentences_and_reports_backend():
    segments, removed, backend = clean_transcript(_ad_cues(), "vid1", restorer=_WorkingRestorer())
    assert [s.text for s in segments] == ["Привет всем", "Готовим борщ"]
    assert backend == "rupunct"
    assert removed == [{"start": 92.0, "end": 125.0}]


def test_clean_transcript_restorer_without_backend_attribute_is_treated_as_none():
    """getattr(restorer, "backend", "none") is the guard: an object that has
    no backend attribute takes the unpunctuated path instead of raising."""
    segments, _, backend = clean_transcript(
        _ad_cues(), "vid1", restorer=_RestorerWithoutBackendAttribute()
    )
    assert [s.text for s in segments] == ["Привет всем", "Готовим борщ"]
    assert backend is None


# =======================================================================
# threshold boundaries (every comparison in this module is >=)
# =======================================================================
@pytest.mark.parametrize(
    "gap, expected_blocks",
    [
        pytest.param(2.9, 1, id="just_under_the_pause"),
        pytest.param(3.0, 2, id="exactly_at_the_pause"),
    ],
)
def test_merge_into_blocks_pause_threshold_is_inclusive(gap, expected_blocks):
    """A pause of exactly BLOCK_GAP_SECONDS already breaks the block: the
    comparison is >=, so the boundary value itself counts as a pause."""
    cues = [make_cue("Раз", 0.0, duration=1.0), make_cue("Два", 1.0 + gap, duration=1.0)]
    assert len(merge_into_blocks(cues, "vid1")) == expected_blocks


@pytest.mark.parametrize(
    "first_len, expected_blocks",
    [
        pytest.param(8, 1, id="just_under_the_hard_limit"),
        pytest.param(9, 2, id="exactly_at_the_hard_limit"),
    ],
)
def test_merge_into_blocks_max_chars_threshold_is_inclusive(
    monkeypatch, first_len, expected_blocks
):
    """The buffered length is len(line)+1; at exactly BLOCK_MAX_CHARS the next
    cue starts a new block."""
    monkeypatch.setattr(cleaning, "BLOCK_MAX_CHARS", 10)
    monkeypatch.setattr(cleaning, "BLOCK_TARGET_CHARS", 1000)
    monkeypatch.setattr(cleaning, "BLOCK_GAP_SECONDS", 1000.0)
    cues = [make_cue("х" * first_len, 0.0), make_cue("хвост", 1.0)]
    assert len(merge_into_blocks(cues, "vid1")) == expected_blocks


@pytest.mark.parametrize(
    "body_len, expected_blocks",
    [
        pytest.param(7, 1, id="just_under_the_target"),
        pytest.param(8, 2, id="exactly_at_the_target"),
    ],
)
def test_merge_into_blocks_target_threshold_is_inclusive(
    monkeypatch, body_len, expected_blocks
):
    """Reaching BLOCK_TARGET_CHARS on a sentence-final cue flushes the block;
    one character short of it does not."""
    monkeypatch.setattr(cleaning, "BLOCK_TARGET_CHARS", 10)
    monkeypatch.setattr(cleaning, "BLOCK_MAX_CHARS", 1000)
    monkeypatch.setattr(cleaning, "BLOCK_GAP_SECONDS", 1000.0)
    cues = [make_cue("х" * body_len + ".", 0.0), make_cue("хвост", 1.0)]
    assert len(merge_into_blocks(cues, "vid1")) == expected_blocks


@pytest.mark.parametrize(
    "first_len, expected_blocks",
    [
        pytest.param(8, 1, id="just_under_the_target"),
        pytest.param(9, 2, id="exactly_at_the_target"),
    ],
)
def test_merge_sentences_into_blocks_target_threshold_is_inclusive(
    monkeypatch, first_len, expected_blocks
):
    """The sentence strategy flushes at exactly BLOCK_TARGET_CHARS too, and
    unlike the cue strategy it does not wait for a sentence-final mark."""
    monkeypatch.setattr(cleaning, "BLOCK_TARGET_CHARS", 10)
    monkeypatch.setattr(cleaning, "BLOCK_MAX_CHARS", 1000)
    monkeypatch.setattr(cleaning, "BLOCK_GAP_SECONDS", 1000.0)
    sentences = [Sentence("х" * first_len, 0.0, 1.0), Sentence("хвост", 1.0, 2.0)]
    assert len(merge_sentences_into_blocks(sentences, "vid1")) == expected_blocks


# =======================================================================
# merge_sentences_into_blocks: the timecode half of the result
# =======================================================================
def test_merge_sentences_into_blocks_spans_from_first_start_to_last_end():
    """A block starts at its first sentence and ends at the LAST sentence's
    own end — this strategy reads `end` from the sentence instead of deriving
    it from start+duration the way the cue strategy does."""
    sentences = [Sentence("Привет.", 10.0, 12.0), Sentence("Мир.", 12.0, 20.0)]
    segments = merge_sentences_into_blocks(sentences, "vid1")
    assert len(segments) == 1
    assert segments[0].start == pytest.approx(10.0)
    assert segments[0].end == pytest.approx(20.0)


def test_merge_sentences_into_blocks_timecode_and_url_come_from_the_start():
    """The player link and the displayed timecode are built from the block's
    start — this is what the UI opens, so it is half of the pipeline's
    output, not a detail."""
    sentences = [Sentence("Привет.", 65.7, 70.0)]
    segments = merge_sentences_into_blocks(sentences, "vid1")
    assert segments[0].timecode == "01:05"
    assert segments[0].url == "https://youtu.be/vid1?t=65"


def test_merge_sentences_into_blocks_rounds_start_and_end_to_two_decimals():
    """Timestamps are rounded before they reach the artifact on disk."""
    sentences = [Sentence("Привет.", 0.123456, 1.111111)]
    segments = merge_sentences_into_blocks(sentences, "vid1")
    assert segments[0].start == pytest.approx(0.12)
    assert segments[0].end == pytest.approx(1.11)
