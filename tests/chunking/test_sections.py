"""Tests for src/chunking/sections.py: heuristic section labelling.

The thresholds here are deliberately picky (see TODO.md and the comment on
RETRIEVAL_SECTION_WEIGHTS in config.py: on the real ASR corpus this labelling
is nearly degenerate — 78% of chunks land in "steps" and only 0.9% in
"ingredients"). These tests pin down exactly where the boundaries sit.
"""
from __future__ import annotations

import pytest

from src.chunking.sections import (
    _scores,
    classify_segment,
    label_segments,
    smooth,
    starts_new_step,
)

# Text with neither ingredient markers, step verbs nor sequence markers.
NEUTRAL_TEXT = "просто наблюдаем за процессом на кухне"


# --- _scores ---------------------------------------------------------
def test_scores_sums_unit_and_phrase_hits_into_ingredient_score():
    """A unit-of-measure match and an ingredient-phrase match both add to the
    ingredient score."""
    assert _scores("500 г муки и соль по вкусу") == (2, 0)


def test_scores_sums_verb_and_seq_marker_hits_into_step_score():
    """A step verb and a sequence marker both add to the SAME step score."""
    assert _scores("сначала обжарьте лук") == (0, 2)


def test_scores_returns_zero_zero_for_text_without_any_marker():
    """Plain narration with no culinary marker scores zero on both axes."""
    assert _scores(NEUTRAL_TEXT) == (0, 0)


# --- classify_segment: intro/outro position gates ---------------------
@pytest.mark.parametrize(
    "position, expected",
    [
        pytest.param(0.12, "intro", id="at-threshold-is-intro"),
        pytest.param(0.13, "other", id="past-threshold-is-not-intro"),
    ],
)
def test_classify_segment_intro_position_boundary(position, expected):
    """The intro gate is position <= 0.12, so 0.12 qualifies and 0.13 does not."""
    assert classify_segment("Всем привет, дорогие друзья!", position) == expected


@pytest.mark.parametrize(
    "position, expected",
    [
        pytest.param(0.82, "outro", id="at-threshold-is-outro"),
        pytest.param(0.81, "other", id="below-threshold-is-not-outro"),
    ],
)
def test_classify_segment_outro_position_boundary(position, expected):
    """The outro gate is position >= 0.82, so 0.82 qualifies and 0.81 does not."""
    assert classify_segment("Спасибо за просмотр, ставьте лайки!", position) == expected


def test_classify_segment_intro_marker_late_in_video_is_not_intro():
    """An intro phrase said late in the video (position beyond 0.12) is not
    tagged intro, since the position gate is checked together with the marker."""
    assert classify_segment("Всем привет, дорогие друзья!", 0.5) == "other"


def test_classify_segment_outro_marker_early_in_video_is_not_outro():
    """An outro phrase said early in the video (position below 0.82) is not
    tagged outro."""
    assert classify_segment("Спасибо за просмотр, ставьте лайки!", 0.05) == "other"


# --- classify_segment: ingredients vs steps ---------------------------
def test_classify_segment_ingredients_when_score_strictly_exceeds_steps():
    """ing >= 2 and ing > step yields 'ingredients'."""
    assert classify_segment("200 г сахара и 300 г муки", 0.5) == "ingredients"


def test_classify_segment_tie_between_ingredients_and_steps_favors_steps():
    """ing == step == 2 fails the strict ing > step check, so the tie resolves
    to 'steps' instead of 'ingredients'."""
    text = "200 г сахара, 300 г муки, нарежьте и обжарьте"
    assert _scores(text.lower()) == (2, 2)
    assert classify_segment(text, 0.5) == "steps"


def test_classify_segment_steps_when_step_score_exceeds_ingredients():
    """step >= 2 and step >= ing yields 'steps'."""
    assert classify_segment("нарежьте лук и обжарьте на сковороде", 0.5) == "steps"


def test_classify_segment_single_seq_marker_without_verbs_is_steps():
    """A single sequence marker (no step verb at all) still totals step == 1
    with ing == 0, and the presence of the marker itself tips it to 'steps'."""
    assert _scores("сначала лук") == (0, 1)
    assert classify_segment("сначала лук", 0.5) == "steps"


def test_classify_segment_single_verb_without_seq_marker_is_other():
    """A lone step verb with no sequence marker falls through every branch."""
    assert classify_segment("обжарьте лук на сковороде", 0.5) == "other"


def test_classify_segment_single_ingredient_hit_is_not_enough():
    """ing == 1 misses the ing >= 2 gate, so a single quantity is 'other'."""
    assert classify_segment("200 г муки", 0.5) == "other"


def test_classify_segment_one_seq_marker_with_one_quantity_is_other():
    """The lone-verb rule demands ing == 0, not merely ing <= step: a sequence
    marker next to a single quantity scores (1, 1) and stays 'other'.

    Loosening that guard would leak stray quantities into 'steps', which on
    this corpus is already 78% of the chunks.
    """
    assert _scores("затем 200 г муки") == (1, 1)
    assert classify_segment("затем 200 г муки", 0.5) == "other"


def test_classify_segment_intro_marker_wins_over_step_verbs():
    """The positional gates are checked FIRST: a greeting that already talks
    about chopping and frying is still 'intro'.

    This is the common shape of a first chunk — the author starts using action
    verbs in the opening seconds — so if the ing/step branches ran first, the
    intro penalty in retrieval would stop recognising these chunks.
    """
    text = "всем привет, сегодня нарежем и обжарим лук"
    assert _scores(text.lower()) == (0, 2)
    assert classify_segment(text, 0.05) == "intro"


def test_classify_segment_outro_marker_wins_over_step_verbs():
    """The same precedence at the end of a video: a farewell that mentions
    cooking is 'outro', not 'steps'."""
    text = "приятного аппетита, а ещё нарежьте и обжарьте по вкусу"
    assert classify_segment(text, 0.95) == "outro"


def test_classify_segment_empty_text_is_other():
    """No text at all matches nothing and lands in 'other'."""
    assert classify_segment("", 0.5) == "other"


def test_classify_segment_is_case_insensitive():
    """Upper-cased input classifies the same as the lower-cased original."""
    lower = classify_segment("нарежьте, обжарьте и потушите", 0.5)
    upper = classify_segment("НАРЕЖЬТЕ, ОБЖАРЬТЕ И ПОТУШИТЕ", 0.5)
    assert lower == upper == "steps"


# --- smooth ------------------------------------------------------------
def test_smooth_fills_single_other_between_identical_ingredients():
    assert smooth(["ingredients", "other", "ingredients"]) == [
        "ingredients", "ingredients", "ingredients",
    ]


def test_smooth_fills_single_other_between_identical_steps():
    assert smooth(["steps", "other", "steps"]) == ["steps", "steps", "steps"]


def test_smooth_does_not_fill_other_between_identical_intro_labels():
    """Only 'ingredients'/'steps' are in the fillable set — 'intro' is not,
    even when both neighbours agree."""
    assert smooth(["intro", "other", "intro"]) == ["intro", "other", "intro"]


def test_smooth_does_not_touch_other_between_different_labels():
    assert smooth(["intro", "other", "outro"]) == ["intro", "other", "outro"]


def test_smooth_fills_gap_flanked_by_two_matching_labels_on_one_side():
    """A run of two identical labels, a single 'other' gap, and one more
    matching label is fully smoothed into a single label.

    Note this is resolved by the FIRST pass (the gap already has matching
    neighbours on both sides); the second pass in sections.py is unreachable —
    see test_smooth_leaves_a_two_cell_gap_alone below.
    """
    assert smooth(["ingredients", "ingredients", "other", "ingredients"]) == [
        "ingredients", "ingredients", "ingredients", "ingredients",
    ]


def test_smooth_leaves_a_two_cell_gap_alone():
    """Smoothing spans exactly one cell: a two-cell gap between identical
    labels is left as is.

    This is also the boundary that keeps the second pass of smooth() dead. Its
    condition is strictly stronger than the first pass's, and the first pass
    reaches the same index earlier, so it never fires — a rewrite that made it
    fire would widen smoothing to two cells and turn this test red.
    """
    assert smooth(["steps", "steps", "other", "other", "steps"]) == [
        "steps", "steps", "other", "other", "steps",
    ]


@pytest.mark.parametrize(
    "labels",
    [
        pytest.param([], id="empty"),
        pytest.param(["other"], id="single-element"),
        pytest.param(["other", "other"], id="two-elements"),
    ],
)
def test_smooth_does_not_crash_on_lists_shorter_than_three(labels):
    assert smooth(labels) == labels


def test_smooth_does_not_mutate_its_input():
    original = ["ingredients", "other", "ingredients"]
    original_copy = list(original)
    result = smooth(original)
    assert original == original_copy
    assert result is not original


# --- label_segments ------------------------------------------------------
def test_label_segments_empty_list_returns_empty_list():
    assert label_segments([]) == []


def test_label_segments_single_text_gets_position_zero():
    """With one text, position is 0/1 == 0.0, which satisfies the intro gate."""
    assert label_segments(["Всем привет, дорогие друзья!"]) == ["intro"]


def test_label_segments_last_segment_position_is_not_rounded_up_to_one():
    """Position is i/n, not i/(n-1): the last of 5 segments sits at 4/5 == 0.8,
    which is BELOW the 0.82 outro gate, so an outro phrase there is missed."""
    texts = [
        NEUTRAL_TEXT,
        NEUTRAL_TEXT,
        NEUTRAL_TEXT,
        NEUTRAL_TEXT,
        "Спасибо за просмотр, ставьте лайки!",
    ]
    result = label_segments(texts)
    assert result[-1] == "other"


def test_label_segments_result_is_smoothed():
    """A lone neutral segment sandwiched between two ingredient segments comes
    back filled in, because label_segments runs its raw output through smooth."""
    texts = [
        "200 г сахара и 300 г муки",
        NEUTRAL_TEXT,
        "400 г масла и 100 г сыра",
    ]
    assert label_segments(texts) == ["ingredients", "ingredients", "ingredients"]


# --- starts_new_step -----------------------------------------------------
def test_starts_new_step_true_when_marker_opens_the_text():
    assert starts_new_step("сначала нарежьте лук") is True


def test_starts_new_step_false_when_marker_is_mid_text():
    """starts_new_step uses re.match (anchored), not re.search, so a marker
    that is not at the very start does not count."""
    assert starts_new_step("лук сначала нарежьте") is False


def test_starts_new_step_ignores_leading_whitespace_and_case():
    assert starts_new_step("  Сначала нарежьте лук") is True


def test_starts_new_step_false_for_empty_string():
    assert starts_new_step("") is False


def test_starts_new_step_respects_word_boundaries():
    """The sequence markers are wrapped in word boundaries, so a word merely
    starting with the same letters ("затемнение" vs "затем") is not a step."""
    assert starts_new_step("затемнение экрана") is False


def test_scores_counts_step_verbs_by_stem_without_word_boundaries():
    """Pins a known imprecision rather than a bug: the step verbs are matched
    as bare stems, so "пожарная" counts as the verb "жар". A single hit is not
    enough to classify on its own, which is what keeps this survivable."""
    assert _scores("пожарная безопасность на кухне") == (0, 1)
    assert classify_segment("пожарная безопасность на кухне", 0.5) == "other"
