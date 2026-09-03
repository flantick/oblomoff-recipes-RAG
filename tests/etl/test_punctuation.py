"""Tests for src/etl/punctuation.py: the RUPunct label decoder, sentence
assembly, silero alignment and PunctuationRestorer.__init__'s backend
selection / restore_cues wiring.

torch and transformers are never really loaded here: PunctuationRestorer is
always constructed with device="cpu" and its lazy _init_rupunct/_init_silero
methods are monkeypatched away, the same pattern tests/retrieval/test_rerank.py
and tests/chunking/test_tokens.py use for their own lazily-imported libraries.
Two tests specifically target the device="cuda"/"cpu" autodetection; those
fake sys.modules["torch"] instead, and use backend="none" so no other lazy
import is reachable.
"""
from __future__ import annotations

import sys
import types

import pytest

from src.etl.punctuation import (
    PunctuationRestorer,
    Sentence,
    _align_restored,
    _assemble_sentences,
    _decode_label,
    _norm,
)
from src.etl.schemas import RawCue


# ===================================================================
# _decode_label
# ===================================================================

@pytest.mark.parametrize(
    "label, expected",
    [
        ("LOWER_O", "привет"),
        ("UPPER_O", "Привет"),
        ("UPPER_TOTAL_O", "ПРИВЕТ"),
    ],
    ids=["lower", "upper_first_letter", "upper_total_whole_word"],
)
def test_decode_label_applies_case_for_each_case_key(label, expected):
    """Every recognised case prefix (LOWER/UPPER/UPPER_TOTAL) is applied to
    the token before any punctuation suffix is appended."""
    assert _decode_label("привет", label) == expected


@pytest.mark.parametrize(
    "punct_key, suffix",
    [
        ("O", ""),
        ("PERIOD", "."),
        ("COMMA", ","),
        ("QUESTION", "?"),
        ("TIRE", " —"),
        ("DVOETOCHIE", ":"),
        ("VOSKL", "!"),
        ("PERIODCOMMA", ";"),
        ("DEFIS", "-"),
        ("MNOGOTOCHIE", "..."),
        ("QUESTIONVOSKL", "?!"),
    ],
    ids=[
        "o_empty", "period", "comma", "question", "tire_with_leading_space",
        "dvoetochie", "voskl", "periodcomma", "defis", "mnogotochie",
        "questionvoskl",
    ],
)
def test_decode_label_appends_each_punctuation_suffix(punct_key, suffix):
    """Every punctuation suffix from _PUNCT_SUFFIX is appended verbatim after
    the (lowercase) token, including the leading space TIRE carries."""
    assert _decode_label("привет", f"LOWER_{punct_key}") == "привет" + suffix


def test_decode_label_unknown_punct_suffix_after_known_case_yields_empty_suffix():
    """A case prefix that RUPunct recognises but a punctuation tag it does
    not (typo/unknown) still applies the case, with an empty suffix instead
    of crashing."""
    assert _decode_label("слово", "LOWER_NOSUCHTAG") == "слово"


def test_decode_label_unknown_case_prefix_returns_token_unchanged():
    """A label without any of the three recognised case prefixes falls
    through the loop untouched and the raw token is returned as-is."""
    assert _decode_label("слово", "RANDOM_LABEL") == "слово"


def test_decode_label_checks_upper_total_before_upper():
    """UPPER_TOTAL must be matched before UPPER, or "UPPER_TOTAL_PERIOD"
    would be parsed as case=UPPER with the unknown punct key "TOTAL_PERIOD"
    (case applied, no punctuation) instead of case=UPPER_TOTAL with
    punct=PERIOD."""
    assert _decode_label("привет", "UPPER_TOTAL_PERIOD") == "ПРИВЕТ."


# ===================================================================
# _assemble_sentences
# ===================================================================

def test_assemble_sentences_empty_input_returns_empty_list():
    assert _assemble_sentences([], [], 400) == []


def test_assemble_sentences_resets_on_sentence_final_punctuation():
    """A token ending in one of .!?… flushes the current sentence, and a new
    one starts at the following token."""
    decoded = ["Привет.", "Как", "дела?"]
    spans = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]
    result = _assemble_sentences(decoded, spans, hard_max_chars=400)
    assert result == [
        Sentence(text="Привет.", start=0.0, end=1.0),
        Sentence(text="Как дела?", start=1.0, end=3.0),
    ]


@pytest.mark.parametrize(
    "first_token, expected_texts",
    [
        pytest.param("1234567,", ["1234567, extra"], id="one_below_hard_max_no_comma_reset"),
        pytest.param("12345678,", ["12345678,", "Extra"], id="exactly_at_hard_max_comma_reset"),
    ],
)
def test_assemble_sentences_comma_reset_threshold_is_inclusive(
    first_token, expected_texts
):
    """hard_max_chars=10: with the first token's length+1 landing at exactly
    9 there is no missed-period safety flush, but at exactly 10 there is —
    the comparison is >=, so the boundary value itself already triggers it.

    The texts are compared, not just their number, so the test also pins WHERE
    the cut falls.
    """
    decoded = [first_token, "extra"]
    spans = [(0.0, 1.0), (2.0, 3.0)]
    result = _assemble_sentences(decoded, spans, hard_max_chars=10)
    assert [s.text for s in result] == expected_texts


def test_assemble_sentences_does_not_reset_at_hard_max_without_a_comma():
    """The safety flush needs BOTH the length and a trailing comma: a long
    token without one keeps accumulating.

    Dropping the comma check would turn the missed-period safeguard into a
    blind length cut, splitting sentences mid-phrase.
    """
    decoded = ["1234567890", "extra"]
    spans = [(0.0, 1.0), (2.0, 3.0)]
    result = _assemble_sentences(decoded, spans, hard_max_chars=5)
    assert [s.text for s in result] == ["1234567890 extra"]


def test_assemble_sentences_empty_tokens_are_skipped_and_do_not_set_start():
    """An empty decoded token is skipped entirely: it must not become the
    sentence's start timestamp."""
    decoded = ["", "Привет."]
    spans = [(0.0, 0.5), (2.0, 3.0)]
    result = _assemble_sentences(decoded, spans, hard_max_chars=400)
    assert result == [Sentence(text="Привет.", start=2.0, end=3.0)]


def test_assemble_sentences_removes_space_before_punctuation():
    """When the decoded token stream contains a lone punctuation token, the
    space the join() introduces before it is stripped."""
    decoded = ["привет", "."]
    spans = [(0.0, 0.0), (1.0, 1.0)]
    result = _assemble_sentences(decoded, spans, hard_max_chars=400)
    assert result == [Sentence(text="Привет.", start=0.0, end=1.0)]


def test_assemble_sentences_collapses_multiple_spaces():
    """A token carrying its own leading space, combined with join()'s
    separator, produces a double space that gets collapsed to one."""
    decoded = ["привет", " мир."]
    spans = [(0.0, 0.0), (1.0, 1.0)]
    result = _assemble_sentences(decoded, spans, hard_max_chars=400)
    assert result == [Sentence(text="Привет мир.", start=0.0, end=1.0)]


def test_assemble_sentences_rounds_start_and_end_to_two_decimals():
    decoded = ["привет."]
    spans = [(0.123456, 0.987654)]
    result = _assemble_sentences(decoded, spans, hard_max_chars=400)
    assert result[0].start == pytest.approx(0.12)
    assert result[0].end == pytest.approx(0.99)


def test_assemble_sentences_unclosed_tail_is_flushed_at_the_end():
    """No token ever ends the sentence and hard_max_chars is never reached,
    but the trailing text is still returned via the final unconditional
    flush() call after the loop."""
    decoded = ["привет", "как"]
    spans = [(0.0, 1.0), (1.0, 2.0)]
    result = _assemble_sentences(decoded, spans, hard_max_chars=400)
    assert result == [Sentence(text="Привет как", start=0.0, end=2.0)]


# ===================================================================
# _norm
# ===================================================================

@pytest.mark.parametrize(
    "text, expected",
    [
        ("Привет!", "привет"),
        ("well-known", "well-known"),
        ("word_word", "word_word"),
        ("123", "123"),
        ("", ""),
    ],
    ids=["strips_punct_and_lowers", "keeps_hyphen", "keeps_underscore", "keeps_digits", "empty"],
)
def test_norm(text, expected):
    assert _norm(text) == expected


# ===================================================================
# _align_restored
# ===================================================================

def test_align_restored_full_match_uses_restored_tokens():
    original = ["привет", "мир"]
    restored = ["Привет,", "мир."]
    assert _align_restored(original, restored) == ["Привет,", "мир."]


def test_align_restored_extra_inserted_token_is_skipped_via_lookahead():
    """silero inserted an extra token ("лишнее") that has no counterpart in
    the original: the j+1 lookahead detects the match one slot further and
    consumes both the extra token and the real one."""
    original = ["привет", "мир"]
    restored = ["лишнее", "Привет,", "мир."]
    assert _align_restored(original, restored) == ["Привет,", "мир."]


def test_align_restored_desync_falls_back_to_original_and_advances_counter():
    """When neither restored[j] nor restored[j+1] matches, the original word
    is kept — but the counter still advances by one, which lets alignment
    resume on the following words (proven here by "c"/"d" coming back from
    restored, with their own punctuation, rather than from the original)."""
    original = ["a", "b", "c", "d"]
    restored = ["A.", "X", "C,", "D!"]
    assert _align_restored(original, restored) == ["A.", "b", "C,", "D!"]


def test_align_restored_shorter_restored_list_pads_tail_with_original():
    original = ["a", "b", "c"]
    restored = ["A."]
    assert _align_restored(original, restored) == ["A.", "b", "c"]


def test_align_restored_empty_original_returns_empty_list():
    assert _align_restored([], ["x", "y"]) == []


def test_align_restored_empty_restored_returns_original_words():
    assert _align_restored(["a", "b"], []) == ["a", "b"]


# ===================================================================
# PunctuationRestorer.__init__ — backend selection
# ===================================================================

def _ok(self) -> None:
    return None


def _boom(self) -> None:
    raise RuntimeError("backend unavailable")


def _must_not_be_called(self) -> None:
    raise AssertionError("this backend must not have been initialised")


def test_init_auto_uses_rupunct_when_it_succeeds(monkeypatch):
    monkeypatch.setattr(PunctuationRestorer, "_init_rupunct", _ok)
    monkeypatch.setattr(PunctuationRestorer, "_init_silero", _must_not_be_called)
    r = PunctuationRestorer(backend="auto", device="cpu")
    assert r.backend == "rupunct"


def test_init_auto_falls_back_to_silero_when_rupunct_fails(monkeypatch):
    monkeypatch.setattr(PunctuationRestorer, "_init_rupunct", _boom)
    monkeypatch.setattr(PunctuationRestorer, "_init_silero", _ok)
    r = PunctuationRestorer(backend="auto", device="cpu")
    assert r.backend == "silero"
    assert r.requested_backend == "auto"


def test_init_auto_ends_up_none_when_both_backends_fail(monkeypatch):
    monkeypatch.setattr(PunctuationRestorer, "_init_rupunct", _boom)
    monkeypatch.setattr(PunctuationRestorer, "_init_silero", _boom)
    r = PunctuationRestorer(backend="auto", device="cpu")
    assert r.backend == "none"
    assert r.requested_backend == "auto"


def test_init_explicit_rupunct_reraises_on_failure(monkeypatch):
    """Unlike "auto", an explicit backend="rupunct" does not fall back — its
    failure must propagate to the caller."""
    monkeypatch.setattr(PunctuationRestorer, "_init_rupunct", _boom)
    with pytest.raises(RuntimeError, match="backend unavailable"):
        PunctuationRestorer(backend="rupunct", device="cpu")


def test_init_explicit_silero_reraises_on_failure(monkeypatch):
    monkeypatch.setattr(PunctuationRestorer, "_init_silero", _boom)
    with pytest.raises(RuntimeError, match="backend unavailable"):
        PunctuationRestorer(backend="silero", device="cpu")


def test_init_backend_none_never_touches_either_init(monkeypatch):
    monkeypatch.setattr(PunctuationRestorer, "_init_rupunct", _must_not_be_called)
    monkeypatch.setattr(PunctuationRestorer, "_init_silero", _must_not_be_called)
    r = PunctuationRestorer(backend="none", device="cpu")
    assert r.backend == "none"


@pytest.mark.parametrize(
    "window, stride, expected_stride",
    [
        pytest.param(150, 0, 1, id="zero_clamped_up_to_one"),
        pytest.param(150, -5, 1, id="negative_clamped_up_to_one"),
        pytest.param(150, 150, 149, id="equal_to_window_clamped_down"),
        pytest.param(150, 200, 149, id="above_window_clamped_down"),
        pytest.param(150, 110, 110, id="within_range_unchanged"),
    ],
)
def test_init_stride_is_clamped_between_one_and_window_minus_one(
    window, stride, expected_stride
):
    r = PunctuationRestorer(backend="none", device="cpu", window=window, stride=stride)
    assert r.stride == expected_stride


def test_init_requested_backend_keeps_original_value_regardless_of_outcome(monkeypatch):
    monkeypatch.setattr(PunctuationRestorer, "_init_rupunct", _ok)
    r = PunctuationRestorer(backend="auto", device="cpu")
    assert r.requested_backend == "auto"
    assert r.backend == "rupunct"


# --- device autodetection (device=None) ---------------------------------

class _FakeCuda:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


def _install_fake_torch(monkeypatch, cuda_available: bool) -> None:
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = _FakeCuda(cuda_available)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


def test_init_device_none_picks_cuda_when_available(monkeypatch):
    _install_fake_torch(monkeypatch, cuda_available=True)
    r = PunctuationRestorer(backend="none", device=None)
    assert r.device == "cuda"


def test_init_device_none_picks_cpu_when_cuda_unavailable(monkeypatch):
    _install_fake_torch(monkeypatch, cuda_available=False)
    r = PunctuationRestorer(backend="none", device=None)
    assert r.device == "cpu"


# ===================================================================
# PunctuationRestorer.restore_cues
# ===================================================================

def test_restore_cues_empty_list_returns_empty_list():
    r = PunctuationRestorer(backend="none", device="cpu")
    assert r.restore_cues([]) == []


def test_restore_cues_backend_none_passes_words_through_but_keeps_source_splits():
    """With no punctuation backend, words are neither re-cased nor
    re-punctuated — but sentence boundaries already present in the source
    text (one full stop per cue here) are still honoured, so the two cues
    become two separate sentences with their own timestamps."""
    cues = [
        RawCue(text="Привет мир.", start=0.0, duration=1.0),
        RawCue(text="Как дела?", start=5.0, duration=1.0),
    ]
    r = PunctuationRestorer(backend="none", device="cpu")
    result = r.restore_cues(cues)
    assert result == [
        Sentence(text="Привет мир.", start=0.0, end=1.0),
        Sentence(text="Как дела?", start=5.0, end=6.0),
    ]


def test_restore_cues_pre_clean_true_strips_filler_words():
    cues = [RawCue(text="ну привет", start=0.0, duration=1.0)]
    r = PunctuationRestorer(backend="none", device="cpu")
    result = r.restore_cues(cues, pre_clean=True)
    assert result == [Sentence(text="Привет", start=0.0, end=1.0)]


def test_restore_cues_pre_clean_false_keeps_filler_words():
    cues = [RawCue(text="ну привет", start=0.0, duration=1.0)]
    r = PunctuationRestorer(backend="none", device="cpu")
    result = r.restore_cues(cues, pre_clean=False)
    assert result == [Sentence(text="Ну привет", start=0.0, end=1.0)]


def test_restore_cues_skips_cues_that_become_empty_after_cleaning():
    """A cue consisting solely of a filler word is cleaned down to an empty
    string and dropped instead of contributing a stray empty span."""
    cues = [
        RawCue(text="ну", start=0.0, duration=1.0),
        RawCue(text="Привет.", start=2.0, duration=1.0),
    ]
    r = PunctuationRestorer(backend="none", device="cpu")
    result = r.restore_cues(cues, pre_clean=True)
    assert result == [Sentence(text="Привет.", start=2.0, end=3.0)]


def test_restore_cues_assigns_each_word_the_span_of_its_own_cue():
    """Words from the second cue must carry the second cue's (start, end),
    not the first cue's — the per-word span list must not slip."""
    cues = [
        RawCue(text="Первое слово.", start=0.0, duration=2.0),
        RawCue(text="Второе слово!", start=10.0, duration=3.0),
    ]
    r = PunctuationRestorer(backend="none", device="cpu")
    result = r.restore_cues(cues, pre_clean=True)
    assert result == [
        Sentence(text="Первое слово.", start=0.0, end=2.0),
        Sentence(text="Второе слово!", start=10.0, end=13.0),
    ]


# ===================================================================
# PunctuationRestorer._label_silero
# ===================================================================

def test_label_silero_collects_every_window_into_the_aligned_result():
    """A 401-word input is windowed at 400 words, and EVERY window's output
    ends up in the aligned result.

    The fake upper-cases what it is given, so the returned words have to be
    upper-cased too: that distinguishes a real alignment from simply echoing
    the input, and it fails if any window is dropped instead of appended.
    """
    calls: list[str] = []

    def fake_apply(piece: str, lan: str = "ru") -> str:
        calls.append(piece)
        return piece.upper()

    r = PunctuationRestorer(backend="none", device="cpu")
    r._silero_apply = fake_apply
    words = [f"word{i}" for i in range(401)]

    result = r._label_silero(words)

    assert result == [w.upper() for w in words]
    assert [len(c.split()) for c in calls] == [400, 1]


def test_label_silero_falls_back_to_the_original_piece_on_exception():
    """If _silero_apply raises, the untouched piece is used instead of
    letting the exception propagate out of restore_cues."""
    def failing_apply(piece: str, lan: str = "ru") -> str:
        raise RuntimeError("silero exploded")

    r = PunctuationRestorer(backend="none", device="cpu")
    r._silero_apply = failing_apply
    words = ["a", "b", "c"]

    result = r._label_silero(words)

    assert result == ["a", "b", "c"]


# ===================================================================
# restore_cues: dispatch to the backend labeller
# ===================================================================
def _labeller(marker: str):
    """A labeller that tags every word, so which one ran is visible."""
    return lambda words: [f"{marker}{w}" for w in words]


def _labeller_that_must_not_run(words):
    raise AssertionError("the other backend's labeller must not be called")


def _dispatching_restorer(backend: str) -> PunctuationRestorer:
    """A restorer wired to a chosen backend without loading anything.

    backend="none" keeps the constructor from touching either lazy init; the
    attribute is set afterwards so restore_cues takes the branch under test.
    """
    r = PunctuationRestorer(backend="none", device="cpu")
    r.backend = backend
    return r


def test_restore_cues_dispatches_to_the_rupunct_labeller():
    """backend="rupunct" routes the words through _label_rupunct, and the
    silero labeller is never touched."""
    r = _dispatching_restorer("rupunct")
    r._label_rupunct = _labeller("R:")
    r._label_silero = _labeller_that_must_not_run

    result = r.restore_cues([RawCue(text="привет мир", start=0.0, duration=1.0)])

    assert [s.text for s in result] == ["R:привет R:мир"]


def test_restore_cues_dispatches_to_the_silero_labeller():
    """backend="silero" routes the words through _label_silero, and the
    rupunct labeller is never touched."""
    r = _dispatching_restorer("silero")
    r._label_silero = _labeller("S:")
    r._label_rupunct = _labeller_that_must_not_run

    result = r.restore_cues([RawCue(text="привет мир", start=0.0, duration=1.0)])

    assert [s.text for s in result] == ["S:привет S:мир"]


def test_restore_cues_unknown_backend_passes_words_through_unlabelled():
    """Any backend name outside the two known ones falls through to the raw
    words rather than picking a labeller at random."""
    r = _dispatching_restorer("something-else")
    r._label_rupunct = _labeller_that_must_not_run
    r._label_silero = _labeller_that_must_not_run

    result = r.restore_cues([RawCue(text="привет мир", start=0.0, duration=1.0)])

    assert [s.text for s in result] == ["Привет мир"]


def test_init_device_none_falls_back_to_cpu_when_torch_is_missing(monkeypatch):
    """On a machine without torch the autodetection must not raise: the
    ImportError is caught and the device falls back to the CPU."""
    monkeypatch.setitem(sys.modules, "torch", None)  # makes `import torch` raise

    r = PunctuationRestorer(backend="none", device=None)

    assert r.device == "cpu"


# ===================================================================
# PunctuationRestorer._label_rupunct
# ===================================================================
# The real path runs a token-classification model over sliding windows and
# keeps, for every word, the label from the window where that word sat closest
# to the centre. None of that needs torch: the method only ever asks the
# tokenizer for word_ids and the model for argmaxed label ids, so a handful of
# stand-ins is enough to exercise the windowing and the centrality rule.

class _FakePreds:
    def __init__(self, preds): self._preds = preds
    def tolist(self): return list(self._preds)


class _FakeLogits:
    def __init__(self, preds): self._preds = preds
    def argmax(self, dim): return [_FakePreds(self._preds)]


class _FakeValue:
    """A stand-in tensor: the method only calls .to(device) on it."""
    def to(self, device): return self


class _FakeEncoding(dict):
    """A stand-in BatchEncoding: a mapping plus word_ids()."""
    def __init__(self, word_ids):
        super().__init__(input_ids=_FakeValue())
        self._word_ids = word_ids

    def word_ids(self, index): return self._word_ids


class _FakeRuPunct:
    """Drives _label_rupunct window by window.

    scripts holds one (word_ids, predicted_label_ids) pair per window, in the
    order the windows are visited. word_ids is given explicitly rather than
    derived, which is what keeps this a test parameter instead of a
    reimplementation of the tokenizer.
    """

    def __init__(self, scripts):
        self.scripts = list(scripts)
        self.chunks: list[list[str]] = []

    def tokenizer(self, chunk, **kw):
        self.chunks.append(list(chunk))
        return _FakeEncoding(self.scripts[len(self.chunks) - 1][0])

    def model(self, **kw):
        preds = self.scripts[len(self.chunks) - 1][1]
        return types.SimpleNamespace(logits=_FakeLogits(preds))


def _rupunct_restorer(scripts, id2label, *, window=150, stride=110):
    """A restorer whose rupunct machinery is entirely faked.

    backend="none" keeps the constructor from loading anything; the model
    plumbing is attached afterwards.
    """
    import contextlib

    r = PunctuationRestorer(backend="none", device="cpu", window=window, stride=stride)
    fake = _FakeRuPunct(scripts)
    r._torch = types.SimpleNamespace(no_grad=contextlib.nullcontext)
    r._tok = fake.tokenizer
    r._model = fake.model
    r._id2label = id2label
    return r, fake


def test_label_rupunct_labels_a_word_from_its_first_subtoken():
    """A word split into several subtokens takes the label of its FIRST one,
    and the special (None) positions around them are ignored.

    Note the `seen` guard in the source is belt-and-braces rather than load
    bearing: centrality depends only on the word index, so a later subtoken of
    the same word always ties and the strict > keeps the first one anyway.
    This test pins the outcome, not the mechanism.
    """
    r, _ = _rupunct_restorer(
        scripts=[([None, 0, 0, None], [9, 0, 1, 9])],
        id2label={0: "UPPER_O", 1: "LOWER_PERIOD"},
    )

    assert r._label_rupunct(["привет"]) == ["Привет"]


def test_label_rupunct_keeps_the_earlier_window_when_centrality_ties():
    """When two windows place a word equally far from their centres, the
    earlier window keeps it: the comparison is a strict >.

    With window=4 the offsets 1 and 3 are equidistant from the centre, so word
    3 — fourth in the first window, second in the next — is an exact tie. A >=
    here would let every later window overwrite its predecessor on ties, which
    silently turns the centrality rule into "last window wins".
    """
    period, comma, question = 0, 1, 2
    r, _ = _rupunct_restorer(
        scripts=[
            ([0, 1, 2, 3], [period] * 4),    # window (0, 4)
            ([0, 1, 2, 3], [comma] * 4),     # window (2, 6)
            ([0, 1], [question] * 2),        # window (4, 6)
        ],
        id2label={period: "LOWER_PERIOD", comma: "LOWER_COMMA", question: "LOWER_QUESTION"},
        window=4,
        stride=2,
    )

    result = r._label_rupunct(["a", "b", "c", "d", "e", "f"])

    # d ties between the first two windows and stays with the first;
    # e and f are genuinely more central in the later windows and move.
    assert result == ["a.", "b.", "c.", "d.", "e,", "f?"]


def test_label_rupunct_prefers_the_window_where_the_word_sits_most_central():
    """Overlapping windows disagree; the label from the window where the word
    is closest to the centre wins.

    Words 0-2 come from the first window, 3-4 from the second. Word 4 is the
    interesting one: it appears in both the second and the third window, and
    the second — where it is less peripheral — has to win.
    """
    period, comma, question = 0, 1, 2
    r, fake = _rupunct_restorer(
        scripts=[
            ([0, 1, 2], [period, period, period]),      # window (0, 3)
            ([0, 1, 2], [comma, comma, comma]),         # window (2, 5)
            ([0], [question]),                          # window (4, 5)
        ],
        id2label={period: "LOWER_PERIOD", comma: "LOWER_COMMA", question: "LOWER_QUESTION"},
        window=3,
        stride=2,
    )

    result = r._label_rupunct(["a", "b", "c", "d", "e"])

    assert result == ["a.", "b.", "c.", "d,", "e,"]
    assert fake.chunks == [["a", "b", "c"], ["c", "d", "e"], ["e"]]


def test_label_rupunct_falls_back_to_lower_o_for_words_no_window_labelled():
    """A word that no window produced a label for (the tokenizer truncates at
    512 subtokens) is emitted unchanged instead of crashing on int(None)."""
    r, _ = _rupunct_restorer(
        scripts=[([0], [0])],  # only the first word gets a word_id
        id2label={0: "LOWER_PERIOD"},
    )

    assert r._label_rupunct(["привет", "мир"]) == ["привет.", "мир"]


def test_label_rupunct_falls_back_to_lower_o_for_an_unknown_label_id():
    """A predicted id missing from the model's id2label map decodes as
    LOWER_O, leaving the word as it was."""
    r, _ = _rupunct_restorer(scripts=[([0], [7])], id2label={})

    assert r._label_rupunct(["привет"]) == ["привет"]
