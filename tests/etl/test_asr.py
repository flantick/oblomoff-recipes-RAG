"""Tests for src/etl/asr.py: sentence splitting, device autodetection, the
faster-whisper pipeline singleton, audio download and the fetch_via_asr
orchestration.

faster_whisper and yt_dlp are real, installed packages, but neither is ever
allowed to do real work here: WhisperModel/BatchedInferencePipeline are lazily
imported inside get_pipeline() (`from faster_whisper import ...`), so a fake
module is installed into sys.modules before the call; YoutubeDL is imported
lazily inside download_audio() (`from yt_dlp import YoutubeDL`), so the real
yt_dlp.YoutubeDL attribute is monkeypatched instead (the local import picks up
whatever is bound there at call time). _MODEL/_PIPELINE are a module-level
singleton cache, reset around every test by an autouse fixture, the same
pattern tests/conftest.py uses for get_retriever()'s _DEFAULT.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from src import config
from src.etl import asr
from src.etl.schemas import RawCue
from src.etl.transcripts import RateLimited, TranscriptError


@pytest.fixture(autouse=True)
def _reset_asr_singleton():
    """get_pipeline() caches a pipeline in module globals — without a reset
    tests would start depending on execution order."""
    saved_model, saved_pipeline = asr._MODEL, asr._PIPELINE
    asr._MODEL, asr._PIPELINE = None, None
    yield
    asr._MODEL, asr._PIPELINE = saved_model, saved_pipeline


# ===================================================================
# _split_into_cues
# ===================================================================

def test_split_into_cues_single_sentence_spans_the_full_interval():
    """A text with no sentence boundary produces exactly one cue, and its
    duration is the whole (start, end) interval, not just a piece of it."""
    text = "Просто один кусок текста без разделителей"
    result = asr._split_into_cues(text, 10.0, 15.0)
    assert result == [RawCue(text=text, start=10.0, duration=5.0)]


def test_split_into_cues_multiple_sentences_durations_are_proportional_and_sum_to_span():
    """Two sentences of different length split the 10 s span in proportion
    to their character length, and the durations still add up to the whole
    interval (not the individual values only)."""
    text = "Коротко. Значительно длиннее это предложение здесь."
    result = asr._split_into_cues(text, 0.0, 10.0)
    assert [c.text for c in result] == [
        "Коротко.",
        "Значительно длиннее это предложение здесь.",
    ]
    len1, len2 = len(result[0].text), len(result[1].text)
    assert result[0].duration == pytest.approx(10.0 * len1 / (len1 + len2))
    assert result[1].duration == pytest.approx(10.0 * len2 / (len1 + len2))
    assert sum(c.duration for c in result) == pytest.approx(10.0)


def test_split_into_cues_each_cues_start_equals_the_previous_ones_end():
    text = "Раз. Два. Три."
    result = asr._split_into_cues(text, 0.0, 9.0)
    assert len(result) == 3
    for prev, nxt in zip(result, result[1:]):
        assert nxt.start == pytest.approx(prev.start + prev.duration)


def test_split_into_cues_end_before_start_clamps_duration_to_zero_single_sentence():
    """A Whisper segment with end < start (bad timestamps) must not yield a
    negative duration: max(end - start, 0.0) floors it at zero."""
    text = "Одно предложение"
    result = asr._split_into_cues(text, 10.0, 5.0)
    assert result == [RawCue(text=text, start=10.0, duration=0.0)]


def test_split_into_cues_end_before_start_clamps_every_sentence_to_zero():
    """The same clamp applies once the text is split into several sentences:
    every resulting cue gets a zero duration and starts exactly at `start`,
    none of them going negative."""
    text = "Раз. Два."
    result = asr._split_into_cues(text, 10.0, 5.0)
    assert [c.duration for c in result] == [0.0, 0.0]
    assert [c.start for c in result] == [10.0, 10.0]


def test_split_into_cues_strips_whitespace_around_each_sentence():
    """Whisper pads its segments, and the split leaves that padding on the
    pieces — each one is stripped before it becomes a cue.

    This is the only input shape that tells the strip apart from its absence:
    everywhere else the text is already clean.
    """
    cues = asr._split_into_cues("  Первое.  Второе.  ", 0.0, 10.0)

    assert [c.text for c in cues] == ["Первое.", "Второе."]


def test_split_into_cues_period_before_lowercase_word_does_not_split():
    """A period followed by a lowercase word (as in an abbreviation like
    "т.д.") is not a sentence boundary: the split regex only fires before an
    uppercase letter/quote/dash, so the whole phrase stays one cue."""
    text = "Смешайте муку и т.д. дальше по рецепту."
    result = asr._split_into_cues(text, 0.0, 4.0)
    assert len(result) == 1
    assert result[0].text == text


@pytest.mark.parametrize(
    "text, expected_texts",
    [
        pytest.param("Первое. Второе.", ["Первое.", "Второе."], id="capital_letter"),
        pytest.param('Первое. "Второе".', ["Первое.", '"Второе".'], id="opening_quote"),
        pytest.param("Первое. — Второе.", ["Первое.", "— Второе."], id="opening_dash"),
    ],
)
def test_split_into_cues_splits_on_end_mark_space_and_sentence_start_marker(
    text, expected_texts
):
    """The boundary is end-mark + space + (capital letter/quote/dash) — each
    of the three accepted sentence-start markers triggers a split."""
    result = asr._split_into_cues(text, 0.0, 2.0)
    assert [c.text for c in result] == expected_texts


# ===================================================================
# _resolve_device
# ===================================================================

class _ExplodingTorch(types.ModuleType):
    """A fake torch module that fails loudly if it is ever touched."""

    @property
    def cuda(self):
        raise AssertionError("torch must not be probed when ASR_DEVICE is set explicitly")


def test_resolve_device_explicit_device_is_used_without_touching_torch(monkeypatch):
    monkeypatch.setattr(config, "ASR_DEVICE", "mps")
    monkeypatch.setattr(config, "ASR_COMPUTE_TYPE", None)
    monkeypatch.setitem(sys.modules, "torch", _ExplodingTorch("torch"))
    assert asr._resolve_device() == ("mps", "int8")


def test_resolve_device_autodetects_cuda_when_available(monkeypatch):
    monkeypatch.setattr(config, "ASR_DEVICE", None)
    monkeypatch.setattr(config, "ASR_COMPUTE_TYPE", None)
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert asr._resolve_device() == ("cuda", "float16")


def test_resolve_device_autodetects_cpu_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(config, "ASR_DEVICE", None)
    monkeypatch.setattr(config, "ASR_COMPUTE_TYPE", None)
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert asr._resolve_device() == ("cpu", "int8")


def test_resolve_device_explicit_compute_type_overrides_cuda_autodetect(monkeypatch):
    monkeypatch.setattr(config, "ASR_DEVICE", None)
    monkeypatch.setattr(config, "ASR_COMPUTE_TYPE", "int8_float16")
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert asr._resolve_device() == ("cuda", "int8_float16")


def test_resolve_device_explicit_compute_type_overrides_cpu_autodetect(monkeypatch):
    monkeypatch.setattr(config, "ASR_DEVICE", None)
    monkeypatch.setattr(config, "ASR_COMPUTE_TYPE", "float32")
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert asr._resolve_device() == ("cpu", "float32")


def test_resolve_device_falls_back_to_cpu_when_torch_is_missing(monkeypatch):
    """On a machine without torch, `import torch` raises inside the try
    block, and the except clause falls back to cpu instead of propagating."""
    monkeypatch.setattr(config, "ASR_DEVICE", None)
    monkeypatch.setattr(config, "ASR_COMPUTE_TYPE", None)
    monkeypatch.setitem(sys.modules, "torch", None)  # makes `import torch` raise
    assert asr._resolve_device() == ("cpu", "int8")


# ===================================================================
# get_pipeline
# ===================================================================

def _install_fake_faster_whisper(monkeypatch):
    """Installs a fake faster_whisper module and returns the list that
    records every WhisperModel construction (video_id, device, compute)."""
    calls: list[tuple[str, str, str]] = []

    class FakeWhisperModel:
        def __init__(self, model_name, device, compute_type):
            calls.append((model_name, device, compute_type))
            self.model_name = model_name

    class FakeBatchedInferencePipeline:
        def __init__(self, model):
            self.model = model

    fake_module = types.ModuleType("faster_whisper")
    fake_module.WhisperModel = FakeWhisperModel
    fake_module.BatchedInferencePipeline = FakeBatchedInferencePipeline
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)
    return calls


def test_get_pipeline_creates_the_model_once_and_caches_it(monkeypatch):
    calls = _install_fake_faster_whisper(monkeypatch)
    monkeypatch.setattr(config, "ASR_MODEL", "tiny")
    monkeypatch.setattr(config, "ASR_DEVICE", "cpu")
    monkeypatch.setattr(config, "ASR_COMPUTE_TYPE", "int8")

    first = asr.get_pipeline()
    second = asr.get_pipeline()

    assert first is second
    assert len(calls) == 1


def test_get_pipeline_forwards_model_name_device_and_compute_type(monkeypatch):
    calls = _install_fake_faster_whisper(monkeypatch)
    monkeypatch.setattr(config, "ASR_MODEL", "large-v3-turbo")
    monkeypatch.setattr(config, "ASR_DEVICE", "cpu")
    monkeypatch.setattr(config, "ASR_COMPUTE_TYPE", "int8")

    pipeline = asr.get_pipeline()

    assert calls == [("large-v3-turbo", "cpu", "int8")]
    # The batched pipeline has to wrap the model that was just built; wrapping
    # anything else would transcribe with a model nobody configured.
    assert pipeline.model is asr._MODEL


# ===================================================================
# transcribe_audio
# ===================================================================

class _FakeSegment:
    def __init__(self, text: str, start: float, end: float) -> None:
        self.text = text
        self.start = start
        self.end = end


class _FakeTranscribePipeline:
    """Stands in for the object get_pipeline() returns."""

    def __init__(self, segments: list) -> None:
        self._segments = segments
        self.calls: list[tuple[str, dict]] = []

    def transcribe(self, path, **kwargs):
        self.calls.append((path, kwargs))
        return self._segments, {"language": "ru"}


def test_transcribe_audio_forwards_the_expected_transcribe_kwargs(monkeypatch, tmp_path):
    pipeline = _FakeTranscribePipeline([])
    monkeypatch.setattr(asr, "get_pipeline", lambda: pipeline)
    monkeypatch.setattr(config, "ASR_LANGUAGE", "ru")
    monkeypatch.setattr(config, "ASR_BATCH_SIZE", 8)
    path = tmp_path / "audio.mp3"

    asr.transcribe_audio(path)

    called_path, kwargs = pipeline.calls[0]
    assert called_path == str(path)
    assert kwargs == {
        "language": "ru",
        "task": "transcribe",
        "batch_size": 8,
        "vad_filter": True,
        "word_timestamps": False,
    }


def test_transcribe_audio_skips_segments_whose_text_is_blank(monkeypatch, tmp_path):
    segments = [
        _FakeSegment(text="   ", start=0.0, end=1.0),
        _FakeSegment(text="Привет.", start=1.0, end=2.0),
    ]
    pipeline = _FakeTranscribePipeline(segments)
    monkeypatch.setattr(asr, "get_pipeline", lambda: pipeline)

    result = asr.transcribe_audio(tmp_path / "audio.wav")

    assert [c.text for c in result] == ["Привет."]


def test_transcribe_audio_splits_each_segment_text_into_sentence_cues(monkeypatch, tmp_path):
    """Each segment's text is run through _split_into_cues rather than kept
    as one cue per segment: a two-sentence segment yields two cues."""
    segments = [_FakeSegment(text="Привет. Пока.", start=0.0, end=2.0)]
    pipeline = _FakeTranscribePipeline(segments)
    monkeypatch.setattr(asr, "get_pipeline", lambda: pipeline)

    result = asr.transcribe_audio(tmp_path / "audio.wav")

    assert [c.text for c in result] == ["Привет.", "Пока."]
    assert result[0].start == pytest.approx(0.0)
    assert result[1].start + result[1].duration == pytest.approx(2.0)


def test_transcribe_audio_no_segments_returns_empty_list(monkeypatch, tmp_path):
    pipeline = _FakeTranscribePipeline([])
    monkeypatch.setattr(asr, "get_pipeline", lambda: pipeline)

    assert asr.transcribe_audio(tmp_path / "audio.wav") == []


# ===================================================================
# download_audio
# ===================================================================

def _install_fake_youtubedl(monkeypatch, on_download):
    """Replaces yt_dlp.YoutubeDL with a fake whose download() runs
    on_download(opts, urls); download_audio's local `from yt_dlp import
    YoutubeDL` picks up whatever is bound to yt_dlp.YoutubeDL at call time."""
    import yt_dlp

    class FakeYoutubeDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def download(self, urls):
            on_download(self.opts, urls)

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYoutubeDL)


def test_download_audio_raises_rate_limited_on_http_429(tmp_path, monkeypatch):
    from yt_dlp.utils import DownloadError

    def on_download(opts, urls):
        raise DownloadError("ERROR: unable to download: HTTP Error 429: Too Many Requests")

    _install_fake_youtubedl(monkeypatch, on_download)

    with pytest.raises(RateLimited):
        asr.download_audio("vid1", tmp_path)


def test_download_audio_raises_transcript_error_on_other_download_errors(tmp_path, monkeypatch):
    from yt_dlp.utils import DownloadError

    def on_download(opts, urls):
        raise DownloadError("ERROR: network is unreachable")

    _install_fake_youtubedl(monkeypatch, on_download)

    with pytest.raises(TranscriptError) as exc_info:
        asr.download_audio("vid2", tmp_path)
    assert type(exc_info.value) is TranscriptError


def test_download_audio_raises_file_not_found_when_yt_dlp_leaves_no_file(tmp_path, monkeypatch):
    _install_fake_youtubedl(monkeypatch, on_download=lambda opts, urls: None)

    with pytest.raises(FileNotFoundError):
        asr.download_audio("vid3", tmp_path)


def test_download_audio_picks_the_largest_of_several_files(tmp_path, monkeypatch):
    def on_download(opts, urls):
        (tmp_path / "vid4.f140.m4a").write_bytes(b"x" * 10)
        (tmp_path / "vid4.f251.webm").write_bytes(b"x" * 1000)

    _install_fake_youtubedl(monkeypatch, on_download)

    result = asr.download_audio("vid4", tmp_path)

    assert result == tmp_path / "vid4.f251.webm"


def test_download_audio_forwards_asr_audio_format_into_the_yt_dlp_options(tmp_path, monkeypatch):
    """Without this, the auto-dubbed English track (first in the format
    list) would be downloaded instead of the original Russian audio."""
    monkeypatch.setattr(config, "ASR_AUDIO_FORMAT", "custom-format-string")
    captured: dict = {}

    def on_download(opts, urls):
        captured["format"] = opts["format"]
        (tmp_path / "vid5.mp3").write_bytes(b"x")

    _install_fake_youtubedl(monkeypatch, on_download)

    asr.download_audio("vid5", tmp_path)

    assert captured["format"] == "custom-format-string"


def test_download_audio_targets_the_right_video_and_destination(tmp_path, monkeypatch):
    """The watch URL is built from the video id, and the output template puts
    the file in the directory the caller asked for.

    Neither is observable from the return value — the tests place the file
    themselves — so both are asserted on what actually reached yt-dlp.
    """
    captured: dict = {}

    def on_download(opts, urls):
        captured["opts"] = opts
        captured["urls"] = urls
        (tmp_path / "vid5.mp3").write_bytes(b"x")

    _install_fake_youtubedl(monkeypatch, on_download)

    asr.download_audio("vid5", tmp_path)

    assert captured["urls"] == ["https://www.youtube.com/watch?v=vid5"]
    assert Path(captured["opts"]["outtmpl"]).parent == tmp_path


# ===================================================================
# fetch_via_asr
# ===================================================================

def test_fetch_via_asr_uses_the_configured_dir_without_a_temp_directory(tmp_path, monkeypatch):
    audio_dir = tmp_path / "configured"
    monkeypatch.setattr(config, "ASR_AUDIO_DIR", audio_dir)
    monkeypatch.setattr(config, "ASR_KEEP_AUDIO", True)
    monkeypatch.setattr(config, "ASR_LANGUAGE", "ru")

    def _forbidden_mkdtemp(*a, **kw):
        raise AssertionError("tempfile.mkdtemp must not run when ASR_AUDIO_DIR is set")

    monkeypatch.setattr(asr.tempfile, "mkdtemp", _forbidden_mkdtemp)

    def fake_download(video_id, dest_dir):
        assert dest_dir == audio_dir
        p = dest_dir / f"{video_id}.mp3"
        p.write_bytes(b"data")
        return p

    monkeypatch.setattr(asr, "download_audio", fake_download)
    monkeypatch.setattr(
        asr, "transcribe_audio", lambda path: [RawCue(text="Привет", start=0.0, duration=1.0)]
    )

    cues, lang, is_generated = asr.fetch_via_asr("vid1")

    assert audio_dir.exists()
    assert lang == "ru"
    assert is_generated is True
    assert [c.text for c in cues] == ["Привет"]


def test_fetch_via_asr_creates_and_removes_a_temp_dir_when_no_dir_is_configured(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ASR_AUDIO_DIR", None)
    monkeypatch.setattr(config, "ASR_KEEP_AUDIO", False)

    captured: dict = {}

    def fake_download(video_id, dest_dir):
        captured["dir"] = dest_dir
        assert dest_dir.exists()
        p = dest_dir / f"{video_id}.mp3"
        p.write_bytes(b"data")
        return p

    monkeypatch.setattr(asr, "download_audio", fake_download)
    monkeypatch.setattr(
        asr, "transcribe_audio", lambda path: [RawCue(text="ok", start=0.0, duration=1.0)]
    )

    asr.fetch_via_asr("vid2")

    assert not captured["dir"].exists()


@pytest.mark.parametrize(
    "keep_audio, should_survive",
    [
        pytest.param(True, True, id="keep_audio_true_keeps_the_file"),
        pytest.param(False, False, id="keep_audio_false_deletes_the_file"),
    ],
)
def test_fetch_via_asr_respects_the_keep_audio_flag(tmp_path, monkeypatch, keep_audio, should_survive):
    audio_dir = tmp_path / "kept"
    monkeypatch.setattr(config, "ASR_AUDIO_DIR", audio_dir)
    monkeypatch.setattr(config, "ASR_KEEP_AUDIO", keep_audio)

    audio_path_holder: dict = {}

    def fake_download(video_id, dest_dir):
        p = dest_dir / f"{video_id}.mp3"
        p.write_bytes(b"data")
        audio_path_holder["path"] = p
        return p

    monkeypatch.setattr(asr, "download_audio", fake_download)
    monkeypatch.setattr(
        asr, "transcribe_audio", lambda path: [RawCue(text="ok", start=0.0, duration=1.0)]
    )

    asr.fetch_via_asr("vid3")

    assert audio_path_holder["path"].exists() is should_survive


def test_fetch_via_asr_raises_transcript_error_when_transcription_is_empty(tmp_path, monkeypatch):
    audio_dir = tmp_path / "empty"
    monkeypatch.setattr(config, "ASR_AUDIO_DIR", audio_dir)
    monkeypatch.setattr(config, "ASR_KEEP_AUDIO", True)

    def fake_download(video_id, dest_dir):
        p = dest_dir / f"{video_id}.mp3"
        p.write_bytes(b"data")
        return p

    monkeypatch.setattr(asr, "download_audio", fake_download)
    monkeypatch.setattr(asr, "transcribe_audio", lambda path: [])

    with pytest.raises(TranscriptError):
        asr.fetch_via_asr("vid4")


def test_fetch_via_asr_removes_the_temp_dir_even_when_transcription_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ASR_AUDIO_DIR", None)

    captured: dict = {}

    def fake_download(video_id, dest_dir):
        captured["dir"] = dest_dir
        p = dest_dir / f"{video_id}.mp3"
        p.write_bytes(b"data")
        return p

    def fake_transcribe(path):
        raise RuntimeError("boom")

    monkeypatch.setattr(asr, "download_audio", fake_download)
    monkeypatch.setattr(asr, "transcribe_audio", fake_transcribe)

    with pytest.raises(RuntimeError, match="boom"):
        asr.fetch_via_asr("vid5")

    assert not captured["dir"].exists()
