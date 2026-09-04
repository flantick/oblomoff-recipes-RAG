"""Tests for src/etl/pipeline.py: collect_videos() and process_video().

main() is mostly a CLI wrapper (argparse, tqdm-driven loop, real sleeps) and is
not covered as such. Three of its decisions are policy rather than plumbing and
ARE covered at the bottom of this file, by calling main(argv) directly with the
two orchestrated functions faked: the consecutive-rate-limit abort, its reset on
a success, and --source asr implying --punct none.

Per the project's I/O table, `fetch_playlist_entries`, `fetch_video_meta` and
`get_transcript` are replaced with fakes on the pipeline module's own symbols
(the functions are imported by name, there is no constructor to inject a fake
through). `clean_transcript` is not literal I/O, but process_video has no seam
to pass it in either, and the task list explicitly calls for patching it to
check that `restorer` is threaded through — so it gets the same treatment.
`classify_recipe`, in contrast, is pure and already covered by its own test
module, so the non-recipe tests drive it for real with a title containing an
actual delivery marker rather than faking its output.
"""
from __future__ import annotations

import json

import pytest

import src.etl.pipeline as pipeline
from src.etl.cleaning import classify_recipe
from src.etl.schemas import CleanSegment, RawCue, VideoMeta, VideoTranscript
from src.etl.transcripts import RateLimited, TranscriptError


# --- fakes -------------------------------------------------------------
def make_meta(**kw) -> VideoMeta:
    defaults = dict(
        video_id="v1",
        title="Борщ классический",
        url="https://www.youtube.com/watch?v=v1",
        playlist_ids=["PL1"],
        playlist_titles=["Супы"],
    )
    defaults.update(kw)
    return VideoMeta(**defaults)


def make_fetch_playlist_entries(mapping: dict[str, list[dict]]):
    """Stands in for src.etl.playlists.fetch_playlist_entries."""

    def _fetch(url: str) -> list[dict]:
        return list(mapping.get(url, []))

    return _fetch


def make_get_transcript(cues, lang="ru", is_generated=True, used_source="ytapi",
                        seen: list | None = None):
    """Stands in for src.etl.transcripts.get_transcript on the happy path.

    `seen` collects the (video_id, source) pairs so a test can check which
    backend was actually asked for.
    """

    def _get(video_id: str, source: str):
        if seen is not None:
            seen.append((video_id, source))
        return list(cues), lang, is_generated, used_source

    return _get


def make_raising_get_transcript(exc: Exception):
    def _get(video_id: str, source: str):
        raise exc

    return _get


def fail_get_transcript(video_id: str, source: str):
    raise AssertionError("get_transcript should not have been called")


def make_fetch_video_meta(returned: VideoMeta):
    """Stands in for src.etl.playlists.fetch_video_meta; returns a fresh copy
    on every call so pipeline's in-place mutation of playlist fields never
    leaks between calls/tests."""

    def _fetch(video_id: str, fallback_title: str = ""):
        return returned.model_copy(deep=True)

    return _fetch


class FakeCleanTranscript:
    """Stands in for src.etl.cleaning.clean_transcript.

    Records every call (cues, video_id, restorer) so tests can check what the
    orchestrator forwarded, e.g. the restorer object.
    """

    def __init__(self, segments: list[CleanSegment], removed: list[dict], punct_backend):
        self.segments = segments
        self.removed = removed
        self.punct_backend = punct_backend
        self.calls: list[tuple] = []

    def __call__(self, cues, video_id, restorer=None):
        self.calls.append((cues, video_id, restorer))
        return list(self.segments), list(self.removed), self.punct_backend


# --- shared fixtures -----------------------------------------------------
@pytest.fixture(autouse=True)
def _fast_sleep(monkeypatch):
    """collect_videos sleeps ETL_SLEEP_SECONDS between playlists; the default
    (1.0s) would trip tests/etl/conftest.py's guarded-sleep fixture."""
    monkeypatch.setattr(pipeline, "ETL_SLEEP_SECONDS", 0)


@pytest.fixture
def data_dirs(tmp_path, monkeypatch):
    """Redirects DATA_RAW_DIR/DATA_PROCESSED_DIR into tmp_path so process_video
    never touches the real data/ tree."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    processed_dir.mkdir()
    monkeypatch.setattr(pipeline, "DATA_RAW_DIR", raw_dir)
    monkeypatch.setattr(pipeline, "DATA_PROCESSED_DIR", processed_dir)
    return {"raw": raw_dir, "processed": processed_dir}


# =========================================================================
# collect_videos
# =========================================================================
def test_collect_videos_single_playlist_builds_videometa(monkeypatch):
    """Each entry becomes a VideoMeta keyed by video_id, with the playlist's
    id/title recorded as a one-element list and the watch URL built from the
    id."""
    mapping = {
        "url1": [
            {"video_id": "v1", "title": "Борщ", "playlist_id": "PL1", "playlist_title": "Супы"},
            {"video_id": "v2", "title": "Щи", "playlist_id": "PL1", "playlist_title": "Супы"},
        ]
    }
    monkeypatch.setattr(pipeline, "fetch_playlist_entries", make_fetch_playlist_entries(mapping))

    result = pipeline.collect_videos(["url1"], limit=None)

    assert set(result) == {"v1", "v2"}
    assert result["v1"] == VideoMeta(
        video_id="v1",
        title="Борщ",
        url="https://www.youtube.com/watch?v=v1",
        playlist_ids=["PL1"],
        playlist_titles=["Супы"],
    )


def test_collect_videos_same_video_in_two_playlists_accumulates_playlist_lists(monkeypatch):
    """A video listed in two overlapping playlists produces one VideoMeta
    entry whose playlist_ids/playlist_titles have accumulated both, while the
    title stays the one first seen."""
    mapping = {
        "url1": [{"video_id": "v1", "title": "Борщ (плейлист супов)",
                   "playlist_id": "PL1", "playlist_title": "Супы"}],
        "url2": [{"video_id": "v1", "title": "Борщ (плейлист горячего)",
                   "playlist_id": "PL2", "playlist_title": "Основные блюда"}],
    }
    monkeypatch.setattr(pipeline, "fetch_playlist_entries", make_fetch_playlist_entries(mapping))

    result = pipeline.collect_videos(["url1", "url2"], limit=None)

    assert len(result) == 1
    assert result["v1"].playlist_ids == ["PL1", "PL2"]
    assert result["v1"].playlist_titles == ["Супы", "Основные блюда"]
    assert result["v1"].title == "Борщ (плейлист супов)"


def test_collect_videos_limit_truncates_each_playlist_independently(monkeypatch):
    """limit slices the entries of EVERY playlist before merging, not the
    combined result — so with limit=2 across two 3-video playlists, 4 (not 2)
    videos survive."""
    mapping = {
        "url1": [
            {"video_id": "a1", "title": "A1", "playlist_id": "PL1", "playlist_title": "A"},
            {"video_id": "a2", "title": "A2", "playlist_id": "PL1", "playlist_title": "A"},
            {"video_id": "a3", "title": "A3", "playlist_id": "PL1", "playlist_title": "A"},
        ],
        "url2": [
            {"video_id": "b1", "title": "B1", "playlist_id": "PL2", "playlist_title": "B"},
            {"video_id": "b2", "title": "B2", "playlist_id": "PL2", "playlist_title": "B"},
            {"video_id": "b3", "title": "B3", "playlist_id": "PL2", "playlist_title": "B"},
        ],
    }
    monkeypatch.setattr(pipeline, "fetch_playlist_entries", make_fetch_playlist_entries(mapping))

    result = pipeline.collect_videos(["url1", "url2"], limit=2)

    assert set(result) == {"a1", "a2", "b1", "b2"}


def test_collect_videos_limit_none_keeps_all_entries(monkeypatch):
    """limit=None takes the "no slicing" branch — every entry survives."""
    mapping = {
        "url1": [
            {"video_id": "a1", "title": "A1", "playlist_id": "PL1", "playlist_title": "A"},
            {"video_id": "a2", "title": "A2", "playlist_id": "PL1", "playlist_title": "A"},
            {"video_id": "a3", "title": "A3", "playlist_id": "PL1", "playlist_title": "A"},
        ],
    }
    monkeypatch.setattr(pipeline, "fetch_playlist_entries", make_fetch_playlist_entries(mapping))

    result = pipeline.collect_videos(["url1"], limit=None)

    assert set(result) == {"a1", "a2", "a3"}


def test_collect_videos_empty_playlist_list_returns_empty_dict():
    """No playlists at all means the loop body never runs — {} comes back
    without needing fetch_playlist_entries at all."""
    assert pipeline.collect_videos([], limit=None) == {}


def test_collect_videos_playlist_with_no_entries_adds_nothing(monkeypatch):
    """A playlist that yields [] (e.g. it's fully dead) contributes no
    videos, and other playlists are unaffected."""
    monkeypatch.setattr(pipeline, "fetch_playlist_entries", make_fetch_playlist_entries({}))

    result = pipeline.collect_videos(["url1"], limit=None)

    assert result == {}


# =========================================================================
# process_video
# =========================================================================
def test_process_video_skip_exists_does_not_fetch_transcript(data_dirs, monkeypatch):
    """When the output file already exists and overwrite=False, process_video
    returns skip_exists immediately and never calls get_transcript."""
    (data_dirs["processed"] / "v1.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(pipeline, "get_transcript", fail_get_transcript)
    meta = make_meta()

    result = pipeline.process_video(
        meta, source="ytapi", full_meta=False, drop_non_recipe=False, overwrite=False,
    )

    assert result == {"video_id": "v1", "status": "skip_exists"}


def test_process_video_overwrite_true_reprocesses_existing_output(data_dirs, monkeypatch):
    """overwrite=True bypasses the skip_exists short-circuit and rewrites the
    existing output file with fresh content."""
    out_path = data_dirs["processed"] / "v1.json"
    out_path.write_text('{"stale": true}', encoding="utf-8")
    cues = [RawCue(text="Раз", start=0.0, duration=1.0)]
    segments = [CleanSegment(text="Раз.", start=0.0, end=1.0, timecode="00:00",
                              url="https://youtu.be/v1?t=0")]
    monkeypatch.setattr(pipeline, "get_transcript", make_get_transcript(cues))
    monkeypatch.setattr(pipeline, "clean_transcript", FakeCleanTranscript(segments, [], None))
    meta = make_meta()

    result = pipeline.process_video(
        meta, source="ytapi", full_meta=False, drop_non_recipe=False, overwrite=True,
    )

    assert result["status"] == "ok"
    assert "stale" not in out_path.read_text(encoding="utf-8")


def test_process_video_full_meta_keeps_original_playlist_fields(data_dirs, monkeypatch):
    """full_meta=True fetches richer metadata via fetch_video_meta, but the
    playlist_ids/playlist_titles accumulated by collect_videos on the ORIGINAL
    meta must survive onto the fetched one (pipeline.py:85-86) — otherwise the
    video would lose its playlist membership."""
    original = make_meta(playlist_ids=["PL1", "PL2"], playlist_titles=["Супы", "Основное"])
    fetched = VideoMeta(
        video_id="v1",
        title="Борщ (полные метаданные)",
        url="https://www.youtube.com/watch?v=v1",
        channel="oblomoffood",
    )
    monkeypatch.setattr(pipeline, "fetch_video_meta", make_fetch_video_meta(fetched))
    cues = [RawCue(text="Раз", start=0.0, duration=1.0)]
    monkeypatch.setattr(pipeline, "get_transcript", make_get_transcript(cues))
    monkeypatch.setattr(pipeline, "clean_transcript", FakeCleanTranscript([], [], None))

    pipeline.process_video(
        original, source="ytapi", full_meta=True, drop_non_recipe=False, overwrite=False,
    )

    saved = VideoTranscript.model_validate_json(
        (data_dirs["processed"] / "v1.json").read_text(encoding="utf-8")
    )
    assert saved.meta.title == "Борщ (полные метаданные)"
    assert saved.meta.playlist_ids == ["PL1", "PL2"]
    assert saved.meta.playlist_titles == ["Супы", "Основное"]


def test_process_video_drop_non_recipe_writes_skip_file_without_transcript(data_dirs, monkeypatch):
    """A video classified as a non-recipe (delivery review) with
    drop_non_recipe=True is written to <id>.skip.json, reports
    skip_non_recipe with the real skip_reason, never calls get_transcript,
    and does NOT create the normal output file."""
    meta = make_meta(title="Большой обзор доставки роллов")
    expected_reason = classify_recipe(meta.model_copy(deep=True)).skip_reason
    assert expected_reason is not None  # sanity: the marker really matched
    monkeypatch.setattr(pipeline, "get_transcript", fail_get_transcript)

    result = pipeline.process_video(
        meta, source="ytapi", full_meta=False, drop_non_recipe=True, overwrite=False,
    )

    assert result == {"video_id": "v1", "status": "skip_non_recipe", "reason": expected_reason}
    skip_payload = json.loads((data_dirs["processed"] / "v1.skip.json").read_text(encoding="utf-8"))
    assert skip_payload["is_recipe"] is False
    assert skip_payload["skip_reason"] == expected_reason
    assert not (data_dirs["processed"] / "v1.json").exists()


def test_process_video_non_recipe_without_drop_flag_processes_normally(data_dirs, monkeypatch):
    """The same delivery-review video with drop_non_recipe=False is fetched
    and processed like any other; is_recipe=False is merely reported."""
    meta = make_meta(title="Большой обзор доставки роллов")
    cues = [RawCue(text="Раз", start=0.0, duration=1.0)]
    monkeypatch.setattr(pipeline, "get_transcript", make_get_transcript(cues))
    monkeypatch.setattr(pipeline, "clean_transcript", FakeCleanTranscript([], [], None))

    result = pipeline.process_video(
        meta, source="ytapi", full_meta=False, drop_non_recipe=False, overwrite=False,
    )

    assert result["status"] == "ok"
    assert result["is_recipe"] is False


def test_process_video_rate_limited_reports_status_without_writing_files(data_dirs, monkeypatch):
    """A RateLimited from get_transcript short-circuits into status
    rate_limited with the exception's message, and no raw/processed file is
    ever written for this video."""
    monkeypatch.setattr(
        pipeline, "get_transcript", make_raising_get_transcript(RateLimited("v1: HTTP 429"))
    )
    meta = make_meta()

    result = pipeline.process_video(
        meta, source="ytapi", full_meta=False, drop_non_recipe=False, overwrite=False,
    )

    assert result == {"video_id": "v1", "status": "rate_limited", "error": "v1: HTTP 429"}
    assert not (data_dirs["processed"] / "v1.json").exists()
    assert not (data_dirs["raw"] / "v1.json").exists()


def test_process_video_transcript_error_reports_no_transcript_status(data_dirs, monkeypatch):
    """A plain TranscriptError (not the RateLimited subclass) maps to
    no_transcript rather than rate_limited."""
    monkeypatch.setattr(
        pipeline, "get_transcript", make_raising_get_transcript(TranscriptError("v1: пусто"))
    )
    meta = make_meta()

    result = pipeline.process_video(
        meta, source="ytapi", full_meta=False, drop_non_recipe=False, overwrite=False,
    )

    assert result == {"video_id": "v1", "status": "no_transcript", "error": "v1: пусто"}


def test_process_video_unexpected_exception_reports_error_status_with_repr(data_dirs, monkeypatch):
    """Any exception other than RateLimited/TranscriptError is swallowed
    (pipeline.py:100) so one bad video does not kill the whole run; the repr
    of the exception is stored, not just its message."""
    monkeypatch.setattr(
        pipeline, "get_transcript", make_raising_get_transcript(ValueError("boom"))
    )
    meta = make_meta()

    result = pipeline.process_video(
        meta, source="ytapi", full_meta=False, drop_non_recipe=False, overwrite=False,
    )

    assert result == {"video_id": "v1", "status": "error", "error": "ValueError('boom')"}


def test_process_video_success_writes_raw_and_processed_files(data_dirs, monkeypatch):
    """The happy path writes both the raw cues file and the processed
    VideoTranscript file, and returns a summary dict with the counts pulled
    from get_transcript/clean_transcript."""
    cues = [
        RawCue(text="Раз", start=0.0, duration=1.0),
        RawCue(text="Два", start=1.0, duration=1.0),
    ]
    segments = [CleanSegment(text="Раз два.", start=0.0, end=2.0, timecode="00:00",
                              url="https://youtu.be/v1?t=0")]
    removed = [{"start": 5.0, "end": 10.0}]
    monkeypatch.setattr(pipeline, "get_transcript",
                         make_get_transcript(cues, lang="ru", is_generated=True, used_source="ytapi"))
    monkeypatch.setattr(pipeline, "clean_transcript", FakeCleanTranscript(segments, removed, "rupunct"))
    meta = make_meta()

    result = pipeline.process_video(
        meta, source="ytapi", full_meta=False, drop_non_recipe=False, overwrite=False,
    )

    assert result == {
        "video_id": "v1",
        "status": "ok",
        "is_recipe": True,
        "segments": 1,
        "raw_cues": 2,
        "ad_spans_removed": 1,
        "source": "ytapi",
        "punct": "rupunct",
    }

    raw_payload = json.loads((data_dirs["raw"] / "v1.json").read_text(encoding="utf-8"))
    assert raw_payload == {
        "video_id": "v1",
        "language": "ru",
        "cues": [c.model_dump() for c in cues],
    }

    saved = VideoTranscript.model_validate_json(
        (data_dirs["processed"] / "v1.json").read_text(encoding="utf-8")
    )
    assert saved.raw_cues_count == 2
    assert saved.removed_ad_spans == removed
    assert saved.segments == segments
    assert saved.source == "ytapi"
    assert saved.punctuation_backend == "rupunct"


def test_process_video_passes_restorer_through_to_clean_transcript(data_dirs, monkeypatch):
    """The `restorer` argument process_video receives is forwarded verbatim
    to clean_transcript rather than dropped or replaced."""
    sentinel = object()
    cues = [RawCue(text="Раз", start=0.0, duration=1.0)]
    monkeypatch.setattr(pipeline, "get_transcript", make_get_transcript(cues))
    fake_clean = FakeCleanTranscript([], [], None)
    monkeypatch.setattr(pipeline, "clean_transcript", fake_clean)
    meta = make_meta()

    pipeline.process_video(
        meta, source="ytapi", full_meta=False, drop_non_recipe=False, overwrite=False,
        restorer=sentinel,
    )

    assert len(fake_clean.calls) == 1
    _, _, received_restorer = fake_clean.calls[0]
    assert received_restorer is sentinel


def test_process_video_forwards_the_requested_transcript_backend(data_dirs, monkeypatch):
    """The --source choice reaches get_transcript.

    It selects the whole ETL backend — the local Whisper run, the transcript
    API or yt-dlp — and nothing else in the returned summary reflects which
    one was asked for, only which one answered.
    """
    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(
        pipeline, "get_transcript",
        make_get_transcript(
            [RawCue(text="Привет", start=0.0, duration=1.0)], used_source="asr", seen=seen
        ),
    )

    pipeline.process_video(
        make_meta(video_id="v1"), source="asr", full_meta=False,
        drop_non_recipe=False, overwrite=False,
    )

    assert seen == [("v1", "asr")]


# =======================================================================
# main(): the run policies that are not argparse
# =======================================================================
# main() is skipped as a CLI wrapper, but three of its decisions are real
# policy rather than plumbing, and all three are reachable without faking
# argparse — main takes argv directly.

def test_main_aborts_after_consecutive_rate_limited_videos(data_dirs, monkeypatch):
    """A run that keeps hitting rate limits stops early and exits 3.

    Without the abort the pipeline would walk the remaining hundreds of videos
    at one request each, deepening the block it already ran into.
    """
    monkeypatch.setattr(
        pipeline, "collect_videos",
        lambda playlists, limit: {f"v{i}": make_meta(video_id=f"v{i}") for i in range(10)},
    )
    processed: list[str] = []

    def always_rate_limited(meta, **kw):
        processed.append(meta.video_id)
        return {"video_id": meta.video_id, "status": "rate_limited", "error": "429"}

    monkeypatch.setattr(pipeline, "process_video", always_rate_limited)

    code = pipeline.main(["--sleep", "0", "--abort-after-rate-limits", "3", "--punct", "none"])

    assert code == 3
    assert len(processed) == 3


def test_main_rate_limit_counter_resets_after_a_success(data_dirs, monkeypatch):
    """The counter tracks CONSECUTIVE failures: one success in between clears
    it, so an intermittent block does not abort a healthy run."""
    metas = {f"v{i}": make_meta(video_id=f"v{i}") for i in range(5)}
    monkeypatch.setattr(pipeline, "collect_videos", lambda playlists, limit: metas)
    statuses = iter(["rate_limited", "rate_limited", "ok", "rate_limited", "ok"])

    def scripted(meta, **kw):
        return {"video_id": meta.video_id, "status": next(statuses)}

    monkeypatch.setattr(pipeline, "process_video", scripted)

    code = pipeline.main(["--sleep", "0", "--abort-after-rate-limits", "3", "--punct", "none"])

    assert code == 0


def test_main_asr_source_turns_punctuation_off_instead_of_loading_a_model(data_dirs, monkeypatch):
    """--source asr implies --punct none: Whisper already returns punctuated
    text, so restoring it again wastes time and degrades the result.

    Observable as the restorer that reaches process_video — and, more to the
    point, as PunctuationRestorer never being imported at all.
    """
    import src.etl.punctuation as punctuation_mod

    def _must_not_be_built(*args, **kwargs):
        raise AssertionError(
            "PunctuationRestorer was constructed for --source asr — this would "
            "download RUPunct from HuggingFace in a unit test"
        )

    # main() imports the restorer lazily, so patching the module attribute is
    # what the local import picks up. Without this a regression here does not
    # merely fail: it fails after a minute of model downloads.
    monkeypatch.setattr(punctuation_mod, "PunctuationRestorer", _must_not_be_built)
    monkeypatch.setattr(
        pipeline, "collect_videos", lambda playlists, limit: {"v1": make_meta(video_id="v1")}
    )
    captured: dict = {}

    def capture(meta, **kw):
        captured.update(kw)
        return {"video_id": meta.video_id, "status": "ok"}

    monkeypatch.setattr(pipeline, "process_video", capture)

    code = pipeline.main(["--sleep", "0", "--source", "asr"])

    assert code == 0
    assert captured["restorer"] is None
    assert captured["source"] == "asr"
