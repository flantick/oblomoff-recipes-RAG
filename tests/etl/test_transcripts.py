"""Tests for src/etl/transcripts.py: the json3/vtt subtitle parsers, the
file-selection logic in _load_subs, the youtube-transcript-api and yt-dlp
backends (fetch_via_ytapi/fetch_via_ytdlp), the get_transcript routing
between them, and the proxy-config helper.

Nothing here touches the network: youtube_transcript_api and yt_dlp are real
installed packages, but every call that would talk to YouTube is replaced —
YouTubeTranscriptApi/_ytapi_inner/_ytdlp_once are monkeypatched on the
src.etl.transcripts module object (the names the functions under test look up
as bare globals), and src.etl.transcripts.time.sleep is patched so the HTTP
429 backoff in fetch_via_ytdlp never actually sleeps.

PROXY and SUB_LANGS are read once, at import time, into module-level names in
src.etl.transcripts (`from src.config import PROXY, SUB_LANGS, ...`). Setting
the underlying environment variables in a test would therefore do nothing to
the already-bound names; every test that needs a particular PROXY/SUB_LANGS
value monkeypatches the attribute directly on the transcripts module.
"""
from __future__ import annotations

import json

import pytest
from yt_dlp.utils import DownloadError
from youtube_transcript_api import FetchedTranscript, FetchedTranscriptSnippet
from youtube_transcript_api.proxies import GenericProxyConfig, WebshareProxyConfig

import src.etl.transcripts as transcripts
from src.etl.schemas import RawCue
from src.etl.transcripts import (
    RateLimited,
    TranscriptError,
    _load_subs,
    _parse_json3,
    _parse_vtt,
    fetch_via_ytapi,
    fetch_via_ytdlp,
    get_transcript,
)

NoTranscriptFound = transcripts.NoTranscriptFound
IpBlocked = transcripts.IpBlocked
RequestBlocked = transcripts.RequestBlocked
TranscriptsDisabled = transcripts.TranscriptsDisabled
VideoUnavailable = transcripts.VideoUnavailable


# ===================================================================
# _parse_json3
# ===================================================================

def _write_json3(tmp_path, data: dict, name: str = "sub.json3"):
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_parse_json3_skips_events_without_segs(tmp_path):
    """An event with no "segs" key at all is dropped, not treated as empty."""
    path = _write_json3(tmp_path, {"events": [
        {"tStartMs": 0},
        {"tStartMs": 1000, "segs": [{"utf8": "Привет"}]},
    ]})
    assert _parse_json3(path) == [RawCue(text="Привет", start=1.0, duration=0.0)]


def test_parse_json3_skips_events_without_tstartms(tmp_path):
    """An event missing "tStartMs" cannot be placed on the timeline and is
    dropped even though it has text."""
    path = _write_json3(tmp_path, {"events": [
        {"segs": [{"utf8": "нет времени"}]},
        {"tStartMs": 2000, "segs": [{"utf8": "есть время"}]},
    ]})
    assert _parse_json3(path) == [RawCue(text="есть время", start=2.0, duration=0.0)]


def test_parse_json3_merges_segments_with_embedded_newline_into_one_cue():
    """A "\\n" segment inside an event is a line break within the SAME
    caption, not a new caption: it must not split the event into two cues."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "sub.json3"
        path.write_text(json.dumps({"events": [
            {"tStartMs": 0, "dDurationMs": 500,
             "segs": [{"utf8": "Hello"}, {"utf8": "\n"}, {"utf8": "World"}]},
        ]}), encoding="utf-8")
        result = _parse_json3(path)
    assert result == [RawCue(text="Hello\nWorld", start=0.0, duration=0.5)]


def test_parse_json3_drops_event_whose_joined_text_is_only_a_newline(tmp_path):
    """An event whose only content is a "\\n" segment joins/strips down to an
    empty string and is dropped instead of producing a blank cue."""
    path = _write_json3(tmp_path, {"events": [
        {"tStartMs": 0, "segs": [{"utf8": "\n"}]},
    ]})
    assert _parse_json3(path) == []


def test_parse_json3_converts_milliseconds_to_seconds(tmp_path):
    path = _write_json3(tmp_path, {"events": [
        {"tStartMs": 1500, "dDurationMs": 250, "segs": [{"utf8": "Hi"}]},
    ]})
    cues = _parse_json3(path)
    assert cues[0].start == pytest.approx(1.5)
    assert cues[0].duration == pytest.approx(0.25)


def test_parse_json3_missing_dduration_defaults_to_zero(tmp_path):
    path = _write_json3(tmp_path, {"events": [
        {"tStartMs": 2000, "segs": [{"utf8": "Hi"}]},
    ]})
    cues = _parse_json3(path)
    assert cues[0].duration == pytest.approx(0.0)


def test_parse_json3_empty_events_list_returns_empty_list(tmp_path):
    path = _write_json3(tmp_path, {"events": []})
    assert _parse_json3(path) == []


# ===================================================================
# _parse_vtt
# ===================================================================

def _write_vtt(tmp_path, content: str, name: str = "sub.vtt"):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_parse_vtt_parses_both_dot_and_comma_fractional_timecodes(tmp_path):
    content = (
        "WEBVTT\n\n"
        "00:00:01.000 --> 00:00:02.500\n"
        "Hello\n\n"
        "00:00:03,000 --> 00:00:04,250\n"
        "World\n"
    )
    path = _write_vtt(tmp_path, content)
    cues = _parse_vtt(path)
    assert cues == [
        RawCue(text="Hello", start=1.0, duration=1.5),
        RawCue(text="World", start=3.0, duration=1.25),
    ]


def test_parse_vtt_strips_html_tags_from_text(tmp_path):
    content = (
        "00:00:00.000 --> 00:00:01.000\n"
        "<i>Hello</i> <b>world</b>\n"
    )
    path = _write_vtt(tmp_path, content)
    cues = _parse_vtt(path)
    assert cues[0].text == "Hello world"


def test_parse_vtt_skips_webvtt_header_and_bare_cue_numbers(tmp_path):
    content = (
        "WEBVTT\n\n"
        "1\n"
        "00:00:00.000 --> 00:00:01.000\n"
        "Text one\n"
    )
    path = _write_vtt(tmp_path, content)
    cues = _parse_vtt(path)
    assert cues == [RawCue(text="Text one", start=0.0, duration=1.0)]


def test_parse_vtt_joins_multiline_cue_with_a_space(tmp_path):
    content = (
        "00:00:00.000 --> 00:00:01.000\n"
        "Line one\n"
        "Line two\n"
    )
    path = _write_vtt(tmp_path, content)
    cues = _parse_vtt(path)
    assert cues[0].text == "Line one Line two"


def test_parse_vtt_flushes_the_final_block_after_the_loop_ends(tmp_path):
    """The first cue is flushed mid-loop when the second timecode line
    arrives; the second cue has no following timecode, so it only reaches the
    result via the unconditional flush() call after the for-loop."""
    content = (
        "00:00:00.000 --> 00:00:01.000\n"
        "First\n\n"
        "00:00:02.000 --> 00:00:03.000\n"
        "Second\n"
    )
    path = _write_vtt(tmp_path, content)
    cues = _parse_vtt(path)
    assert [c.text for c in cues] == ["First", "Second"]
    assert cues[1].start == pytest.approx(2.0)
    assert cues[1].duration == pytest.approx(1.0)


# ===================================================================
# _load_subs
# ===================================================================

def test_load_subs_prefers_json3_over_srv3_and_vtt(tmp_path):
    (tmp_path / "vid1.json3").write_text(json.dumps({"events": []}), encoding="utf-8")
    (tmp_path / "vid1.srv3").write_text(json.dumps({"events": []}), encoding="utf-8")
    (tmp_path / "vid1.vtt").write_text("WEBVTT\n", encoding="utf-8")
    _cues, chosen = _load_subs(tmp_path, "vid1")
    assert chosen.name == "vid1.json3"


def test_load_subs_prefers_srv3_over_vtt_when_no_json3(tmp_path):
    (tmp_path / "vid1.srv3").write_text(json.dumps({"events": []}), encoding="utf-8")
    (tmp_path / "vid1.vtt").write_text("WEBVTT\n", encoding="utf-8")
    _cues, chosen = _load_subs(tmp_path, "vid1")
    assert chosen.name == "vid1.srv3"


def test_load_subs_ties_on_extension_broken_by_sub_langs_order(tmp_path, monkeypatch):
    """Both files are .vtt, so the extension order cannot decide: the one
    whose name matches the earlier SUB_LANGS entry wins."""
    monkeypatch.setattr(transcripts, "SUB_LANGS", ["ru", "en"])
    (tmp_path / "vid1.en.vtt").write_text("WEBVTT\n", encoding="utf-8")
    (tmp_path / "vid1.ru.vtt").write_text("WEBVTT\n", encoding="utf-8")
    _cues, chosen = _load_subs(tmp_path, "vid1")
    assert chosen.name == "vid1.ru.vtt"


def test_load_subs_vtt_extension_is_parsed_with_the_vtt_parser(tmp_path):
    """If a .vtt file were routed to the json3 parser, json.loads would blow
    up on this non-JSON content; getting a correctly parsed cue back proves
    _parse_vtt was used."""
    (tmp_path / "vid1.vtt").write_text(
        "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nHello\n", encoding="utf-8",
    )
    cues, chosen = _load_subs(tmp_path, "vid1")
    assert chosen.suffix == ".vtt"
    assert cues == [RawCue(text="Hello", start=1.0, duration=1.0)]


def test_load_subs_json3_extension_is_parsed_with_the_json3_parser(tmp_path):
    """If a .json3 file were routed to the vtt parser, the regex would find
    no timecodes and no cues would come back; getting the JSON-derived cue
    back proves _parse_json3 was used."""
    (tmp_path / "vid1.json3").write_text(json.dumps({"events": [
        {"tStartMs": 500, "dDurationMs": 500, "segs": [{"utf8": "Hi"}]},
    ]}), encoding="utf-8")
    cues, chosen = _load_subs(tmp_path, "vid1")
    assert chosen.suffix == ".json3"
    assert cues == [RawCue(text="Hi", start=0.5, duration=0.5)]


def test_load_subs_raises_transcript_error_when_no_matching_files_exist(tmp_path):
    (tmp_path / "vid1.description").write_text("nope", encoding="utf-8")
    with pytest.raises(TranscriptError):
        _load_subs(tmp_path, "vid1")


# ===================================================================
# fetch_via_ytapi — exception mapping around _ytapi_inner
# ===================================================================

@pytest.mark.parametrize(
    "make_exc, expected_type",
    [
        pytest.param(lambda: IpBlocked("vid1"), RateLimited, id="ip_blocked_to_rate_limited"),
        pytest.param(lambda: RequestBlocked("vid1"), RateLimited, id="request_blocked_to_rate_limited"),
        pytest.param(lambda: TranscriptsDisabled("vid1"), TranscriptError, id="transcripts_disabled_to_transcript_error"),
        pytest.param(lambda: VideoUnavailable("vid1"), TranscriptError, id="video_unavailable_to_transcript_error"),
    ],
)
def test_fetch_via_ytapi_maps_backend_exceptions(monkeypatch, make_exc, expected_type):
    original = make_exc()

    def _raise(video_id):
        raise original

    monkeypatch.setattr(transcripts, "_ytapi_inner", _raise)
    with pytest.raises(expected_type) as excinfo:
        fetch_via_ytapi("vid1")
    assert excinfo.value.__cause__ is original


def test_fetch_via_ytapi_returns_ytapi_inner_result_on_success(monkeypatch):
    cues = [RawCue(text="Hi", start=0.0, duration=1.0)]

    def _ok(video_id):
        return cues, "ru", False

    monkeypatch.setattr(transcripts, "_ytapi_inner", _ok)
    assert fetch_via_ytapi("vid1") == (cues, "ru", False)


# ===================================================================
# _ytapi_inner — transcript selection and cue assembly
# ===================================================================

class _FakeTList:
    """Stands in for youtube_transcript_api's TranscriptList."""

    def __init__(self, *, manual=None, generated=None, entries=()):
        self._manual = manual
        self._generated = generated
        self._entries = list(entries)

    def find_manually_created_transcript(self, langs):
        if self._manual is None:
            raise NoTranscriptFound("vid1", langs, self)
        return self._manual

    def find_generated_transcript(self, langs):
        if self._generated is None:
            raise NoTranscriptFound("vid1", langs, self)
        return self._generated

    def __iter__(self):
        return iter(self._entries)


class _FakeTranscriptEntry:
    """Stands in for youtube_transcript_api's Transcript."""

    def __init__(self, *, language_code="ru", is_generated=False, snippets=(),
                 is_translatable=False, translation=None):
        self.language_code = language_code
        self.is_generated = is_generated
        self.is_translatable = is_translatable
        self._snippets = list(snippets)
        self._translation = translation

    def fetch(self):
        return FetchedTranscript(
            snippets=list(self._snippets), video_id="vid1", language="Russian",
            language_code=self.language_code, is_generated=self.is_generated,
        )

    def translate(self, lang):
        return self._translation


def _fake_ytapi(tlist):
    class _Api:
        def __init__(self, *, proxy_config=None):
            self.proxy_config = proxy_config

        def list(self, video_id):
            return tlist

    return _Api


def _wire_ytapi(monkeypatch, tlist):
    monkeypatch.setattr(transcripts, "YouTubeTranscriptApi", _fake_ytapi(tlist))
    monkeypatch.setattr(transcripts, "_proxy_config", lambda: None)


def test_ytapi_inner_prefers_manually_created_transcript(monkeypatch):
    manual = _FakeTranscriptEntry(
        language_code="ru", is_generated=False,
        snippets=[FetchedTranscriptSnippet(text="Ручной", start=0.0, duration=1.0)],
    )
    generated = _FakeTranscriptEntry(
        language_code="ru", is_generated=True,
        snippets=[FetchedTranscriptSnippet(text="Авто", start=0.0, duration=1.0)],
    )
    tlist = _FakeTList(manual=manual, generated=generated)
    _wire_ytapi(monkeypatch, tlist)

    cues, lang, is_generated = fetch_via_ytapi("vid1")
    assert cues == [RawCue(text="Ручной", start=0.0, duration=1.0)]
    assert lang == "ru"
    assert is_generated is False


def test_ytapi_inner_falls_back_to_generated_when_no_manual_transcript(monkeypatch):
    generated = _FakeTranscriptEntry(
        language_code="ru", is_generated=True,
        snippets=[FetchedTranscriptSnippet(text="Авто", start=0.0, duration=1.0)],
    )
    tlist = _FakeTList(manual=None, generated=generated)
    _wire_ytapi(monkeypatch, tlist)

    cues, lang, is_generated = fetch_via_ytapi("vid1")
    assert cues == [RawCue(text="Авто", start=0.0, duration=1.0)]
    assert is_generated is True


def test_ytapi_inner_falls_back_to_translating_the_first_translatable_transcript(monkeypatch):
    translated = _FakeTranscriptEntry(
        language_code="ru", is_generated=True,
        snippets=[FetchedTranscriptSnippet(text="Переведено", start=0.0, duration=1.0)],
    )
    not_translatable = _FakeTranscriptEntry(is_translatable=False)
    translatable = _FakeTranscriptEntry(is_translatable=True, translation=translated)
    tlist = _FakeTList(manual=None, generated=None, entries=[not_translatable, translatable])
    _wire_ytapi(monkeypatch, tlist)

    cues, lang, is_generated = fetch_via_ytapi("vid1")
    assert cues == [RawCue(text="Переведено", start=0.0, duration=1.0)]


def test_ytapi_inner_raises_when_nothing_is_translatable_either(monkeypatch):
    """No manual, no generated, and no translatable transcript in the list at
    all: transcript stays None and a TranscriptError is raised."""
    tlist = _FakeTList(manual=None, generated=None, entries=[_FakeTranscriptEntry(is_translatable=False)])
    _wire_ytapi(monkeypatch, tlist)

    with pytest.raises(TranscriptError, match="нет подходящего транскрипта"):
        fetch_via_ytapi("vid1")


def test_ytapi_inner_drops_blank_snippets_but_keeps_raw_text_of_the_rest(monkeypatch):
    """A whitespace-only snippet is filtered out; a kept snippet's text is
    stored as-is (no stripping) in the resulting RawCue."""
    manual = _FakeTranscriptEntry(snippets=[
        FetchedTranscriptSnippet(text="   ", start=0.0, duration=1.0),
        FetchedTranscriptSnippet(text=" Привет ", start=1.0, duration=1.0),
    ])
    tlist = _FakeTList(manual=manual)
    _wire_ytapi(monkeypatch, tlist)

    cues, _lang, _gen = fetch_via_ytapi("vid1")
    assert cues == [RawCue(text=" Привет ", start=1.0, duration=1.0)]


def test_ytapi_inner_raises_when_all_snippets_are_blank(monkeypatch):
    manual = _FakeTranscriptEntry(snippets=[FetchedTranscriptSnippet(text="  ", start=0.0, duration=1.0)])
    tlist = _FakeTList(manual=manual)
    _wire_ytapi(monkeypatch, tlist)

    with pytest.raises(TranscriptError, match="пустой транскрипт"):
        fetch_via_ytapi("vid1")


# ===================================================================
# fetch_via_ytdlp — HTTP 429 backoff
# ===================================================================

class _SequencedYtdlpOnce:
    """Replays a fixed script of results/exceptions, one per call."""

    def __init__(self, script: list):
        self._script = list(script)
        self.calls = 0

    def __call__(self, video_id):
        item = self._script[self.calls]
        self.calls += 1
        if isinstance(item, BaseException):
            raise item
        return item


class _SleepSpy:
    def __init__(self):
        self.calls: list[float] = []

    def __call__(self, seconds):
        self.calls.append(seconds)


def _wire_ytdlp(monkeypatch, script, retries_429=2):
    once = _SequencedYtdlpOnce(script)
    sleep_spy = _SleepSpy()
    monkeypatch.setattr(transcripts, "_ytdlp_once", once)
    monkeypatch.setattr(transcripts, "YTDLP_RETRIES_429", retries_429)
    monkeypatch.setattr(transcripts.time, "sleep", sleep_spy)
    return once, sleep_spy


def test_fetch_via_ytdlp_returns_on_first_try_without_sleeping(monkeypatch):
    ok = ([RawCue(text="Hi", start=0.0, duration=1.0)], "ru", True)
    _once, sleep_spy = _wire_ytdlp(monkeypatch, [ok])

    result = fetch_via_ytdlp("vid1")

    assert result == ok
    assert sleep_spy.calls == []


@pytest.mark.parametrize(
    "message",
    ["HTTP Error 429: Too Many Requests", "ERROR: [youtube] Too Many Requests"],
    ids=["message_contains_429", "message_contains_too_many_requests_text"],
)
def test_fetch_via_ytdlp_retries_once_after_a_429_then_succeeds(monkeypatch, message):
    ok = ([RawCue(text="Hi", start=0.0, duration=1.0)], "ru", True)
    _once, sleep_spy = _wire_ytdlp(monkeypatch, [DownloadError(message), ok])

    result = fetch_via_ytdlp("vid1")

    assert result == ok
    assert sleep_spy.calls == [30]


def test_fetch_via_ytdlp_raises_rate_limited_once_retries_are_exhausted(monkeypatch):
    err = DownloadError("HTTP Error 429: Too Many Requests")
    _once, sleep_spy = _wire_ytdlp(monkeypatch, [err, err], retries_429=1)

    with pytest.raises(RateLimited):
        fetch_via_ytdlp("vid1")
    assert sleep_spy.calls == [30]


def test_fetch_via_ytdlp_non_429_download_error_raises_immediately_without_retry(monkeypatch):
    err = DownloadError("ERROR: Video unavailable")
    _once, sleep_spy = _wire_ytdlp(monkeypatch, [err])

    with pytest.raises(TranscriptError) as excinfo:
        fetch_via_ytdlp("vid1")
    assert excinfo.value.__cause__ is err
    assert sleep_spy.calls == []
    assert _once.calls == 1


def test_fetch_via_ytdlp_backoff_grows_as_30_times_the_attempt_number(monkeypatch):
    err = DownloadError("HTTP Error 429: Too Many Requests")
    ok = ([RawCue(text="Hi", start=0.0, duration=1.0)], "ru", True)
    _once, sleep_spy = _wire_ytdlp(monkeypatch, [err, err, err, ok], retries_429=3)

    result = fetch_via_ytdlp("vid1")

    assert result == ok
    assert sleep_spy.calls == [30, 60, 90]


# ===================================================================
# get_transcript — routing between backends
# ===================================================================

def test_get_transcript_source_asr_calls_fetch_via_asr_and_tags_source(monkeypatch):
    import src.etl.asr as asr_mod

    cues = [RawCue(text="Hi", start=0.0, duration=1.0)]

    def _fake_fetch_via_asr(video_id):
        return cues, "ru", True

    monkeypatch.setattr(asr_mod, "fetch_via_asr", _fake_fetch_via_asr)

    result = get_transcript("vid1", source="asr")
    assert result == (cues, "ru", True, "asr")


def test_get_transcript_source_ytapi_success_tags_source_ytapi(monkeypatch):
    cues = [RawCue(text="Hi", start=0.0, duration=1.0)]
    monkeypatch.setattr(transcripts, "fetch_via_ytapi", lambda vid: (cues, "ru", False))

    result = get_transcript("vid1", source="ytapi")
    assert result == (cues, "ru", False, "ytapi")


def test_get_transcript_source_ytapi_transcript_error_propagates_without_fallback(monkeypatch):
    """source="ytapi" is explicit: unlike "auto", it must not fall back to
    yt-dlp on failure."""
    def _boom(video_id):
        raise TranscriptError("no transcript")

    def _must_not_run(video_id):
        raise AssertionError("fetch_via_ytdlp must not be called for source='ytapi'")

    monkeypatch.setattr(transcripts, "fetch_via_ytapi", _boom)
    monkeypatch.setattr(transcripts, "fetch_via_ytdlp", _must_not_run)

    with pytest.raises(TranscriptError):
        get_transcript("vid1", source="ytapi")


def test_get_transcript_source_auto_falls_back_to_ytdlp_on_transcript_error(monkeypatch):
    cues = [RawCue(text="Hi", start=0.0, duration=1.0)]

    def _boom(video_id):
        raise TranscriptError("no transcript")

    monkeypatch.setattr(transcripts, "fetch_via_ytapi", _boom)
    monkeypatch.setattr(transcripts, "fetch_via_ytdlp", lambda vid: (cues, "ru", True))

    result = get_transcript("vid1", source="auto")
    assert result == (cues, "ru", True, "ytdlp")


def test_get_transcript_source_auto_rate_limited_propagates_without_fallback(monkeypatch):
    """A RateLimited from ytapi means YouTube is IP-blocking this client — the
    same block would hit yt-dlp too, so "auto" deliberately does not waste
    time falling back to it and re-raises instead."""
    def _boom(video_id):
        raise RateLimited("blocked")

    def _must_not_run(video_id):
        raise AssertionError("fetch_via_ytdlp must not be called after a RateLimited")

    monkeypatch.setattr(transcripts, "fetch_via_ytapi", _boom)
    monkeypatch.setattr(transcripts, "fetch_via_ytdlp", _must_not_run)

    with pytest.raises(RateLimited):
        get_transcript("vid1", source="auto")


def test_get_transcript_source_ytdlp_never_calls_fetch_via_ytapi(monkeypatch):
    cues = [RawCue(text="Hi", start=0.0, duration=1.0)]

    def _must_not_run(video_id):
        raise AssertionError("fetch_via_ytapi must not be called for source='ytdlp'")

    monkeypatch.setattr(transcripts, "fetch_via_ytapi", _must_not_run)
    monkeypatch.setattr(transcripts, "fetch_via_ytdlp", lambda vid: (cues, "ru", True))

    result = get_transcript("vid1", source="ytdlp")
    assert result == (cues, "ru", True, "ytdlp")


# ===================================================================
# _proxy_config
# ===================================================================

def test_proxy_config_returns_none_when_nothing_is_configured(monkeypatch):
    monkeypatch.delenv("WEBSHARE_PROXY_USERNAME", raising=False)
    monkeypatch.delenv("WEBSHARE_PROXY_PASSWORD", raising=False)
    monkeypatch.setattr(transcripts, "PROXY", None)
    assert transcripts._proxy_config() is None


def test_proxy_config_uses_generic_proxy_when_only_proxy_is_set(monkeypatch):
    monkeypatch.delenv("WEBSHARE_PROXY_USERNAME", raising=False)
    monkeypatch.delenv("WEBSHARE_PROXY_PASSWORD", raising=False)
    monkeypatch.setattr(transcripts, "PROXY", "http://proxy.example:8080")

    cfg = transcripts._proxy_config()

    assert isinstance(cfg, GenericProxyConfig)
    assert cfg.http_url == "http://proxy.example:8080"
    assert cfg.https_url == "http://proxy.example:8080"


def test_proxy_config_prefers_webshare_when_its_credentials_are_set(monkeypatch):
    monkeypatch.setenv("WEBSHARE_PROXY_USERNAME", "user1")
    monkeypatch.setenv("WEBSHARE_PROXY_PASSWORD", "pass1")
    monkeypatch.setattr(transcripts, "PROXY", "http://ignored:8080")

    cfg = transcripts._proxy_config()

    assert isinstance(cfg, WebshareProxyConfig)
    assert cfg.proxy_username == "user1"
    assert cfg.proxy_password == "pass1"


# ===================================================================
# fetch_via_ytdlp: the retry-count boundary
# ===================================================================
def test_fetch_via_ytdlp_zero_retries_tries_once_then_gives_up(monkeypatch):
    """YTDLP_RETRIES_429=0 means "no retries", not "no attempts": one request
    is made, and a 429 turns into RateLimited without sleeping."""
    once, sleep_spy = _wire_ytdlp(
        monkeypatch, [DownloadError("HTTP Error 429")], retries_429=0
    )

    with pytest.raises(RateLimited):
        fetch_via_ytdlp("vid1")

    assert once.calls == 1
    assert sleep_spy.calls == []


@pytest.mark.xfail(
    reason="transcripts.py:223 - the retry loop is range(1, YTDLP_RETRIES_429 + 2), "
    "so a negative setting makes it empty: the body never runs, _ytdlp_once is "
    "never called, and execution falls through to the RateLimited at line 236. "
    "Setting YTDLP_RETRIES_429=-1 to switch retries off therefore marks every "
    "video rate_limited without a single request. The value comes from the "
    "environment (config.py:33), so this is reachable in a real run.",
    strict=True,
)
def test_fetch_via_ytdlp_negative_retries_still_makes_one_attempt(monkeypatch):
    """A negative retry count should still fetch once, not fail outright."""
    ok = ([RawCue(text="Hi", start=0.0, duration=1.0)], "ru", True)
    once, _sleep_spy = _wire_ytdlp(monkeypatch, [ok], retries_429=-1)

    result = fetch_via_ytdlp("vid1")

    assert once.calls == 1
    assert result == ok


def test_fetch_via_ytdlp_negative_retries_currently_fails_without_trying(monkeypatch):
    """Pins today's behaviour for the negative setting, alongside the xfail
    above that says what it ought to be."""
    once, _sleep_spy = _wire_ytdlp(monkeypatch, [], retries_429=-1)

    with pytest.raises(RateLimited):
        fetch_via_ytdlp("vid1")

    assert once.calls == 0


# ===================================================================
# get_transcript: an unrecognised source
# ===================================================================
def test_get_transcript_unknown_source_falls_through_to_ytdlp(monkeypatch):
    """A source name outside {asr, auto, ytapi, ytdlp} is not rejected: the
    routing is a chain of ifs, so anything unrecognised ends up on yt-dlp.

    Pinned as current behaviour — the CLI restricts the flag with argparse
    choices, so a bad value cannot arrive from there.
    """
    ok = ([RawCue(text="Hi", start=0.0, duration=1.0)], "ru", True)

    def _must_not_run(video_id):
        raise AssertionError("ytapi must not be tried for an unknown source")

    monkeypatch.setattr(transcripts, "fetch_via_ytapi", _must_not_run)
    monkeypatch.setattr(transcripts, "fetch_via_ytdlp", lambda video_id: ok)

    cues, lang, gen, used = get_transcript("vid1", source="мусор")

    assert (cues, lang, gen, used) == (*ok, "ytdlp")


# ===================================================================
# _ytdlp_once
# ===================================================================
# The function downloads subtitles into a temporary directory of its own, then
# hands that directory to _load_subs. yt_dlp.YoutubeDL is imported lazily
# inside it, so the fake goes onto the yt_dlp package attribute — the same
# seam tests/etl/test_asr.py uses for download_audio.

def _install_fake_ytdlp_download(monkeypatch, on_download):
    """Replaces yt_dlp.YoutubeDL with a fake whose download() runs
    on_download(opts), typically writing a subtitle file into the directory
    named by opts["outtmpl"]."""
    import yt_dlp

    class FakeYoutubeDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def download(self, urls):
            on_download(self.opts)

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYoutubeDL)


def _subtitle_writer(filename: str, payload: str):
    """Builds an on_download that drops one subtitle file next to outtmpl."""
    from pathlib import Path

    captured: dict = {}

    def on_download(opts):
        captured["opts"] = opts
        Path(opts["outtmpl"]).parent.joinpath(filename).write_text(payload, encoding="utf-8")

    return on_download, captured


JSON3_PAYLOAD = json.dumps(
    {"events": [{"tStartMs": 1000, "dDurationMs": 2000, "segs": [{"utf8": "Привет"}]}]}
)


def test_ytdlp_once_parses_the_downloaded_json3_and_reports_its_language(monkeypatch):
    """The happy path end to end: yt-dlp leaves a json3 file, it is parsed,
    and the language is read off the filename."""
    monkeypatch.setattr(transcripts, "SUB_LANGS", ["ru", "ru-RU"])
    on_download, _captured = _subtitle_writer("vid1.ru.json3", JSON3_PAYLOAD)
    _install_fake_ytdlp_download(monkeypatch, on_download)

    cues, lang, is_generated = transcripts._ytdlp_once("vid1")

    assert cues == [RawCue(text="Привет", start=1.0, duration=2.0)]
    assert lang == "ru"
    assert is_generated is True


def test_ytdlp_once_defaults_to_ru_when_the_filename_names_no_known_language(monkeypatch):
    """A subtitle file whose name carries no configured language code still
    yields a transcript, tagged with the fallback language."""
    monkeypatch.setattr(transcripts, "SUB_LANGS", ["ru"])
    on_download, _captured = _subtitle_writer("vid1.json3", JSON3_PAYLOAD)
    _install_fake_ytdlp_download(monkeypatch, on_download)

    _cues, lang, _gen = transcripts._ytdlp_once("vid1")

    assert lang == "ru"


def test_ytdlp_once_empty_subtitle_file_raises_transcript_error(monkeypatch):
    """A file that parses but holds no cues is a failure, not an empty
    success: an empty transcript downstream looks like a video with no speech."""
    monkeypatch.setattr(transcripts, "SUB_LANGS", ["ru"])
    on_download, _captured = _subtitle_writer("vid1.ru.json3", json.dumps({"events": []}))
    _install_fake_ytdlp_download(monkeypatch, on_download)

    with pytest.raises(TranscriptError):
        transcripts._ytdlp_once("vid1")


def test_ytdlp_once_player_client_survives_the_network_extractor_args(monkeypatch):
    """player_client is merged AFTER the network options on purpose.

    ytdlp_network_opts contributes its own extractor_args.youtube (carrying
    the interface language); a plain dict update would drop the client choice,
    and yt-dlp would fall back to the web client that serves no subtitles.
    Both keys have to survive.
    """
    monkeypatch.setattr(transcripts, "SUB_LANGS", ["ru"])
    monkeypatch.setattr(
        transcripts,
        "ytdlp_network_opts",
        lambda: {"extractor_args": {"youtube": {"lang": ["ru"]}}},
    )
    on_download, captured = _subtitle_writer("vid1.ru.json3", JSON3_PAYLOAD)
    _install_fake_ytdlp_download(monkeypatch, on_download)

    transcripts._ytdlp_once("vid1")

    youtube_args = captured["opts"]["extractor_args"]["youtube"]
    assert youtube_args["lang"] == ["ru"]
    assert youtube_args["player_client"] == ["android", "ios", "web"]


def test_ytdlp_once_requests_the_configured_subtitle_languages(monkeypatch):
    """The configured language list reaches yt-dlp: asking for the wrong ones
    downloads nothing while looking like a successful run."""
    monkeypatch.setattr(transcripts, "SUB_LANGS", ["ru", "ru-RU"])
    on_download, captured = _subtitle_writer("vid1.ru.json3", JSON3_PAYLOAD)
    _install_fake_ytdlp_download(monkeypatch, on_download)

    transcripts._ytdlp_once("vid1")

    assert captured["opts"]["subtitleslangs"] == ["ru", "ru-RU"]


# ===================================================================
# remaining branches: empty vtt cues, odd json3 timings, half-set proxy creds
# ===================================================================
def test_parse_vtt_block_whose_text_is_only_tags_is_dropped(tmp_path):
    """After the tags are stripped the block has no text left, so it produces
    no cue rather than an empty one."""
    path = tmp_path / "sub.vtt"
    path.write_text(
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:01.000\n"
        "<c></c>\n\n"
        "00:00:01.000 --> 00:00:02.000\n"
        "Привет\n",
        encoding="utf-8",
    )

    assert _parse_vtt(path) == [RawCue(text="Привет", start=1.0, duration=1.0)]


def test_parse_vtt_trailing_block_with_only_tags_is_dropped(tmp_path):
    """The same holds for the final block, which is flushed after the loop."""
    path = tmp_path / "sub.vtt"
    path.write_text(
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:01.000\n"
        "Привет\n\n"
        "00:00:01.000 --> 00:00:02.000\n"
        "<c></c>\n",
        encoding="utf-8",
    )

    assert _parse_vtt(path) == [RawCue(text="Привет", start=0.0, duration=1.0)]


@pytest.mark.parametrize(
    "start_ms, duration_ms, expected_start, expected_duration",
    [
        pytest.param(0, 0, 0.0, 0.0, id="zero_start_and_duration"),
        pytest.param(-500, 1000, -0.5, 1.0, id="negative_start_passes_through"),
        pytest.param(1000, -500, 1.0, -0.5, id="negative_duration_passes_through"),
    ],
)
def test_parse_json3_passes_odd_timings_through_unvalidated(
    tmp_path, start_ms, duration_ms, expected_start, expected_duration
):
    """The parser converts milliseconds and validates nothing.

    Pinned because a negative start ends up in the ?t= link of a segment: the
    parser is not where that gets caught today, and a silent clamp added here
    would change the artifacts on disk.
    """
    path = _write_json3(tmp_path, {"events": [
        {"tStartMs": start_ms, "dDurationMs": duration_ms, "segs": [{"utf8": "Текст"}]}
    ]})

    result = _parse_json3(path)

    assert result[0].start == pytest.approx(expected_start)
    assert result[0].duration == pytest.approx(expected_duration)


@pytest.mark.parametrize(
    "username, password",
    [
        pytest.param("user", None, id="only_username"),
        pytest.param(None, "pass", id="only_password"),
    ],
)
def test_proxy_config_ignores_half_configured_webshare_credentials(
    monkeypatch, username, password
):
    """Both Webshare variables are required: with only one of them set the
    code must not build a WebshareProxyConfig carrying a None.

    The guard is `and`; an `or` there would hand the library a half-filled
    config and fail at request time, far from the cause.
    """
    monkeypatch.delenv("WEBSHARE_PROXY_USERNAME", raising=False)
    monkeypatch.delenv("WEBSHARE_PROXY_PASSWORD", raising=False)
    if username is not None:
        monkeypatch.setenv("WEBSHARE_PROXY_USERNAME", username)
    if password is not None:
        monkeypatch.setenv("WEBSHARE_PROXY_PASSWORD", password)
    monkeypatch.setattr(transcripts, "PROXY", None)

    assert transcripts._proxy_config() is None
