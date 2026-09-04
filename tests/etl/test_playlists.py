"""Tests for src/etl/playlists.py: fetch_playlist_entries() (fast, flat
listing of a playlist) and fetch_video_meta() (slower, per-video request).

Both functions import `YoutubeDL` at module scope and construct it directly
(`with YoutubeDL(opts) as ydl: ...`), so there is no constructor parameter to
inject a fake through. Per the project's conventions this is one of the five sanctioned
monkeypatch spots: we replace `src.etl.playlists.YoutubeDL` with a fake
context manager that records the opts it was built with and returns/raises a
canned `extract_info()` result instead of touching the network.
"""
from __future__ import annotations

from typing import Any

import pytest
from yt_dlp.utils import DownloadError

import src.etl.playlists as playlists
from src.etl.schemas import VideoMeta


class _FakeYoutubeDL:
    """Stands in for yt_dlp.YoutubeDL.

    The instance itself plays both roles: calling it (`YoutubeDL(opts)`)
    records `opts` and returns `self`; using it as a context manager
    (`with ... as ydl`) just hands back `self` too. `extract_info` returns
    the canned `response` or raises the canned `error`.
    """

    def __init__(self, response: Any = None, error: Exception | None = None):
        self._response = response
        self._error = error
        self.opts: dict | None = None
        self.url: str | None = None

    def __call__(self, opts: dict) -> "_FakeYoutubeDL":
        self.opts = opts
        return self

    def __enter__(self) -> "_FakeYoutubeDL":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def extract_info(self, url: str, download: bool = False):
        self.url = url
        if self._error is not None:
            raise self._error
        return self._response


def make_fake_ydl(*, response: Any = None, error: Exception | None = None) -> _FakeYoutubeDL:
    return _FakeYoutubeDL(response=response, error=error)


# =======================================================================
# fetch_playlist_entries
# =======================================================================
def test_fetch_playlist_entries_happy_path(monkeypatch):
    """A normal playlist response turns into one dict per entry, carrying the
    playlist's own id/title alongside each video."""
    info = {
        "id": "PL1",
        "title": "Супы",
        "entries": [
            {"id": "v1", "title": "Борщ"},
            {"id": "v2", "title": "Щи"},
        ],
    }
    fake = make_fake_ydl(response=info)
    monkeypatch.setattr(playlists, "YoutubeDL", fake)

    result = playlists.fetch_playlist_entries("https://youtube.com/playlist?list=PL1")

    assert result == [
        {"video_id": "v1", "title": "Борщ", "playlist_id": "PL1", "playlist_title": "Супы"},
        {"video_id": "v2", "title": "Щи", "playlist_id": "PL1", "playlist_title": "Супы"},
    ]


def test_fetch_playlist_entries_none_response_returns_empty_list(monkeypatch):
    """extract_info() returning None (e.g. a fully dead playlist) must not
    raise — it logs a warning and yields no videos."""
    fake = make_fake_ydl(response=None)
    monkeypatch.setattr(playlists, "YoutubeDL", fake)

    assert playlists.fetch_playlist_entries("url") == []


@pytest.mark.parametrize(
    "info",
    [
        pytest.param({"id": "PL1", "title": "Супы"}, id="entries_key_missing"),
        pytest.param({"id": "PL1", "title": "Супы", "entries": None}, id="entries_is_none"),
    ],
)
def test_fetch_playlist_entries_missing_entries_returns_empty_list(monkeypatch, info):
    """Whether `entries` is absent or explicitly None, the result is []."""
    fake = make_fake_ydl(response=info)
    monkeypatch.setattr(playlists, "YoutubeDL", fake)

    assert playlists.fetch_playlist_entries("url") == []


def test_fetch_playlist_entries_skips_entries_without_id_or_none(monkeypatch):
    """A None entry (ignoreerrors leaves a hole) and an entry lacking an id
    are both dropped; only the well-formed entry survives."""
    info = {
        "id": "PL1",
        "title": "Супы",
        "entries": [
            None,
            {"title": "no id"},
            {"id": "v2", "title": "Щи"},
        ],
    }
    fake = make_fake_ydl(response=info)
    monkeypatch.setattr(playlists, "YoutubeDL", fake)

    result = playlists.fetch_playlist_entries("url")

    assert [e["video_id"] for e in result] == ["v2"]


@pytest.mark.parametrize(
    "entry",
    [
        pytest.param({"id": "v1"}, id="title_key_missing"),
        pytest.param({"id": "v1", "title": None}, id="title_is_none"),
    ],
)
def test_fetch_playlist_entries_missing_entry_title_becomes_empty_string(monkeypatch, entry):
    """A missing or explicitly-None entry title becomes "" — never None."""
    info = {"id": "PL1", "title": "Супы", "entries": [entry]}
    fake = make_fake_ydl(response=info)
    monkeypatch.setattr(playlists, "YoutubeDL", fake)

    result = playlists.fetch_playlist_entries("url")

    assert result[0]["title"] == ""


def test_fetch_playlist_entries_missing_playlist_id_and_title_become_empty_strings(monkeypatch):
    """A playlist response without its own id/title still yields entries,
    with playlist_id/playlist_title defaulted to "" rather than None."""
    info = {"entries": [{"id": "v1", "title": "Борщ"}]}
    fake = make_fake_ydl(response=info)
    monkeypatch.setattr(playlists, "YoutubeDL", fake)

    result = playlists.fetch_playlist_entries("url")

    assert result[0]["playlist_id"] == ""
    assert result[0]["playlist_title"] == ""


def test_fetch_playlist_entries_uses_flat_extract_and_safety_opts(monkeypatch):
    """extract_flat='in_playlist' keeps the listing fast by not opening every
    video; skip_download/ignoreerrors make it a safe, offline-metadata-only
    call."""
    monkeypatch.setattr(playlists, "ytdlp_network_opts", lambda: {})
    fake = make_fake_ydl(response={"id": "PL1", "title": "T", "entries": []})
    monkeypatch.setattr(playlists, "YoutubeDL", fake)

    playlists.fetch_playlist_entries("url")

    assert fake.opts["extract_flat"] == "in_playlist"
    assert fake.opts["skip_download"] is True
    assert fake.opts["ignoreerrors"] is True


def test_fetch_playlist_entries_merges_network_opts(monkeypatch):
    """Options coming from ytdlp_network_opts() (proxy, cookies, ...) reach
    the real YoutubeDL call, merged on top of the base flat opts."""
    monkeypatch.setattr(
        playlists,
        "ytdlp_network_opts",
        lambda: {"proxy": "socks5://127.0.0.1:9050", "cookiefile": "cookies.txt"},
    )
    fake = make_fake_ydl(response={"id": "PL1", "title": "T", "entries": []})
    monkeypatch.setattr(playlists, "YoutubeDL", fake)

    playlists.fetch_playlist_entries("url")

    assert fake.opts["proxy"] == "socks5://127.0.0.1:9050"
    assert fake.opts["cookiefile"] == "cookies.txt"
    # the base flat opts are still there too — it is a merge, not a replace
    assert fake.opts["extract_flat"] == "in_playlist"


# =======================================================================
# fetch_video_meta
# =======================================================================
def test_fetch_video_meta_full_response_returns_populated_videometa(monkeypatch):
    """A full extract_info() response fills every VideoMeta field and the
    url is built as watch?v=<id>."""
    info = {
        "title": "Борщ классический",
        "channel": "oblomoffood",
        "uploader": "someone_else",
        "upload_date": "20230101",
        "duration": 600,
        "description": "Вкусный борщ",
    }
    fake = make_fake_ydl(response=info)
    monkeypatch.setattr(playlists, "YoutubeDL", fake)

    result = playlists.fetch_video_meta("v1", fallback_title="fallback")

    assert result == VideoMeta(
        video_id="v1",
        title="Борщ классический",
        url="https://www.youtube.com/watch?v=v1",
        channel="oblomoffood",
        upload_date="20230101",
        duration=600,
        description="Вкусный борщ",
    )
    assert fake.url == "https://www.youtube.com/watch?v=v1"


def test_fetch_video_meta_channel_falls_back_to_uploader(monkeypatch):
    """When "channel" is absent, "uploader" is used instead."""
    info = {"title": "Борщ", "uploader": "oblomoffood"}
    fake = make_fake_ydl(response=info)
    monkeypatch.setattr(playlists, "YoutubeDL", fake)

    result = playlists.fetch_video_meta("v1")

    assert result.channel == "oblomoffood"


def test_fetch_video_meta_download_error_returns_fallback_videometa(monkeypatch):
    """A DownloadError (e.g. a deleted/private video) is swallowed: the
    function returns a minimal VideoMeta built from fallback_title instead of
    propagating the exception."""
    fake = make_fake_ydl(error=DownloadError("network blocked"))
    monkeypatch.setattr(playlists, "YoutubeDL", fake)

    result = playlists.fetch_video_meta("v1", fallback_title="Борщ (плейлист)")

    assert result == VideoMeta(
        video_id="v1",
        title="Борщ (плейлист)",
        url="https://www.youtube.com/watch?v=v1",
    )


def test_fetch_video_meta_none_response_returns_fallback_videometa(monkeypatch):
    """extract_info() returning None takes the same fallback path as a
    DownloadError."""
    fake = make_fake_ydl(response=None)
    monkeypatch.setattr(playlists, "YoutubeDL", fake)

    result = playlists.fetch_video_meta("v1", fallback_title="Борщ (плейлист)")

    assert result == VideoMeta(
        video_id="v1",
        title="Борщ (плейлист)",
        url="https://www.youtube.com/watch?v=v1",
    )


def test_fetch_video_meta_missing_title_uses_fallback_title(monkeypatch):
    """A response without a "title" key falls back to fallback_title rather
    than storing an empty/None title."""
    info = {"channel": "oblomoffood"}
    fake = make_fake_ydl(response=info)
    monkeypatch.setattr(playlists, "YoutubeDL", fake)

    result = playlists.fetch_video_meta("v1", fallback_title="Борщ (плейлист)")

    assert result.title == "Борщ (плейлист)"


def test_fetch_video_meta_passes_full_opts_without_extract_flat(monkeypatch):
    """Unlike the playlist listing, the per-video fetch does NOT set
    extract_flat (it needs the full metadata), and still merges in network
    opts and the download-safety flags."""
    monkeypatch.setattr(
        playlists, "ytdlp_network_opts", lambda: {"proxy": "socks5://127.0.0.1:9050"}
    )
    fake = make_fake_ydl(response={"title": "Борщ"})
    monkeypatch.setattr(playlists, "YoutubeDL", fake)

    playlists.fetch_video_meta("v1")

    assert "extract_flat" not in fake.opts
    assert fake.opts["skip_download"] is True
    assert fake.opts["ignoreerrors"] is True
    assert fake.opts["proxy"] == "socks5://127.0.0.1:9050"


def test_fetch_video_meta_non_download_error_propagates(monkeypatch):
    """Only DownloadError is caught (playlists.py:64): any other exception
    from yt-dlp is not swallowed and crashes the caller. Documented here as
    the current, deliberate behaviour (a narrow except, not a bug) rather
    than an oversight — a broad `except Exception` would also hide real
    programming errors."""
    fake = make_fake_ydl(error=RuntimeError("unexpected"))
    monkeypatch.setattr(playlists, "YoutubeDL", fake)

    with pytest.raises(RuntimeError, match="unexpected"):
        playlists.fetch_video_meta("v1")
