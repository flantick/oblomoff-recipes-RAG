"""Fetching Russian subtitles with timecodes.

primary : youtube-transcript-api (v1.x)
fallback: yt-dlp (downloads auto-subs in the json3 format and parses them)
"""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

from loguru import logger

from src import config
from src.config import PROXY, SUB_LANGS, YTDLP_RETRIES_429, YTDLP_SLEEP_SUBTITLES
from src.etl.schemas import RawCue
from src.etl.ytdlp_common import ytdlp_network_opts

# --- youtube-transcript-api (1.x compatibility) ----------------------
try:  # 1.x exports the errors from the package root
    from youtube_transcript_api import (  # type: ignore
        YouTubeTranscriptApi,
        NoTranscriptFound,
        TranscriptsDisabled,
        VideoUnavailable,
    )
except ImportError:  # pragma: no cover - in case of older builds
    from youtube_transcript_api import YouTubeTranscriptApi  # type: ignore
    from youtube_transcript_api._errors import (  # type: ignore
        NoTranscriptFound,
        TranscriptsDisabled,
        VideoUnavailable,
    )

try:  # YouTube rate limits (missing in some versions)
    from youtube_transcript_api import IpBlocked, RequestBlocked  # type: ignore
except ImportError:  # pragma: no cover
    class IpBlocked(Exception):  # type: ignore
        ...

    class RequestBlocked(Exception):  # type: ignore
        ...


class TranscriptError(RuntimeError):
    """No backend managed to fetch the subtitles."""


class RateLimited(TranscriptError):
    """YouTube is temporarily blocking requests (IpBlocked / HTTP 429)."""


def _proxy_config() -> Any | None:
    # Webshare residential is the workaround recommended for bulk collection
    # (see the youtube-transcript-api README).
    import os

    ws_user = os.getenv("WEBSHARE_PROXY_USERNAME")
    ws_pass = os.getenv("WEBSHARE_PROXY_PASSWORD")
    if ws_user and ws_pass:
        try:
            from youtube_transcript_api.proxies import WebshareProxyConfig  # type: ignore

            return WebshareProxyConfig(proxy_username=ws_user, proxy_password=ws_pass)
        except Exception:  # pragma: no cover
            pass
    if PROXY:
        try:
            from youtube_transcript_api.proxies import GenericProxyConfig  # type: ignore

            return GenericProxyConfig(http_url=PROXY, https_url=PROXY)
        except Exception:  # pragma: no cover
            return None
    return None


def fetch_via_ytapi(video_id: str) -> tuple[list[RawCue], str, bool]:
    """Returns (cues, language_code, is_generated)."""
    try:
        return _ytapi_inner(video_id)
    except (IpBlocked, RequestBlocked) as exc:
        # the block can arrive from both .list() and .fetch()
        raise RateLimited(f"{video_id}: {exc.__class__.__name__}") from exc
    except (TranscriptsDisabled, VideoUnavailable) as exc:
        raise TranscriptError(f"{video_id}: {exc.__class__.__name__}") from exc


def _ytapi_inner(video_id: str) -> tuple[list[RawCue], str, bool]:
    api = YouTubeTranscriptApi(proxy_config=_proxy_config())
    tlist = api.list(video_id)

    transcript = None
    try:
        transcript = tlist.find_manually_created_transcript(SUB_LANGS)
    except NoTranscriptFound:
        try:
            transcript = tlist.find_generated_transcript(SUB_LANGS)
        except NoTranscriptFound:
            # take whatever is available and translate it into Russian
            for t in tlist:
                if t.is_translatable:
                    transcript = t.translate("ru")
                    break

    if transcript is None:
        raise TranscriptError(f"{video_id}: нет подходящего транскрипта")

    fetched = transcript.fetch()
    snippets = fetched.snippets if hasattr(fetched, "snippets") else fetched
    cues = [
        RawCue(text=s.text, start=float(s.start), duration=float(s.duration))
        for s in snippets
        if getattr(s, "text", "").strip()
    ]
    if not cues:
        raise TranscriptError(f"{video_id}: пустой транскрипт")
    return cues, transcript.language_code, bool(transcript.is_generated)


def _parse_json3(path: Path) -> list[RawCue]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cues: list[RawCue] = []
    for ev in data.get("events", []):
        segs = ev.get("segs")
        if not segs or "tStartMs" not in ev:
            continue
        text = "".join(s.get("utf8", "") for s in segs).strip()
        if not text or text == "\n":
            continue
        start = ev["tStartMs"] / 1000.0
        dur = ev.get("dDurationMs", 0) / 1000.0
        cues.append(RawCue(text=text, start=start, duration=dur))
    return cues


def _ytdlp_once(video_id: str) -> tuple[list[RawCue], str, bool]:
    from yt_dlp import YoutubeDL

    url = f"https://www.youtube.com/watch?v={video_id}"
    with tempfile.TemporaryDirectory() as tmp:
        outtmpl = str(Path(tmp) / "%(id)s.%(ext)s")
        opts: dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": SUB_LANGS,
            "subtitlesformat": "json3/srv3/vtt/best",
            "outtmpl": outtmpl,
            "sleep_interval_subtitles": YTDLP_SLEEP_SUBTITLES,
            "retries": 5,
            "extractor_retries": 3,
            **ytdlp_network_opts(),
        }
        # player_client is added AFTER the network options, otherwise their
        # extractor_args (carrying lang) would wipe out the client choice
        ea = opts.setdefault("extractor_args", {})
        ea["youtube"] = {**ea.get("youtube", {}), "player_client": ["android", "ios", "web"]}
        with YoutubeDL(opts) as ydl:
            ydl.download([url])

        cues, chosen = _load_subs(Path(tmp), video_id)
        if not cues:
            raise TranscriptError(f"{video_id}: пустой субтитр")
        lang = next((l for l in SUB_LANGS if l in chosen.name), "ru")
        return cues, lang, True


def _load_subs(tmp: Path, video_id: str) -> tuple[list[RawCue], Path]:
    """Finds the downloaded subtitle file (json3 preferred, vtt otherwise)."""
    order = ["json3", "srv3", "vtt"]
    files = sorted(
        (p for p in tmp.glob(f"{video_id}*") if p.suffix.lstrip(".") in order),
        key=lambda p: (
            order.index(p.suffix.lstrip(".")),
            next((i for i, l in enumerate(SUB_LANGS) if l in p.name), 99),
        ),
    )
    if not files:
        raise TranscriptError(f"{video_id}: yt-dlp не отдал субтитры")
    chosen = files[0]
    ext = chosen.suffix.lstrip(".")
    cues = _parse_json3(chosen) if ext in ("json3", "srv3") else _parse_vtt(chosen)
    return cues, chosen


def _parse_vtt(path: Path) -> list[RawCue]:
    import re

    def ts(v: str) -> float:
        h, m, s = v.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s.replace(",", "."))

    cues: list[RawCue] = []
    block_time: tuple[float, float] | None = None
    text_lines: list[str] = []
    rx = re.compile(r"(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[.,]\d{3})")
    for raw in path.read_text(encoding="utf-8").splitlines():
        m = rx.search(raw)
        if m:
            if block_time and text_lines:
                txt = " ".join(text_lines).strip()
                if txt:
                    cues.append(RawCue(text=txt, start=block_time[0], duration=block_time[1] - block_time[0]))
            block_time = (ts(m.group(1)), ts(m.group(2)))
            text_lines = []
        elif raw.strip() and "-->" not in raw and not raw.strip().isdigit() and raw.strip() != "WEBVTT":
            text_lines.append(re.sub(r"<[^>]+>", "", raw).strip())
    if block_time and text_lines:
        txt = " ".join(text_lines).strip()
        if txt:
            cues.append(RawCue(text=txt, start=block_time[0], duration=block_time[1] - block_time[0]))
    return cues


def fetch_via_ytdlp(video_id: str) -> tuple[list[RawCue], str, bool]:
    """yt-dlp -> json3, with a backoff on HTTP 429."""
    from yt_dlp.utils import DownloadError

    for attempt in range(1, YTDLP_RETRIES_429 + 2):
        try:
            return _ytdlp_once(video_id)
        except DownloadError as exc:
            msg = str(exc)
            if "429" in msg or "Too Many Requests" in msg:
                if attempt > YTDLP_RETRIES_429:
                    raise RateLimited(f"{video_id}: HTTP 429 (yt-dlp), исчерпаны ретраи") from exc
                wait = 30 * attempt
                logger.warning("429 на {}, пауза {}s (попытка {}/{})", video_id, wait, attempt, YTDLP_RETRIES_429)
                time.sleep(wait)
                continue
            raise TranscriptError(f"{video_id}: {msg[:200]}") from exc
    raise RateLimited(f"{video_id}: HTTP 429 (yt-dlp)")


def get_transcript(video_id: str, source: str = "ytdlp") -> tuple[list[RawCue], str, bool, str]:
    """Returns (cues, language, is_generated, used_source).

    source: "asr" (local Whisper over the audio — never touches timedtext at
    all), "ytdlp", "ytapi", "auto".
    """
    if source == "asr":
        from src.etl.asr import fetch_via_asr  # lazy import: faster-whisper is heavy

        cues, lang, gen = fetch_via_asr(video_id)
        return cues, lang, gen, "asr"

    if source in ("auto", "ytapi"):
        try:
            cues, lang, gen = fetch_via_ytapi(video_id)
            return cues, lang, gen, "ytapi"
        except RateLimited:
            # the same IP block will hit yt-dlp too — do not waste time on it
            raise
        except TranscriptError as exc:
            if source == "ytapi":
                raise
            logger.warning("ytapi не сработал ({}), пробую yt-dlp", exc)

    cues, lang, gen = fetch_via_ytdlp(video_id)
    return cues, lang, gen, "ytdlp"
