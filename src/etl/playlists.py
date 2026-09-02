"""Playlist -> list of videos + metadata (via yt-dlp)."""
from __future__ import annotations

from typing import Any

from loguru import logger
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

from src.etl.schemas import VideoMeta
from src.etl.ytdlp_common import ytdlp_network_opts

_FLAT_OPTS: dict[str, Any] = {
    "quiet": True,
    "no_warnings": True,
    "skip_download": True,
    "extract_flat": "in_playlist",   # do not open every video — keeps it fast
    "ignoreerrors": True,
}

_FULL_OPTS: dict[str, Any] = {
    "quiet": True,
    "no_warnings": True,
    "skip_download": True,
    "ignoreerrors": True,
}


def fetch_playlist_entries(playlist_url: str) -> list[dict[str, str]]:
    """Returns [{video_id, title, playlist_id, playlist_title}, ...]."""
    with YoutubeDL({**_FLAT_OPTS, **ytdlp_network_opts()}) as ydl:
        info = ydl.extract_info(playlist_url, download=False)

    if not info:
        logger.warning("Пустой ответ по плейлисту: {}", playlist_url)
        return []

    playlist_id = info.get("id") or ""
    playlist_title = info.get("title") or ""
    entries = info.get("entries") or []

    result: list[dict[str, str]] = []
    for e in entries:
        if not e or not e.get("id"):
            continue
        result.append(
            {
                "video_id": e["id"],
                "title": e.get("title") or "",
                "playlist_id": playlist_id,
                "playlist_title": playlist_title,
            }
        )
    logger.info("Плейлист {} ({}): {} видео", playlist_id, playlist_title, len(result))
    return result


def fetch_video_meta(video_id: str, fallback_title: str = "") -> VideoMeta:
    """Full metadata of a single video (slower — a separate request)."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        with YoutubeDL({**_FULL_OPTS, **ytdlp_network_opts()}) as ydl:
            info = ydl.extract_info(url, download=False)
    except DownloadError as exc:
        logger.warning("Не удалось получить метаданные {}: {}", video_id, exc)
        info = None

    if not info:
        return VideoMeta(video_id=video_id, title=fallback_title, url=url)

    return VideoMeta(
        video_id=video_id,
        title=info.get("title") or fallback_title,
        url=url,
        channel=info.get("channel") or info.get("uploader"),
        upload_date=info.get("upload_date"),
        duration=info.get("duration"),
        description=info.get("description"),
    )
