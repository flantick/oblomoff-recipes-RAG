"""Shared yt-dlp network options: proxy + browser cookies (rate-limit workaround).

The values are read from src.config on every call, so the pipeline CLI flags can
override them at runtime (see pipeline.main)."""
from __future__ import annotations

from src import config


def ytdlp_network_opts() -> dict:
    o: dict = {}
    lang = getattr(config, "YTDLP_LANG", None)
    if lang:
        # without this, title/description come back auto-translated
        o["extractor_args"] = {"youtube": {"lang": [lang]}}
    if config.PROXY:
        o["proxy"] = config.PROXY

    cookies_file = getattr(config, "YTDLP_COOKIES_FILE", None)
    cookies_browser = getattr(config, "YTDLP_COOKIES_FROM_BROWSER", None)
    if cookies_file:
        o["cookiefile"] = cookies_file
    elif cookies_browser:
        # (browser, profile, keyring, container)
        o["cookiesfrombrowser"] = (cookies_browser, None, None, None)
    return o
