"""Parsing the query intent into a soft playlist filter (Step 4)."""
from __future__ import annotations

import re

from qdrant_client import models

from src.config import PLAYLIST_INTENTS

_COMPILED = [(re.compile(pat, re.IGNORECASE), title) for pat, title in PLAYLIST_INTENTS]


def detect_playlists(query: str) -> list[str]:
    hits: list[str] = []
    for rx, title in _COMPILED:
        if rx.search(query) and title not in hits:
            hits.append(title)
    return hits


def build_intent_filter(query: str) -> tuple[models.Filter | None, list[str]]:
    """Returns (filter | None, matched_playlists).
    A 'should' filter — a point passes if its playlist_titles contains at least
    one of the guessed playlists. If nothing was guessed, there is no filter."""
    titles = detect_playlists(query)
    if not titles:
        return None, []
    flt = models.Filter(
        should=[
            models.FieldCondition(key="playlist_titles", match=models.MatchValue(value=t))
            for t in titles
        ]
    )
    return flt, titles
