"""Heuristic labelling of segments by recipe section.

Labels: intro | ingredients | steps | outro | other
The oblomoff transcripts are a conversational monologue, so this is a soft
signal: it is used to (a) cut chunks on section borders and (b) tag a chunk for
later filtering in retrieval (Step 4)."""
from __future__ import annotations

import re

from src.config import (
    INGREDIENT_PHRASES,
    INGREDIENT_UNIT_RE,
    INTRO_MARKERS,
    OUTRO_MARKERS,
    SEQ_MARKERS,
    STEP_VERBS,
)

_UNIT_RE = re.compile(INGREDIENT_UNIT_RE, re.IGNORECASE)
_STEP_RE = re.compile(r"(?:" + "|".join(re.escape(v) for v in STEP_VERBS) + r")", re.IGNORECASE)
_SEQ_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(m) for m in SEQ_MARKERS) + r")\b", re.IGNORECASE
)
_INTRO_RE = re.compile("|".join(re.escape(m) for m in INTRO_MARKERS), re.IGNORECASE)
_OUTRO_RE = re.compile("|".join(re.escape(m) for m in OUTRO_MARKERS), re.IGNORECASE)
_PHRASE_RE = re.compile("|".join(re.escape(p) for p in INGREDIENT_PHRASES), re.IGNORECASE)


def _scores(text: str) -> tuple[int, int]:
    """(ingredient_score, step_score)."""
    ing = len(_UNIT_RE.findall(text)) + len(_PHRASE_RE.findall(text))
    step = len(_STEP_RE.findall(text)) + len(_SEQ_RE.findall(text))
    return ing, step


def classify_segment(text: str, position: float) -> str:
    """position — the relative offset from the start of the video [0..1]."""
    low = text.lower()
    ing, step = _scores(low)

    if position <= 0.12 and _INTRO_RE.search(low):
        return "intro"
    if position >= 0.82 and _OUTRO_RE.search(low):
        return "outro"
    if ing >= 2 and ing > step:
        return "ingredients"
    if step >= 2 and step >= ing:
        return "steps"
    if step == 1 and ing == 0 and _SEQ_RE.search(low):
        return "steps"
    return "other"


def smooth(labels: list[str]) -> list[str]:
    """Fills single 'other' labels between identical neighbours and joins
    'ingredients'/'steps' across a one-label gap."""
    out = list(labels)
    for i in range(1, len(out) - 1):
        if out[i] == "other" and out[i - 1] == out[i + 1] and out[i - 1] in {"ingredients", "steps"}:
            out[i] = out[i - 1]
    for i in range(2, len(out) - 2):
        if (
            out[i] == "other"
            and out[i - 1] in {"ingredients", "steps"}
            and out[i - 2] == out[i - 1]
            and out[i + 1] == out[i - 1]
        ):
            out[i] = out[i - 1]
    return out


def label_segments(texts: list[str]) -> list[str]:
    n = max(len(texts), 1)
    raw = [classify_segment(t, i / n) for i, t in enumerate(texts)]
    return smooth(raw)


def starts_new_step(text: str) -> bool:
    return bool(_SEQ_RE.match(text.strip().lower()))
