"""Transcript cleaning and merging into semantic blocks (Step 1)."""
from __future__ import annotations

import re

from loguru import logger

from src.config import (
    AD_MARKERS,
    BLOCK_GAP_SECONDS,
    BLOCK_MAX_CHARS,
    BLOCK_TARGET_CHARS,
    DELIVERY_MARKERS,
    FILLER_WORDS,
    INTERJECTIONS,
)
from src.etl.schemas import CleanSegment, RawCue, VideoMeta

try:
    from razdel import sentenize
    _HAS_RAZDEL = True
except ImportError:  # pragma: no cover
    _HAS_RAZDEL = False

# The window around an ad marker that gets cut out.
AD_WINDOW_BEFORE = 8.0   # sec before
AD_WINDOW_AFTER = 25.0   # sec after (an integration usually runs forward)

_BRACKETS_RE = re.compile(r"[\[(][^\])]*[\])]")          # [музыка], (смех)
_SPEAKER_RE = re.compile(r"^\s*(>>|-)\s+")
_MULTISPACE_RE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.!?…:;])")
_REPEAT_WORD_RE = re.compile(r"\b(\w+)(\s+\1\b){1,}", re.IGNORECASE)

_FILLER_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in sorted(FILLER_WORDS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
_INTERJ_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in INTERJECTIONS) + r")\b",
    re.IGNORECASE,
)
_AD_RE = re.compile("|".join(re.escape(m) for m in AD_MARKERS), re.IGNORECASE)
_DELIVERY_RE = re.compile("|".join(re.escape(m) for m in DELIVERY_MARKERS), re.IGNORECASE)


# --- video classification ------------------------------------------
def classify_recipe(meta: VideoMeta) -> VideoMeta:
    """Heuristically marks delivery reviews as non-recipes."""
    haystack = f"{meta.title}\n{meta.description or ''}"
    m = _DELIVERY_RE.search(haystack)
    if m:
        meta.is_recipe = False
        meta.skip_reason = f"delivery_marker:{m.group(0)!r}"
    return meta


# --- cleaning of a single cue ------------------------------------
def clean_line(text: str) -> str:
    text = text.replace("\n", " ")
    text = _BRACKETS_RE.sub(" ", text)
    text = _SPEAKER_RE.sub("", text)
    text = _INTERJ_RE.sub(" ", text)
    text = _FILLER_RE.sub(" ", text)
    text = _REPEAT_WORD_RE.sub(r"\1", text)
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = _MULTISPACE_RE.sub(" ", text).strip()
    text = re.sub(r"^[,.;:!?…\-\s]+", "", text)
    return text


# --- cutting out the ad windows ---------------------------------
def strip_ad_windows(cues: list[RawCue]) -> tuple[list[RawCue], list[dict]]:
    ad_centers = [c.start for c in cues if _AD_RE.search(c.text)]
    if not ad_centers:
        return cues, []

    spans: list[tuple[float, float]] = []
    for center in ad_centers:
        spans.append((center - AD_WINDOW_BEFORE, center + AD_WINDOW_AFTER))
    # merge overlapping windows
    spans.sort()
    merged: list[list[float]] = [list(spans[0])]
    for s, e in spans[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    def in_ad(t: float) -> bool:
        return any(s <= t <= e for s, e in merged)

    kept = [c for c in cues if not in_ad(c.start)]
    removed = [{"start": round(s, 1), "end": round(e, 1)} for s, e in merged]
    logger.info("Вырезано рекламных окон: {} (реплик убрано: {})",
                len(merged), len(cues) - len(kept))
    return kept, removed


# --- merging into semantic blocks ------------------------------
def _fmt_timecode(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def _ends_sentence(text: str) -> bool:
    return text.rstrip().endswith((".", "!", "?", "…"))


def merge_into_blocks(cues: list[RawCue], video_id: str) -> list[CleanSegment]:
    """Merges short cues into blocks of ~BLOCK_TARGET_CHARS without breaking
    sentences and while respecting pauses in speech."""
    segments: list[CleanSegment] = []
    buf: list[str] = []
    buf_start: float | None = None
    buf_end: float = 0.0
    prev_end: float | None = None

    def flush() -> None:
        nonlocal buf, buf_start, buf_end
        if not buf or buf_start is None:
            buf = []
            buf_start = None
            return
        raw = " ".join(buf).strip()
        text = _resegment(raw)
        if text:
            segments.append(
                CleanSegment(
                    text=text,
                    start=round(buf_start, 2),
                    end=round(buf_end, 2),
                    timecode=_fmt_timecode(buf_start),
                    url=f"https://youtu.be/{video_id}?t={int(buf_start)}",
                )
            )
        buf = []
        buf_start = None

    for c in cues:
        line = clean_line(c.text)
        if not line:
            continue
        gap = (c.start - prev_end) if prev_end is not None else 0.0
        cur_len = sum(len(x) + 1 for x in buf)

        if buf and (gap >= BLOCK_GAP_SECONDS or cur_len >= BLOCK_MAX_CHARS):
            flush()

        if buf_start is None:
            buf_start = c.start
        buf.append(line)
        buf_end = c.start + c.duration
        prev_end = buf_end

        cur_len = sum(len(x) + 1 for x in buf)
        if cur_len >= BLOCK_TARGET_CHARS and _ends_sentence(line):
            flush()

    flush()
    return segments


def _resegment(text: str) -> str:
    """Cosmetics: auto-subs have no punctuation, so we merely normalise the case
    of the first letter. Proper punctuation is the job of the later LLM cleanup."""
    if _HAS_RAZDEL:
        parts = [s.text.strip() for s in sentenize(text)]
        text = " ".join(p for p in parts if p)
    return text[:1].upper() + text[1:]


def merge_sentences_into_blocks(sentences, video_id: str) -> list[CleanSegment]:
    """Merges already punctuated sentences (from punctuation.py) into blocks.
    We flush on a sentence boundary once BLOCK_TARGET_CHARS is reached, and cut
    hard at BLOCK_MAX_CHARS or on a BLOCK_GAP_SECONDS pause."""
    segments: list[CleanSegment] = []
    buf: list[str] = []
    buf_start: float | None = None
    buf_end: float = 0.0
    prev_end: float | None = None

    def flush() -> None:
        nonlocal buf, buf_start, buf_end
        if not buf or buf_start is None:
            buf, buf_start = [], None
            return
        text = " ".join(buf).strip()
        if text:
            segments.append(
                CleanSegment(
                    text=text,
                    start=round(buf_start, 2),
                    end=round(buf_end, 2),
                    timecode=_fmt_timecode(buf_start),
                    url=f"https://youtu.be/{video_id}?t={int(buf_start)}",
                )
            )
        buf, buf_start = [], None

    for sent in sentences:
        gap = (sent.start - prev_end) if prev_end is not None else 0.0
        cur_len = sum(len(x) + 1 for x in buf)
        if buf and (gap >= BLOCK_GAP_SECONDS or cur_len >= BLOCK_MAX_CHARS):
            flush()

        if buf_start is None:
            buf_start = sent.start
        buf.append(sent.text)
        buf_end = sent.end
        prev_end = sent.end

        if sum(len(x) + 1 for x in buf) >= BLOCK_TARGET_CHARS:
            flush()

    flush()
    return segments


# --- entry point of the cleaning step ---------------------------
def clean_transcript(
    cues: list[RawCue],
    video_id: str,
    restorer=None,
) -> tuple[list[CleanSegment], list[dict], str | None]:
    """Returns (segments, removed_ad_spans, punctuation_backend)."""
    kept, removed = strip_ad_windows(cues)
    if restorer is not None and getattr(restorer, "backend", "none") != "none":
        sentences = restorer.restore_cues(kept)
        segments = merge_sentences_into_blocks(sentences, video_id)
        return segments, removed, restorer.backend
    segments = merge_into_blocks(kept, video_id)
    return segments, removed, None
