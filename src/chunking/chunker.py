"""Splitting a video transcript into chunks for indexing (Step 2).

The strategy is tailored to video recipes:
- we work at the level of already punctuated sentence-segments (Step 1.5), each
  carrying its own timecode;
- segments accumulate up to CHUNK_TARGET_TOKENS, with CHUNK_MAX_TOKENS as the
  hard limit;
- we prefer to break on a "good" boundary: a section change (intro→ingredients→
  steps→outro) or the start of a new step ("затем", "далее", …);
- a dense ingredient list is never split: while the ingredients section runs and
  the limit is not reached, no break happens;
- adjacent chunks overlap by CHUNK_OVERLAP_TOKENS, except across section borders;
- a tail shorter than CHUNK_MIN_TOKENS is glued to the previous chunk.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.config import (
    CHUNK_MAX_TOKENS,
    CHUNK_MIN_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    CHUNK_TARGET_TOKENS,
)
from src.chunking.sections import label_segments, starts_new_step
from src.chunking.tokens import TokenCounter
from src.etl.schemas import Chunk, VideoTranscript


@dataclass
class ChunkConfig:
    target: int = CHUNK_TARGET_TOKENS
    max: int = CHUNK_MAX_TOKENS
    min: int = CHUNK_MIN_TOKENS
    overlap: int = CHUNK_OVERLAP_TOKENS


@dataclass
class _Seg:
    text: str
    start: float
    end: float
    label: str
    tokens: int


def _fmt_timecode(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def _dominant_label(labels: list[str]) -> str:
    core = [x for x in labels if x not in {"other"}] or labels
    counts: dict[str, int] = {}
    for x in core:
        counts[x] = counts.get(x, 0) + 1
    top = max(counts.values())
    winners = sorted(k for k, v in counts.items() if v == top)
    return winners[0] if len(winners) == 1 else "mixed"


def _should_break(cur_tokens: int, cur_last: _Seg, nxt: _Seg, cfg: ChunkConfig) -> bool:
    if cur_tokens >= cfg.max:
        return True
    if cur_tokens < cfg.target:
        return False
    # the target is reached — look for a good boundary
    if nxt.label != cur_last.label:
        return True
    if starts_new_step(nxt.text):
        return True
    # deep inside an ingredient list — stretch up to the limit
    if cur_last.label == "ingredients" and nxt.label == "ingredients":
        return False
    return True


def _overlap_tail(segs: list[_Seg], overlap: int) -> list[_Seg]:
    if overlap <= 0:
        return []
    picked: list[_Seg] = []
    total = 0
    for s in reversed(segs):
        if picked and total >= overlap:
            break
        if len(picked) >= 2:
            break
        picked.append(s)
        total += s.tokens
    return list(reversed(picked))


def chunk_transcript(vt: VideoTranscript, counter: TokenCounter, cfg: ChunkConfig | None = None) -> list[Chunk]:
    cfg = cfg or ChunkConfig()
    if not vt.segments:
        return []

    labels = label_segments([s.text for s in vt.segments])
    segs = [
        _Seg(text=s.text, start=s.start, end=s.end, label=lab, tokens=counter.count(s.text))
        for s, lab in zip(vt.segments, labels)
    ]

    groups: list[list[_Seg]] = []
    cur: list[_Seg] = []
    cur_tokens = 0
    for seg in segs:
        if cur and _should_break(cur_tokens, cur[-1], seg, cfg):
            section_change = seg.label != cur[-1].label
            groups.append(cur)
            tail = [] if section_change else _overlap_tail(cur, cfg.overlap)
            cur = list(tail)
            cur_tokens = sum(t.tokens for t in cur)
        cur.append(seg)
        cur_tokens += seg.tokens
    if cur:
        groups.append(cur)

    # glue a too short tail onto the previous group
    if len(groups) >= 2 and sum(s.tokens for s in groups[-1]) < cfg.min:
        # without the duplicates that could have come in through the overlap
        prev_ids = {id(s) for s in groups[-2]}
        groups[-2].extend(s for s in groups[-1] if id(s) not in prev_ids)
        groups.pop()

    return _materialize(vt, groups, counter)


def _materialize(vt: VideoTranscript, groups: list[list[_Seg]], counter: TokenCounter) -> list[Chunk]:
    chunks: list[Chunk] = []
    n = len(groups)
    for idx, g in enumerate(groups):
        text = " ".join(s.text for s in g).strip()
        start = g[0].start
        end = g[-1].end
        labels = [s.label for s in g]
        chunks.append(
            Chunk(
                chunk_id=f"{vt.meta.video_id}::{idx:03d}",
                video_id=vt.meta.video_id,
                title=vt.meta.title,
                url=f"https://youtu.be/{vt.meta.video_id}?t={int(start)}",
                timecode=_fmt_timecode(start),
                start=round(start, 2),
                end=round(end, 2),
                chunk_index=idx,
                n_chunks=n,
                section=_dominant_label(labels),
                has_ingredients="ingredients" in labels,
                has_steps="steps" in labels,
                char_len=len(text),
                token_len=counter.count(text),
                playlist_ids=vt.meta.playlist_ids,
                playlist_titles=vt.meta.playlist_titles,
                text=text,
            )
        )
    return chunks
