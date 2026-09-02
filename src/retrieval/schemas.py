"""Result structures of the retrieval layer (Step 4)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RetrievedChunk:
    chunk_id: str
    video_id: str
    title: str
    url: str
    timecode: str
    start: float
    end: float
    chunk_index: int
    n_chunks: int
    section: str
    text: str
    score: float                       # final rank (rerank or fusion)
    retriever_score: float             # original fusion/vector score


@dataclass
class RetrievedPassage:
    video_id: str
    title: str
    url: str                           # link to the beginning of the passage
    timecode: str
    start: float
    end: float
    text: str                          # stitched adjacent chunks
    chunk_ids: list[str]
    score: float


@dataclass
class RetrievedVideo:
    video_id: str
    title: str
    score: float
    passages: list[RetrievedPassage] = field(default_factory=list)


@dataclass
class RetrievalResult:
    query: str
    videos: list[RetrievedVideo]
    context: str                       # ready-made text for the LLM
    citations: list[dict]              # [{n, title, url, timecode, video_id}]
    used_reranker: bool
    mode: str
    debug: dict = field(default_factory=dict)
