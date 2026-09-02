"""Pydantic schemas of the ETL artifacts (Step 1)."""
from __future__ import annotations

from pydantic import BaseModel, Field


class VideoMeta(BaseModel):
    video_id: str
    title: str
    url: str
    playlist_ids: list[str] = Field(default_factory=list)
    playlist_titles: list[str] = Field(default_factory=list)
    channel: str | None = None
    upload_date: str | None = None          # YYYYMMDD
    duration: int | None = None             # seconds
    description: str | None = None
    is_recipe: bool = True
    skip_reason: str | None = None


class RawCue(BaseModel):
    """A raw subtitle cue."""
    text: str
    start: float                            # seconds from the start of the video
    duration: float


class CleanSegment(BaseModel):
    """A cleaned semantic block bound to a timecode."""
    text: str
    start: float
    end: float
    timecode: str                          # "MM:SS" or "H:MM:SS"
    url: str                               # video link with ?t=<start>


class VideoTranscript(BaseModel):
    """The resulting data/processed/<video_id>.json artifact."""
    meta: VideoMeta
    language: str
    is_generated: bool
    source: str                            # "ytapi" | "ytdlp" | "asr"
    punctuation_backend: str | None = None  # "rupunct" | "silero" | None
    raw_cues_count: int
    removed_ad_spans: list[dict] = Field(default_factory=list)
    segments: list[CleanSegment] = Field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n".join(s.text for s in self.segments)


class Chunk(BaseModel):
    """The unit of indexing for the vector database (Step 2)."""
    chunk_id: str                          # "<video_id>::<index:03d>"
    video_id: str
    title: str
    url: str                               # video link with ?t=<start>
    timecode: str
    start: float
    end: float
    chunk_index: int
    n_chunks: int
    section: str                           # intro|ingredients|steps|outro|other|mixed
    has_ingredients: bool
    has_steps: bool
    char_len: int
    token_len: int
    playlist_ids: list[str] = Field(default_factory=list)
    playlist_titles: list[str] = Field(default_factory=list)
    text: str
