"""Schemas of the structured recipe answer (Step 5)."""
from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator


class SourceRef(BaseModel):
    n: int
    video_id: str
    title: str
    url: str                              # video link with a timecode
    timecode: str


class LLMRecipe(BaseModel):
    """What the model is required to return as JSON."""
    found: bool
    dish: str = ""
    ingredients: list[str] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
    source_n: int | None = None           # number [n] of the main fragment
    notes: str | None = None

    @field_validator("source_n", mode="before")
    @classmethod
    def _coerce_source_n(cls, v):
        """Accepts [3, 4], "[1]" and "фрагмент 2" as well as a plain 3.

        The model regularly cites several fragments here instead of one. A
        strict int failed validation, and answer() then discarded a fully
        extracted recipe over the format of a citation number — to the user that
        looked exactly like "there is no answer in the transcripts".
        """
        if v is None or isinstance(v, bool):
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, (list, tuple, set)):
            return cls._coerce_source_n(next(iter(v), None))
        if isinstance(v, float):
            return int(v)
        if isinstance(v, str):
            m = re.search(r"\d+", v)
            return int(m.group()) if m else None
        return None


class RecipeAnswer(BaseModel):
    """The final answer of the pipeline (served by FastAPI in Step 6)."""
    query: str
    found: bool
    dish: str = ""
    ingredients: list[str] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
    notes: str | None = None
    source: SourceRef | None = None       # the main source
    sources: list[SourceRef] = Field(default_factory=list)  # every cited fragment
    model: str = ""
    used_reranker: bool = False
