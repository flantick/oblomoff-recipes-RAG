"""Schemas of the structured recipe answer (Step 5)."""
from __future__ import annotations

from pydantic import BaseModel, Field


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
