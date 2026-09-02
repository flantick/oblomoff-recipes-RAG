"""The golden set used to measure RAG quality (Step 7).

One entry is a user question plus the list of video_ids that count as a correct
answer. Labelling is done at the VIDEO level, not the chunk level: a recipe is
stretched over the whole clip, so demanding a hit on one particular chunk would
be meaningless.

kind:
    exact       — the dish name appears in the video title
    paraphrase  — the query uses the user's words, not the title's
    descriptive — the dish is not named; an ingredient or a task is given
    negative    — the corpus has no such recipe, we expect an honest found=false
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from src.config import ROOT_DIR

DEFAULT_GOLDEN = ROOT_DIR / "data" / "eval" / "golden.jsonl"


class GoldenItem(BaseModel):
    id: str
    query: str
    relevant: list[str] = Field(default_factory=list)   # video_id, any of them counts
    kind: str = "exact"
    expect_found: bool = True
    note: str = ""

    @property
    def is_negative(self) -> bool:
        return not self.relevant


def load_golden(path: Path | str = DEFAULT_GOLDEN, kinds: list[str] | None = None) -> list[GoldenItem]:
    rows = [
        GoldenItem.model_validate_json(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if kinds:
        rows = [r for r in rows if r.kind in kinds]
    return rows


def dump_report(path: Path | str, payload: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
