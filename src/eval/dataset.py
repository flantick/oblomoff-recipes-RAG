"""Golden-набор для замера качества RAG (Шаг 7).

Один запрос — это вопрос пользователя и список video_id, любой из которых
считается правильным ответом. Разметка на уровне ВИДЕО, а не чанка: рецепт
растянут на весь ролик, и требовать попадания в конкретный чанк бессмысленно.

kind:
    exact       — название блюда прямо в заголовке ролика
    paraphrase  — запрос словами пользователя, а не заголовка
    descriptive — блюдо не названо, задан продукт или задача
    negative    — в корпусе этого нет, ожидаем честный found=false
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
    relevant: list[str] = Field(default_factory=list)   # video_id, любой засчитывается
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
