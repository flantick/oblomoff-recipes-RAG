"""CLI for the end-to-end pipeline (Step 5).

    python -m src.generation.ask "как приготовить сочный стейк"
    python -m src.generation.ask "борщ" --intent-filter --json
"""
from __future__ import annotations

import argparse
import sys

from src.config import RETRIEVAL_PER_VIDEO, RETRIEVAL_TOP_VIDEOS
from src.generation.answer import answer
from src.generation.llm import LLMClient


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Вопрос -> рецепт (retrieval + vLLM)")
    p.add_argument("query")
    p.add_argument("--top-videos", type=int, default=RETRIEVAL_TOP_VIDEOS)
    p.add_argument("--per-video", type=int, default=RETRIEVAL_PER_VIDEO)
    p.add_argument("--intent-filter", action="store_true")
    p.add_argument("--json", action="store_true", help="печатать сырой JSON ответа")
    args = p.parse_args(argv)

    sys.stdout.reconfigure(encoding="utf-8")

    llm = LLMClient()
    if not llm.health():
        print("!! LLM недоступен. Подними vLLM:  docker compose up -d vllm", file=sys.stderr)
        return 2

    res = answer(
        args.query,
        llm=llm,
        top_videos=args.top_videos,
        per_video=args.per_video,
        use_intent_filter=args.intent_filter,
    )

    if args.json:
        print(res.model_dump_json(indent=2))
        return 0

    print(f"\nВОПРОС: {res.query}")
    print(f"(модель: {res.model}, reranker: {res.used_reranker})\n")
    if not res.found:
        print("Ничего подходящего в расшифровках не нашлось.")
        if res.sources:
            print("\nБлижайшие фрагменты:")
            for s in res.sources:
                print(f"  [{s.n}] {s.title} — {s.url} ({s.timecode})")
        return 0

    print(f"🍽  {res.dish}\n")
    print("Ингредиенты:")
    for i in res.ingredients:
        print(f"  • {i}")
    print("\nПриготовление:")
    for k, s in enumerate(res.steps, 1):
        print(f"  {k}. {s}")
    if res.notes:
        print(f"\nЗаметки: {res.notes}")
    if res.source:
        print(f"\nИсточник: {res.source.title} — {res.source.url} ({res.source.timecode})")
    if len(res.sources) > 1:
        print("Ещё фрагменты в контексте:")
        for s in res.sources:
            if not res.source or s.n != res.source.n:
                print(f"  [{s.n}] {s.title} — {s.url} ({s.timecode})")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
