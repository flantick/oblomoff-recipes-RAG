"""A CLI sanity check for retrieval (Step 4).

    python -m src.retrieval.retrieve "как приготовить сочный стейк"
    python -m src.retrieval.retrieve "салат с тунцом" --intent-filter --no-rerank
"""
from __future__ import annotations

import argparse
import sys

from src.config import (
    RETRIEVAL_PER_VIDEO,
    RETRIEVAL_TOKEN_BUDGET,
    RETRIEVAL_TOP_VIDEOS,
)
from src.retrieval.retriever import Retriever


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Проверка retrieval-слоя")
    p.add_argument("query")
    p.add_argument("--mode", choices=["hybrid", "dense", "sparse"], default="hybrid")
    p.add_argument("--top-videos", type=int, default=RETRIEVAL_TOP_VIDEOS)
    p.add_argument("--per-video", type=int, default=RETRIEVAL_PER_VIDEO)
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("--intent-filter", action="store_true")
    p.add_argument("--budget", type=int, default=RETRIEVAL_TOKEN_BUDGET)
    args = p.parse_args(argv)

    sys.stdout.reconfigure(encoding="utf-8")
    r = Retriever(reranker=None if args.no_rerank else "auto")
    res = r.retrieve(
        args.query,
        mode=args.mode,
        top_videos=args.top_videos,
        per_video=args.per_video,
        rerank=not args.no_rerank,
        token_budget=args.budget,
        use_intent_filter=args.intent_filter,
    )

    print(f"\nЗАПРОС: {res.query}")
    print(f"режим={res.mode} reranker={res.used_reranker} debug={res.debug}\n")
    for v in res.videos:
        print(f"▶ [{v.score:.4f}] {v.title}  ({v.video_id})")
        for pa in v.passages:
            print(f"   {pa.timecode}  {pa.url}")
            print(f"   {pa.text[:220]}…")
    print("\n--- CONTEXT ДЛЯ LLM ---")
    print(res.context)
    print("\n--- CITATIONS ---")
    for c in res.citations:
        print(f"[{c['n']}] {c['title']} — {c['url']} ({c['timecode']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
