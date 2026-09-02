"""Index search — a CLI for sanity checks and a shared function for Steps 4-5.

    python -m src.index.search "как приготовить сочный стейк" -k 5
    python -m src.index.search "маринад для шашлыка" --mode dense
"""
from __future__ import annotations

import argparse
import sys

from qdrant_client import models

from src.index.embedder import BGEM3Embedder
from src.index.store import VectorStore


def make_filter(
    *,
    video_id: str | None = None,
    section: str | None = None,
    has_ingredients: bool | None = None,
) -> models.Filter | None:
    must: list[models.FieldCondition] = []
    if video_id:
        must.append(models.FieldCondition(key="video_id", match=models.MatchValue(value=video_id)))
    if section:
        must.append(models.FieldCondition(key="section", match=models.MatchValue(value=section)))
    if has_ingredients is not None:
        must.append(
            models.FieldCondition(key="has_ingredients", match=models.MatchValue(value=has_ingredients))
        )
    return models.Filter(must=must) if must else None


def search(store: VectorStore, embedder: BGEM3Embedder, query: str, *, k: int = 5,
           mode: str = "hybrid", query_filter: models.Filter | None = None) -> list[models.ScoredPoint]:
    emb = embedder.encode_queries([query])[0]
    return store.search(emb, k=k, mode=mode, query_filter=query_filter)


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Поиск по индексу рецептов")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--mode", choices=["hybrid", "dense", "sparse"], default="hybrid")
    p.add_argument("--qdrant-path", default=None)
    args = p.parse_args(argv)

    store = VectorStore(path=args.qdrant_path)
    embedder = BGEM3Embedder()
    hits = search(store, embedder, args.query, k=args.k, mode=args.mode)

    sys.stdout.reconfigure(encoding="utf-8")
    for h in hits:
        pl = h.payload or {}
        print(f"\n[{h.score:.4f}] {pl.get('title')}  ({pl.get('timecode')})  {pl.get('section')}")
        print(f"  {pl.get('url')}")
        print(f"  {pl.get('text', '')[:280]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
