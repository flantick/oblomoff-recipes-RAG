"""Vectorising the chunks and loading them into Qdrant (Step 3).

Preconditions:
    docker compose up -d qdrant
    python -m src.chunking.run            # creates data/chunks/chunks.jsonl

Usage:
    python -m src.index.build --recreate
    python -m src.index.build --limit 200            # partial load for a test
    python -m src.index.build --qdrant-path data/qdrant   # embedded mode, no docker
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from src.config import DENSE_VECTOR_SIZE, QDRANT_COLLECTION, QDRANT_URL, ROOT_DIR
from src.index.embedder import BGEM3Embedder, passage_text
from src.index.store import VectorStore

DEFAULT_CHUNKS = ROOT_DIR / "data" / "chunks" / "chunks.jsonl"


def non_negative_int(value: str) -> int:
    """argparse type for --limit: a slice bound, so a negative one is a typo."""
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError(f"ожидается неотрицательное число, получено {value}")
    return n


def load_chunks(path: Path, limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit is not None else rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Векторизация чанков -> Qdrant")
    p.add_argument("--chunks", default=str(DEFAULT_CHUNKS))
    p.add_argument("--collection", default=QDRANT_COLLECTION)
    p.add_argument("--qdrant-url", default=QDRANT_URL)
    p.add_argument("--qdrant-path", default=None, help="встроенный Qdrant в папку (без докера)")
    p.add_argument("--recreate", action="store_true", help="пересоздать коллекцию с нуля")
    p.add_argument("--batch-size", type=int, default=12, help="батч эмбеддера")
    p.add_argument("--upsert-batch", type=int, default=128)
    p.add_argument("--device", default=None, help="cuda | cpu (по умолчанию авто)")
    p.add_argument("--no-fp16", action="store_true")
    p.add_argument("--limit", type=non_negative_int, default=None, help="загрузить только первые N чанков")
    args = p.parse_args(argv)

    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {level: <7} | {message}")

    chunks = load_chunks(Path(args.chunks), args.limit)
    if not chunks:
        logger.error("Нет чанков в {}", args.chunks)
        return 1
    logger.info("Чанков к загрузке: {}", len(chunks))

    store = VectorStore(url=args.qdrant_url, collection=args.collection, path=args.qdrant_path)
    store.ensure_collection(dense_size=DENSE_VECTOR_SIZE, recreate=args.recreate)

    embedder = BGEM3Embedder(
        device=args.device,
        use_fp16=not args.no_fp16,
        batch_size=args.batch_size,
    )

    written = 0
    step = max(args.batch_size * 8, args.upsert_batch)
    for i in tqdm(range(0, len(chunks), step), desc="embed+upsert", unit="batch"):
        part = chunks[i:i + step]
        embs = embedder.encode_passages([passage_text(c) for c in part])
        written += store.upsert(part, embs, batch=args.upsert_batch)

    total = store.count()
    logger.info("Готово. upsert={} / точек в коллекции={}", written, total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
