"""Chunking of every processed transcript (Step 2).

Usage:
    python -m src.chunking.run
    python -m src.chunking.run --tokenizer intfloat/multilingual-e5-base
    python -m src.chunking.run --target 400 --max 550 --overlap 60 --overwrite

Input:  data/processed/<video_id>.json   (VideoTranscript, Step 1/1.5)
Output: data/chunks/<video_id>.json      (list[Chunk])
        data/chunks/chunks.jsonl         (every chunk in one file — for Step 3)
        data/chunks/manifest.json        (summary)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger
from pydantic import ValidationError
from tqdm import tqdm

from src.chunking.chunker import ChunkConfig, chunk_transcript
from src.chunking.tokens import TokenCounter
from src.config import DATA_PROCESSED_DIR, ROOT_DIR
from src.etl.schemas import VideoTranscript

DATA_CHUNKS_DIR = ROOT_DIR / "data" / "chunks"


def _dump_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Чанкинг транскриптов oblomoffood")
    p.add_argument("--in", dest="in_dir", default=str(DATA_PROCESSED_DIR))
    p.add_argument("--out", dest="out_dir", default=str(DATA_CHUNKS_DIR))
    p.add_argument("--tokenizer", default=None, help="HF-модель для точного счёта токенов (опц.)")
    p.add_argument("--target", type=int, default=None)
    p.add_argument("--max", type=int, default=None)
    p.add_argument("--min", type=int, default=None)
    p.add_argument("--overlap", type=int, default=None)
    p.add_argument("--include-non-recipe", action="store_true", help="не пропускать is_recipe=False")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {level: <7} | {message}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = ChunkConfig()
    for k in ("target", "max", "min", "overlap"):
        v = getattr(args, k)
        if v is not None:
            setattr(cfg, k, v)

    counter = TokenCounter(args.tokenizer)
    if args.tokenizer:
        logger.info("Счёт токенов: {}", args.tokenizer)

    files = sorted(Path(args.in_dir).glob("*.json"))
    files = [f for f in files if f.name not in {"manifest.json", "failures.json"}]

    manifest: list[dict] = []
    all_chunks: list[dict] = []
    total = 0

    for f in tqdm(files, desc="chunking", unit="vid"):
        try:
            vt = VideoTranscript.model_validate_json(f.read_text(encoding="utf-8"))
        except ValidationError as exc:
            logger.warning("Пропуск {}: {}", f.name, exc.errors()[:1])
            continue

        if not vt.meta.is_recipe and not args.include_non_recipe:
            manifest.append({"video_id": vt.meta.video_id, "status": "skip_non_recipe"})
            continue

        out_path = out_dir / f"{vt.meta.video_id}.json"
        if out_path.exists() and not args.overwrite:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            all_chunks.extend(existing)
            total += len(existing)
            manifest.append({"video_id": vt.meta.video_id, "status": "skip_exists", "chunks": len(existing)})
            continue

        chunks = chunk_transcript(vt, counter, cfg)
        payload = [c.model_dump() for c in chunks]
        _dump_json(out_path, payload)
        all_chunks.extend(payload)
        total += len(chunks)

        toks = [c.token_len for c in chunks] or [0]
        manifest.append(
            {
                "video_id": vt.meta.video_id,
                "status": "ok",
                "chunks": len(chunks),
                "tok_min": min(toks),
                "tok_avg": round(sum(toks) / len(toks)),
                "tok_max": max(toks),
                "sections": sorted({c.section for c in chunks}),
            }
        )

    with (out_dir / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for c in all_chunks:
            fh.write(json.dumps(c, ensure_ascii=False, default=str) + "\n")
    _dump_json(out_dir / "manifest.json", manifest)

    ok = sum(1 for m in manifest if m["status"] == "ok")
    logger.info("Готово. видео_ok={} / чанков всего={} / файл={}", ok, total, out_dir / "chunks.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
