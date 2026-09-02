"""ETL orchestrator (Step 1).

Usage:
    python -m src.etl.pipeline --limit 3            # smoke test: 3 videos per playlist
    python -m src.etl.pipeline                      # full run
    python -m src.etl.pipeline --drop-non-recipe --full-meta

Artifacts:
    data/raw/<video_id>.json         — raw subtitle cues
    data/processed/<video_id>.json   — VideoTranscript (meta + cleaned blocks)
    data/processed/manifest.json     — summary over all videos
    data/processed/failures.json     — what failed to download and why
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from src import config
from src.config import (
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    ETL_SLEEP_SECONDS,
    PLAYLISTS,
)
from src.etl.cleaning import classify_recipe, clean_transcript
from src.etl.playlists import fetch_playlist_entries, fetch_video_meta
from src.etl.schemas import RawCue, VideoMeta, VideoTranscript
from src.etl.transcripts import RateLimited, TranscriptError, get_transcript


def _dump_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def collect_videos(playlists: list[str], limit: int | None) -> dict[str, VideoMeta]:
    """Collects the unique videos across all playlists (handling overlaps)."""
    videos: dict[str, VideoMeta] = {}
    for url in playlists:
        entries = fetch_playlist_entries(url)
        if limit:
            entries = entries[:limit]
        for e in entries:
            vid = e["video_id"]
            if vid in videos:
                videos[vid].playlist_ids.append(e["playlist_id"])
                videos[vid].playlist_titles.append(e["playlist_title"])
                continue
            videos[vid] = VideoMeta(
                video_id=vid,
                title=e["title"],
                url=f"https://www.youtube.com/watch?v={vid}",
                playlist_ids=[e["playlist_id"]],
                playlist_titles=[e["playlist_title"]],
            )
        time.sleep(ETL_SLEEP_SECONDS)
    logger.info("Всего уникальных видео: {}", len(videos))
    return videos


def process_video(
    meta: VideoMeta,
    *,
    source: str,
    full_meta: bool,
    drop_non_recipe: bool,
    overwrite: bool,
    restorer=None,
) -> dict:
    out_path = DATA_PROCESSED_DIR / f"{meta.video_id}.json"
    if out_path.exists() and not overwrite:
        return {"video_id": meta.video_id, "status": "skip_exists"}

    if full_meta:
        fetched = fetch_video_meta(meta.video_id, fallback_title=meta.title)
        fetched.playlist_ids = meta.playlist_ids
        fetched.playlist_titles = meta.playlist_titles
        meta = fetched

    meta = classify_recipe(meta)
    if not meta.is_recipe and drop_non_recipe:
        _dump_json(DATA_PROCESSED_DIR / f"{meta.video_id}.skip.json", meta.model_dump())
        return {"video_id": meta.video_id, "status": "skip_non_recipe", "reason": meta.skip_reason}

    try:
        cues, lang, is_generated, used_source = get_transcript(meta.video_id, source=source)
    except RateLimited as exc:
        return {"video_id": meta.video_id, "status": "rate_limited", "error": str(exc)}
    except TranscriptError as exc:
        return {"video_id": meta.video_id, "status": "no_transcript", "error": str(exc)}
    except Exception as exc:  # network/parsing — do not kill the whole run
        logger.exception("Ошибка на видео {}", meta.video_id)
        return {"video_id": meta.video_id, "status": "error", "error": repr(exc)}

    _dump_json(
        DATA_RAW_DIR / f"{meta.video_id}.json",
        {"video_id": meta.video_id, "language": lang, "cues": [c.model_dump() for c in cues]},
    )

    segments, removed, punct_backend = clean_transcript(cues, meta.video_id, restorer=restorer)
    transcript = VideoTranscript(
        meta=meta,
        language=lang,
        is_generated=is_generated,
        source=used_source,
        punctuation_backend=punct_backend,
        raw_cues_count=len(cues),
        removed_ad_spans=removed,
        segments=segments,
    )
    _dump_json(out_path, transcript.model_dump())
    return {
        "video_id": meta.video_id,
        "status": "ok",
        "is_recipe": meta.is_recipe,
        "segments": len(segments),
        "raw_cues": len(cues),
        "ad_spans_removed": len(removed),
        "source": used_source,
        "punct": punct_backend,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ETL: субтитры oblomoffood -> очищенные блоки")
    parser.add_argument("--limit", type=int, default=None, help="сколько видео брать с каждого плейлиста")
    parser.add_argument("--source", choices=["auto", "ytapi", "ytdlp", "asr"], default="ytdlp",
                        help="asr — качать аудио и распознавать локально (обходит блокировку субтитров)")
    parser.add_argument("--full-meta", action="store_true", help="тянуть полные метаданные каждого видео (медленно)")
    parser.add_argument("--drop-non-recipe", action="store_true", help="не скачивать обзоры доставок")
    parser.add_argument("--overwrite", action="store_true", help="перезаписывать уже обработанные видео")
    parser.add_argument("--sleep", type=float, default=ETL_SLEEP_SECONDS, help="пауза между видео, сек")
    parser.add_argument("--cookies-from-browser", default=None,
                        help="chrome|firefox|edge|brave — куки залогиненного браузера (обход рейтлимитов YouTube)")
    parser.add_argument("--cookies-file", default=None, help="путь к cookies.txt (Netscape формат)")
    parser.add_argument("--abort-after-rate-limits", type=int, default=10,
                        help="прервать прогон после N подряд rate_limited видео")
    parser.add_argument("--pause-every", type=int, default=0,
                        help="каждые N успешных запросов делать длинную паузу (антиблок YouTube)")
    parser.add_argument("--pause-seconds", type=float, default=120,
                        help="длительность паузы для --pause-every")
    parser.add_argument(
        "--punct",
        choices=["auto", "rupunct", "silero", "none"],
        default="auto",
        help="восстановление пунктуации: auto=rupunct→silero→none",
    )
    parser.add_argument("--punct-model", default="RUPunct/RUPunct_big", help="HF-модель для бэкенда rupunct")
    parser.add_argument("--punct-device", default=None, help="cuda | cpu (по умолчанию — авто)")
    args = parser.parse_args(argv)

    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {level: <7} | {message}")

    if args.cookies_from_browser:
        config.YTDLP_COOKIES_FROM_BROWSER = args.cookies_from_browser
    if args.cookies_file:
        config.YTDLP_COOKIES_FILE = args.cookies_file
    if config.YTDLP_COOKIES_FROM_BROWSER or config.YTDLP_COOKIES_FILE:
        logger.info("yt-dlp куки: {}", args.cookies_from_browser or args.cookies_file or "из окружения")

    # Whisper already returns text with punctuation and capitals — RUPunct only
    # wastes time here and degrades the result by re-punctuating it.
    if args.source == "asr" and args.punct == "auto":
        logger.info("source=asr: пунктуация не нужна (Whisper даёт её сам) — --punct none")
        args.punct = "none"

    restorer = None
    if args.punct != "none":
        from src.etl.punctuation import PunctuationRestorer

        restorer = PunctuationRestorer(
            backend=args.punct,
            model_name=args.punct_model,
            device=args.punct_device,
        )
        logger.info("Пунктуация: бэкенд = {}", restorer.backend)

    videos = collect_videos(PLAYLISTS, args.limit)
    manifest: list[dict] = []
    failures: list[dict] = []

    consecutive_rl = 0
    aborted = False
    fetched_since_pause = 0
    for meta in tqdm(list(videos.values()), desc="videos", unit="vid"):
        if args.pause_every and fetched_since_pause >= args.pause_every:
            logger.info("Пауза {}s после {} запросов (антиблок)", args.pause_seconds, fetched_since_pause)
            time.sleep(args.pause_seconds)
            fetched_since_pause = 0
        res = process_video(
            meta,
            source=args.source,
            full_meta=args.full_meta,
            drop_non_recipe=args.drop_non_recipe,
            overwrite=args.overwrite,
            restorer=restorer,
        )
        manifest.append(res)
        if res["status"] in {"no_transcript", "error", "rate_limited"}:
            failures.append(res)

        if res["status"] == "rate_limited":
            consecutive_rl += 1
            if consecutive_rl >= args.abort_after_rate_limits:
                logger.error(
                    "{} rate_limited подряд — YouTube блокирует. Прерываю. "
                    "Запусти позже с --cookies-from-browser (докачает недостающее).",
                    consecutive_rl,
                )
                aborted = True
                break
        elif res["status"] != "skip_exists":
            consecutive_rl = 0

        if res["status"] not in {"skip_exists"}:
            fetched_since_pause += 1
            time.sleep(args.sleep)

    _dump_json(DATA_PROCESSED_DIR / "manifest.json", manifest)
    _dump_json(DATA_PROCESSED_DIR / "failures.json", failures)

    ok = sum(1 for m in manifest if m["status"] == "ok")
    rl = sum(1 for m in manifest if m["status"] == "rate_limited")
    logger.info("{} ok={} / всего={} / ошибок={} (rate_limited={})",
                "ПРЕРВАНО." if aborted else "Готово.", ok, len(manifest), len(failures), rl)
    return 3 if aborted else 0


if __name__ == "__main__":
    raise SystemExit(main())
