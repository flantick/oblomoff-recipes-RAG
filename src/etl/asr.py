"""Transcription with a local Whisper (bypassing the timedtext block).

YouTube throttles the subtitle endpoint (timedtext) hard per IP, while the media
CDN serves audio without restrictions. So we download the audio track and
transcribe it locally on the GPU — which turns out to be both more reliable and
more accurate than YouTube auto-ASR (punctuation comes out of the box, so RUPunct
is not needed on this path).

Note: the channel videos carry an auto-dubbed English track and it comes first in
the format list. The selector below explicitly prefers the original Russian one.
"""
from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from loguru import logger

from src import config
from src.etl.schemas import RawCue
from src.etl.ytdlp_common import ytdlp_network_opts

_MODEL: Any = None
_PIPELINE: Any = None

# Sentence boundary: an end mark + a space + a capital letter/quote/dash.
_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+(?=[«\"—A-ZА-ЯЁ])")


def _resolve_device() -> tuple[str, str]:
    """(device, compute_type) — honouring the explicit settings from config."""
    device = config.ASR_DEVICE
    if not device:
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # pragma: no cover - torch is always there, but do not break the run
            device = "cpu"
    compute = config.ASR_COMPUTE_TYPE or ("float16" if device == "cuda" else "int8")
    return device, compute


def get_pipeline():
    """Lazy singleton: the model is loaded once per run (~40 s)."""
    global _MODEL, _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    from faster_whisper import BatchedInferencePipeline, WhisperModel

    device, compute = _resolve_device()
    logger.info("ASR: загружаю {} на {} ({})", config.ASR_MODEL, device, compute)
    _MODEL = WhisperModel(config.ASR_MODEL, device=device, compute_type=compute)
    _PIPELINE = BatchedInferencePipeline(model=_MODEL)
    return _PIPELINE


def download_audio(video_id: str, dest_dir: Path) -> Path:
    """Downloads the original (Russian) audio track. Returns the file path."""
    from yt_dlp import YoutubeDL

    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "format": config.ASR_AUDIO_FORMAT,
        "outtmpl": str(dest_dir / "%(id)s.%(ext)s"),
        "retries": 5,
        "extractor_retries": 3,
        "noprogress": True,
        **ytdlp_network_opts(),
    }
    from yt_dlp.utils import DownloadError

    try:
        with YoutubeDL(opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
    except DownloadError as exc:
        # the media CDN hardly ever rate-limits, but if it starts to, let the
        # global --abort-after-rate-limits guard fire instead of 600 useless retries
        msg = str(exc)
        from src.etl.transcripts import RateLimited, TranscriptError

        if "429" in msg or "Too Many Requests" in msg:
            raise RateLimited(f"{video_id}: HTTP 429 на аудио") from exc
        raise TranscriptError(f"{video_id}: {msg[:200]}") from exc

    files = [p for p in dest_dir.glob(f"{video_id}.*") if p.is_file()]
    if not files:
        raise FileNotFoundError(f"{video_id}: yt-dlp не оставил аудиофайл")
    return max(files, key=lambda p: p.stat().st_size)


def transcribe_audio(path: Path) -> list[RawCue]:
    pipeline = get_pipeline()
    segments, _info = pipeline.transcribe(
        str(path),
        language=config.ASR_LANGUAGE,
        task="transcribe",
        batch_size=config.ASR_BATCH_SIZE,
        vad_filter=True,
        word_timestamps=False,
    )
    cues: list[RawCue] = []
    for s in segments:
        text = (s.text or "").strip()
        if not text:
            continue
        cues.extend(_split_into_cues(text, float(s.start), float(s.end)))
    return cues


def _split_into_cues(text: str, start: float, end: float) -> list[RawCue]:
    """Splits a long Whisper segment into sentences.

    Whisper returns pieces of 30-60 s (~600 characters) — twice the size of
    YouTube cues, which made merge_into_blocks produce blocks beyond
    BLOCK_MAX_CHARS and degraded the timecode in the output to minute
    granularity. Whisper brings its own punctuation, so we cut on sentence
    boundaries and spread the time inside a segment proportionally to length —
    the timecode accuracy drops to a couple of seconds, which is plenty for
    linking to a moment in the video.
    """
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
    if len(parts) < 2:
        return [RawCue(text=text, start=start, duration=max(end - start, 0.0))]

    total = sum(len(p) for p in parts) or 1
    span = max(end - start, 0.0)
    cues: list[RawCue] = []
    offset = start
    for part in parts:
        dur = span * len(part) / total
        cues.append(RawCue(text=part, start=offset, duration=dur))
        offset += dur
    return cues


def fetch_via_asr(video_id: str) -> tuple[list[RawCue], str, bool]:
    """Returns (cues, language_code, is_generated) — like the other backends."""
    keep_dir = config.ASR_AUDIO_DIR
    if keep_dir:
        keep_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = keep_dir
        cleanup = False
    else:
        tmp_dir = Path(tempfile.mkdtemp(prefix=f"asr_{video_id}_"))
        cleanup = True

    try:
        audio = download_audio(video_id, tmp_dir)
        cues = transcribe_audio(audio)
        if not cues:
            from src.etl.transcripts import TranscriptError

            raise TranscriptError(f"{video_id}: ASR не дал ни одного сегмента")
        if not config.ASR_KEEP_AUDIO:
            audio.unlink(missing_ok=True)
        return cues, config.ASR_LANGUAGE, True
    finally:
        if cleanup:
            shutil.rmtree(tmp_dir, ignore_errors=True)
