"""Tests for the pydantic schemas of the ETL artifacts.

This module is almost entirely bare dataclasses (by convention: schemas are
excluded from the coverage priority). We only test the real logic
(VideoTranscript.full_text), the default values other modules rely on, and
the field contract between Chunk and the code that reads it back out of a
Qdrant payload.
"""
from __future__ import annotations

from src.etl.schemas import Chunk, CleanSegment, VideoMeta, VideoTranscript
from tests.conftest import make_payload


def _meta(**kw) -> VideoMeta:
    return VideoMeta(video_id="v1", title="Title", url="https://youtu.be/v1", **kw)


def _segment(text: str, start: float, end: float) -> CleanSegment:
    return CleanSegment(
        text=text,
        start=start,
        end=end,
        timecode="00:00",
        url="https://youtu.be/v1?t=0",
    )


def _transcript(segments: list[CleanSegment]) -> VideoTranscript:
    return VideoTranscript(
        meta=_meta(),
        language="ru",
        is_generated=False,
        source="ytapi",
        raw_cues_count=len(segments),
        segments=segments,
    )


def test_full_text_joins_segments_in_order_using_text_field():
    """full_text concatenates segment.text (not timecode/url) with '\\n', in list order."""
    transcript = _transcript([
        _segment("Режем лук", 0.0, 5.0),
        _segment("Обжариваем на сковороде", 5.0, 10.0),
    ])

    assert transcript.full_text == "Режем лук\nОбжариваем на сковороде"


def test_full_text_returns_empty_string_for_no_segments():
    """An empty segments list joins into an empty string, not None or an error."""
    transcript = _transcript([])

    assert transcript.full_text == ""


def test_full_text_single_segment_has_no_newline():
    """A single segment produces its bare text with no trailing/leading newline."""
    transcript = _transcript([_segment("Только один шаг", 0.0, 3.0)])

    assert transcript.full_text == "Только один шаг"


def test_video_meta_defaults():
    """A freshly built VideoMeta defaults to is_recipe=True with empty lists and
    unset optional fields — classify_recipe() and the chunking/run.py filter both
    rely on is_recipe being True unless a classifier explicitly flips it."""
    meta = _meta()

    assert meta.is_recipe is True
    assert meta.skip_reason is None
    assert meta.playlist_ids == []
    assert meta.playlist_titles == []
    assert meta.channel is None
    assert meta.upload_date is None
    assert meta.duration is None
    assert meta.description is None


def test_video_transcript_defaults():
    """A freshly built VideoTranscript has no punctuation backend and empty
    ad-span/segment lists until the later ETL steps fill them in."""
    transcript = VideoTranscript(
        meta=_meta(),
        language="ru",
        is_generated=False,
        source="ytapi",
        raw_cues_count=0,
    )

    assert transcript.punctuation_backend is None
    assert transcript.removed_ad_spans == []
    assert transcript.segments == []


def test_chunk_defaults():
    """A freshly built Chunk has empty playlist lists until index/build.py
    attaches the video's playlist membership."""
    chunk = Chunk(
        chunk_id="v1::001",
        video_id="v1",
        title="Title",
        url="https://youtu.be/v1?t=0",
        timecode="00:00",
        start=0.0,
        end=5.0,
        chunk_index=1,
        n_chunks=3,
        section="steps",
        has_ingredients=False,
        has_steps=True,
        char_len=10,
        token_len=3,
        text="Режем лук",
    )

    assert chunk.playlist_ids == []
    assert chunk.playlist_titles == []


def test_chunk_covers_every_field_read_back_out_of_the_qdrant_payload():
    """chunking/run.py serialises chunk.model_dump() into the payload that
    index/build.py uploads, and three modules read fixed keys back out of it.

    A renamed field would not raise anywhere: retrieval/retriever.py reads via
    payload.get(key, default) and would silently substitute the default,
    retrieval/filters.py would build a filter matching nothing, and
    index/store.py would create a payload index on a field that no longer
    exists. make_payload() in tests/conftest.py already mirrors what retrieval
    reads, so it serves as the single source of truth here instead of a second
    hand-maintained list.
    """
    fields = set(Chunk.model_fields)

    assert set(make_payload()) <= fields
    filtered_and_indexed = {
        "playlist_titles",                # retrieval/filters.py intent filter
        "has_ingredients", "has_steps",   # index/store.py payload indexes
    }
    assert filtered_and_indexed <= fields


def test_video_transcript_survives_a_json_round_trip():
    """The artifact is written as data/processed/<video_id>.json and read back
    with model_validate_json in chunking/run.py, so the round trip has to be
    lossless — an unserialisable field would only surface at ETL time."""
    transcript = _transcript([
        _segment("Режем лук", 0.0, 5.0),
        _segment("Обжариваем", 5.0, 10.0),
    ])

    assert VideoTranscript.model_validate_json(transcript.model_dump_json()) == transcript


def test_full_text_stays_out_of_the_serialised_artifact():
    """full_text is a derived property, not a field: were it ever turned into
    one, every artifact on disk would carry the whole transcript twice."""
    transcript = _transcript([_segment("Режем лук", 0.0, 5.0)])

    assert "full_text" not in transcript.model_dump()
