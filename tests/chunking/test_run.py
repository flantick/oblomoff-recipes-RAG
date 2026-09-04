"""Tests for src/chunking/run.py: main(argv) is the real orchestration logic
(skip-existing, overwrite, non-recipe filtering, manifest bookkeeping, and the
combined chunks.jsonl), not a thin argparse wrapper, so it is exercised
end-to-end by calling main(argv) with real files under tmp_path.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from loguru import logger

from src.chunking.run import main
from src.etl.schemas import CleanSegment, VideoMeta, VideoTranscript

# A short line whose section heuristic (src/chunking/sections.py) reliably
# labels it "steps": two step-verb stems ("обжар", "посол") and no ingredient
# markers. Kept as a module constant so every test that needs a "steps"
# segment agrees on the same text/label/heuristic-token-count triple.
STEP_TEXT = "Обжарьте лук на сковороде и посолите блюдо"
STEP_TOKENS = 17  # TokenCounter() heuristic: round(len(STEP_TEXT) / 2.5)


# loguru handler isolation is handled globally in tests/conftest.py: main()
# calls logger.remove(), which stops the process-wide sinks, and a live one has
# to be rebuilt rather than the stopped objects restored.


# --- local factories (deliberately not imported from test_chunker.py) -
def make_meta(
    *,
    video_id: str = "vid1",
    title: str = "Рецепт борща",
    is_recipe: bool = True,
) -> VideoMeta:
    return VideoMeta(
        video_id=video_id,
        title=title,
        url=f"https://youtu.be/{video_id}",
        is_recipe=is_recipe,
    )


def make_transcript(
    texts_with_times: list[tuple[str, float, float]],
    *,
    video_id: str = "vid1",
    title: str = "Рецепт борща",
    is_recipe: bool = True,
) -> VideoTranscript:
    segments = [
        CleanSegment(text=t, start=s, end=e, timecode="00:00", url="")
        for t, s, e in texts_with_times
    ]
    return VideoTranscript(
        meta=make_meta(video_id=video_id, title=title, is_recipe=is_recipe),
        language="ru",
        is_generated=False,
        source="ytapi",
        raw_cues_count=len(segments),
        segments=segments,
    )


def write_transcript(path: Path, vt: VideoTranscript) -> None:
    path.write_text(vt.model_dump_json(), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def manifest_of(out_dir: Path) -> list[dict]:
    return json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))


# =======================================================================
# empty input / output directory handling
# =======================================================================
def test_empty_input_dir_creates_empty_chunks_and_manifest(tmp_path):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()

    code = main(["--in", str(in_dir), "--out", str(out_dir)])

    assert code == 0
    assert (out_dir / "chunks.jsonl").read_text(encoding="utf-8") == ""
    assert manifest_of(out_dir) == []


def test_output_dir_is_created_when_missing(tmp_path):
    """out_dir (and any missing parent) does not need to pre-exist."""
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    out_dir = tmp_path / "nested" / "out"
    assert not out_dir.exists()

    main(["--in", str(in_dir), "--out", str(out_dir)])

    assert out_dir.is_dir()


# =======================================================================
# a single valid transcript -> ok status, per-video file, combined jsonl
# =======================================================================
def test_single_valid_transcript_is_chunked_and_recorded_as_ok(tmp_path):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    vt = make_transcript([(STEP_TEXT, 0.0, 5.0)], video_id="rec1", title="Борщ")
    write_transcript(in_dir / "rec1.json", vt)

    code = main(["--in", str(in_dir), "--out", str(out_dir)])

    assert code == 0
    per_video = json.loads((out_dir / "rec1.json").read_text(encoding="utf-8"))
    assert len(per_video) == 1
    chunk = per_video[0]
    assert chunk["chunk_id"] == "rec1::000"
    assert chunk["video_id"] == "rec1"
    assert chunk["title"] == "Борщ"
    assert chunk["text"] == STEP_TEXT
    assert chunk["token_len"] == STEP_TOKENS

    jsonl_rows = read_jsonl(out_dir / "chunks.jsonl")
    assert jsonl_rows == per_video

    manifest = manifest_of(out_dir)
    assert manifest == [
        {
            "video_id": "rec1",
            "status": "ok",
            "chunks": 1,
            "tok_min": STEP_TOKENS,
            "tok_avg": STEP_TOKENS,
            "tok_max": STEP_TOKENS,
            "sections": ["steps"],
        }
    ]


# =======================================================================
# manifest.json / failures.json in the INPUT dir are ETL leftovers, not
# transcripts, and must be filtered out by name before parsing
# =======================================================================
def test_manifest_and_failures_json_in_input_dir_are_not_treated_as_transcripts(tmp_path):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()

    real = make_transcript([(STEP_TEXT, 0.0, 5.0)], video_id="real1")
    write_transcript(in_dir / "real1.json", real)

    # If the filename filter (run.py glob) were missing, these two would be
    # parsed as perfectly valid transcripts and show up as extra "ok" entries.
    decoy_a = make_transcript([(STEP_TEXT, 0.0, 5.0)], video_id="decoy_manifest")
    write_transcript(in_dir / "manifest.json", decoy_a)
    decoy_b = make_transcript([(STEP_TEXT, 0.0, 5.0)], video_id="decoy_failures")
    write_transcript(in_dir / "failures.json", decoy_b)

    main(["--in", str(in_dir), "--out", str(out_dir)])

    manifest = manifest_of(out_dir)
    assert [m["video_id"] for m in manifest] == ["real1"]


# =======================================================================
# a transcript file that is invalid JSON / fails VideoTranscript validation
# =======================================================================
def test_invalid_transcript_file_is_skipped_and_others_still_processed(tmp_path, capsys):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()

    (in_dir / "bad.json").write_text("not valid json {", encoding="utf-8")
    good = make_transcript([(STEP_TEXT, 0.0, 5.0)], video_id="good1")
    write_transcript(in_dir / "good.json", good)

    code = main(["--in", str(in_dir), "--out", str(out_dir)])

    assert code == 0
    manifest = manifest_of(out_dir)
    assert [m["video_id"] for m in manifest] == ["good1"]
    assert not (out_dir / "bad.json").exists()
    # the warning branch (run.py: `logger.warning("Пропуск {}: ...")`) fires
    err = capsys.readouterr().err
    assert "bad.json" in err


# =======================================================================
# is_recipe filtering
# =======================================================================
def test_non_recipe_video_is_skipped_by_default(tmp_path):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    vt = make_transcript([(STEP_TEXT, 0.0, 5.0)], video_id="notrecipe1", is_recipe=False)
    write_transcript(in_dir / "notrecipe1.json", vt)

    main(["--in", str(in_dir), "--out", str(out_dir)])

    manifest = manifest_of(out_dir)
    assert manifest == [{"video_id": "notrecipe1", "status": "skip_non_recipe"}]
    assert not (out_dir / "notrecipe1.json").exists()
    assert (out_dir / "chunks.jsonl").read_text(encoding="utf-8") == ""


def test_include_non_recipe_flag_processes_the_video_anyway(tmp_path):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    vt = make_transcript([(STEP_TEXT, 0.0, 5.0)], video_id="notrecipe1", is_recipe=False)
    write_transcript(in_dir / "notrecipe1.json", vt)

    main(["--in", str(in_dir), "--out", str(out_dir), "--include-non-recipe"])

    manifest = manifest_of(out_dir)
    assert manifest[0]["status"] == "ok"
    assert (out_dir / "notrecipe1.json").exists()


# =======================================================================
# skip_exists vs --overwrite
# =======================================================================
def test_skip_exists_rereads_existing_chunks_into_the_combined_jsonl(tmp_path):
    """status=skip_exists must not mean the video's chunks are missing from
    chunks.jsonl: run.py reads the existing per-video file back in and
    extends all_chunks with it (run.py: `all_chunks.extend(existing)`)."""
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()

    vt = make_transcript([(STEP_TEXT, 0.0, 5.0)], video_id="vid1")
    write_transcript(in_dir / "vid1.json", vt)

    # A pre-existing chunk file from an earlier run. Its content is a marker
    # unrelated to what chunk_transcript() would produce, so any trace of it
    # in the output proves the file was reread rather than reconstructed.
    existing_chunks = [{"chunk_id": "vid1::pre", "marker": "pre-existing-chunk"}]
    (out_dir / "vid1.json").write_text(json.dumps(existing_chunks), encoding="utf-8")

    main(["--in", str(in_dir), "--out", str(out_dir)])

    # the file on disk was left untouched (not regenerated)
    assert json.loads((out_dir / "vid1.json").read_text(encoding="utf-8")) == existing_chunks

    manifest = manifest_of(out_dir)
    assert manifest == [{"video_id": "vid1", "status": "skip_exists", "chunks": 1}]

    jsonl_rows = read_jsonl(out_dir / "chunks.jsonl")
    assert jsonl_rows == existing_chunks


def test_overwrite_flag_regenerates_an_existing_output_file(tmp_path):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()

    vt = make_transcript([(STEP_TEXT, 0.0, 5.0)], video_id="vid1")
    write_transcript(in_dir / "vid1.json", vt)

    existing_chunks = [{"chunk_id": "vid1::pre", "marker": "pre-existing-chunk"}]
    (out_dir / "vid1.json").write_text(json.dumps(existing_chunks), encoding="utf-8")

    main(["--in", str(in_dir), "--out", str(out_dir), "--overwrite"])

    regenerated = json.loads((out_dir / "vid1.json").read_text(encoding="utf-8"))
    assert regenerated != existing_chunks
    assert regenerated[0]["text"] == STEP_TEXT

    manifest = manifest_of(out_dir)
    assert manifest[0]["status"] == "ok"


# =======================================================================
# --target/--max/--min/--overlap actually drive ChunkConfig
# =======================================================================
def test_target_and_max_flags_change_the_number_of_chunks(tmp_path):
    """target=max=1 forces a break after every segment (cur_tokens >= max is
    unconditional), while a target/max far above the whole text's token count
    keeps everything in a single chunk — proving the CLI flags reach
    ChunkConfig rather than being ignored."""
    texts = [("Раз", 0.0, 1.0), ("Два", 1.0, 2.0), ("Три", 2.0, 3.0), ("Четыре", 3.0, 4.0)]

    in_tight = tmp_path / "in_tight"
    out_tight = tmp_path / "out_tight"
    in_tight.mkdir()
    write_transcript(in_tight / "vid1.json", make_transcript(texts, video_id="vid1"))
    main(["--in", str(in_tight), "--out", str(out_tight), "--target", "1", "--max", "1", "--min", "0", "--overlap", "0"])
    tight_chunks = json.loads((out_tight / "vid1.json").read_text(encoding="utf-8"))

    in_loose = tmp_path / "in_loose"
    out_loose = tmp_path / "out_loose"
    in_loose.mkdir()
    write_transcript(in_loose / "vid1.json", make_transcript(texts, video_id="vid1"))
    main(["--in", str(in_loose), "--out", str(out_loose), "--target", "1000", "--max", "1000", "--min", "0", "--overlap", "0"])
    loose_chunks = json.loads((out_loose / "vid1.json").read_text(encoding="utf-8"))

    assert len(tight_chunks) == 4
    assert len(loose_chunks) == 1


# =======================================================================
# chunks.jsonl format
# =======================================================================
def test_chunks_jsonl_has_one_json_object_per_line_with_unescaped_cyrillic(tmp_path):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    write_transcript(in_dir / "vid1.json", make_transcript([(STEP_TEXT, 0.0, 5.0)], video_id="vid1"))
    write_transcript(in_dir / "vid2.json", make_transcript([(STEP_TEXT, 0.0, 5.0)], video_id="vid2"))

    main(["--in", str(in_dir), "--out", str(out_dir)])

    raw = (out_dir / "chunks.jsonl").read_text(encoding="utf-8")
    lines = raw.splitlines()
    assert len(lines) == 2
    for line in lines:
        row = json.loads(line)  # each line parses on its own as one object
        assert row["text"] == STEP_TEXT
    # ensure_ascii=False: Cyrillic appears literally, not as \uXXXX escapes
    assert "Обжарьте" in raw
    assert "\\u" not in raw


def test_videos_are_processed_in_sorted_filename_order(tmp_path):
    """The input files are walked in sorted order, so the manifest and the
    combined jsonl are reproducible between runs.

    Without a stable order a re-run reshuffles chunks.jsonl and every diff of
    the artifacts becomes unreadable.
    """
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    in_dir.mkdir()
    for vid in ("vid3", "vid1", "vid2"):
        write_transcript(
            in_dir / f"{vid}.json",
            make_transcript([("Нарежьте лук и обжарьте его.", 0.0, 5.0)], video_id=vid),
        )

    main(["--in", str(in_dir), "--out", str(out_dir)])

    assert [m["video_id"] for m in manifest_of(out_dir)] == ["vid1", "vid2", "vid3"]
    assert [row["video_id"] for row in read_jsonl(out_dir / "chunks.jsonl")] == [
        "vid1", "vid2", "vid3",
    ]


@pytest.mark.parametrize(
    "overlap, expect_repeat",
    [
        pytest.param(0, False, id="no_overlap"),
        pytest.param(5, True, id="with_overlap"),
    ],
)
def test_overlap_flag_controls_whether_chunks_repeat_the_previous_segment(
    tmp_path, overlap, expect_repeat
):
    """--overlap is forwarded to ChunkConfig on its own.

    The other three size flags are covered together elsewhere; this one needs
    its own case because dropping it leaves the NUMBER of chunks unchanged and
    only alters their text — the two runs below differ in exactly that.
    """
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    in_dir.mkdir()
    first = "Обжарьте лук и посолите блюдо."
    segments = [
        (first, 0.0, 5.0),
        ("Потушите мясо и добавьте лук.", 5.0, 10.0),
        ("Посыпьте зеленью и полейте соусом.", 10.0, 15.0),
    ]
    write_transcript(in_dir / "vid1.json", make_transcript(segments, video_id="vid1"))

    main(["--in", str(in_dir), "--out", str(out_dir), "--target", "2",
          "--max", "1000", "--min", "0", "--overlap", str(overlap)])

    chunks = read_jsonl(out_dir / "chunks.jsonl")
    assert len(chunks) == 3
    assert chunks[1]["text"].startswith(first) is expect_repeat
