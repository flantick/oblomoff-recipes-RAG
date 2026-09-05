"""Tests for src/index/build.py: load_chunks() and non_negative_int().

main() is a CLI wrapper (argparse, tqdm, a real embedder and a real Qdrant
client) and is intentionally not covered by project convention. The one
exception below stops at argument parsing and never reaches that machinery.
"""
from __future__ import annotations

import argparse
import json

import pytest

from src.index.build import load_chunks, main as build_main, non_negative_int


def _write_jsonl(tmp_path, rows: list[dict]):
    path = tmp_path / "chunks.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return path


ROWS = [{"chunk_id": "a"}, {"chunk_id": "b"}, {"chunk_id": "c"}]


@pytest.mark.parametrize(
    "limit, expected",
    [
        pytest.param(None, ROWS, id="limit_none_returns_all_rows_in_order"),
        pytest.param(2, ROWS[:2], id="limit_less_than_row_count"),
        pytest.param(3, ROWS, id="limit_equal_to_row_count"),
        pytest.param(10, ROWS, id="limit_greater_than_row_count_no_error"),
    ],
)
def test_load_chunks_limit_variants(tmp_path, limit, expected):
    """load_chunks slices the parsed rows to `limit`, or returns them all."""
    path = _write_jsonl(tmp_path, ROWS)
    assert load_chunks(path, limit) == expected


def test_load_chunks_limit_zero_loads_nothing(tmp_path):
    """limit=0 means zero rows, not "all of them": only limit=None means
    "no limit", so a zero is taken at face value."""
    path = _write_jsonl(tmp_path, ROWS)
    assert load_chunks(path, 0) == []


def test_load_chunks_skips_blank_and_whitespace_only_lines(tmp_path):
    """Blank lines and lines containing only whitespace are filtered out
    before json.loads is called on them."""
    path = tmp_path / "chunks.jsonl"
    path.write_text(
        '{"chunk_id": "a"}\n\n   \n{"chunk_id": "b"}\n\t\n',
        encoding="utf-8",
    )
    assert load_chunks(path, None) == [{"chunk_id": "a"}, {"chunk_id": "b"}]


def test_load_chunks_empty_file_returns_empty_list(tmp_path):
    """An empty file has no non-blank lines, so the result is []."""
    path = tmp_path / "chunks.jsonl"
    path.write_text("", encoding="utf-8")
    assert load_chunks(path, None) == []


def test_load_chunks_reads_cyrillic_text_as_utf8(tmp_path):
    """The file is decoded with an explicit encoding="utf-8", so Cyrillic
    text round-trips correctly even though Windows' default locale encoding
    is not UTF-8."""
    row = {"chunk_id": "a", "text": "Обжарьте стейк на сильном огне до румяной корочки."}
    path = _write_jsonl(tmp_path, [row])
    assert load_chunks(path, None) == [row]


def test_load_chunks_invalid_json_raises_json_decode_error(tmp_path):
    """A malformed JSON line is not swallowed: json.JSONDecodeError propagates
    to the caller instead of being skipped or replaced with a default."""
    path = tmp_path / "chunks.jsonl"
    path.write_text('{"chunk_id": "a"}\n{not valid json}\n', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_chunks(path, None)


def test_non_negative_int_rejects_a_negative_limit():
    """The CLI refuses --limit -1 instead of silently slicing off the last
    rows: as a plain slice bound a negative value quietly loads everything
    except the tail, which is never what the flag was meant to do."""
    with pytest.raises(argparse.ArgumentTypeError):
        non_negative_int("-1")


@pytest.mark.parametrize("value, expected", [("0", 0), ("200", 200)])
def test_non_negative_int_accepts_zero_and_positive(value, expected):
    """Zero and positive values pass through as plain ints."""
    assert non_negative_int(value) == expected


def test_build_cli_exits_on_a_negative_limit(capsys):
    """argparse turns the rejection into the usual exit code 2 plus a usage
    message, rather than starting an embedding run over a truncated corpus."""
    with pytest.raises(SystemExit) as exc:
        build_main(["--limit", "-1"])
    assert exc.value.code == 2
    assert "--limit" in capsys.readouterr().err
