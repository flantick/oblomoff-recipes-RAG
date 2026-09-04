"""Tests for src/eval/dataset.py: GoldenItem, load_golden(), dump_report().

DEFAULT_GOLDEN points at the real data/eval/golden.jsonl — tests never read it
and never write outside tmp_path.
"""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from src.eval.dataset import GoldenItem, dump_report, load_golden


# --- GoldenItem.is_negative -------------------------------------------
@pytest.mark.parametrize(
    "relevant, expected",
    [
        pytest.param([], True, id="empty_relevant_is_negative"),
        pytest.param(["vid1"], False, id="non_empty_relevant_is_not_negative"),
    ],
)
def test_is_negative_depends_only_on_relevant(relevant, expected):
    """is_negative is True exactly when relevant is empty."""
    item = GoldenItem(id="q1", query="что-то", relevant=relevant)
    assert item.is_negative is expected


def test_is_negative_ignores_expect_found_when_they_disagree():
    """relevant=[] together with expect_found=True is a real, if odd,
    combination in the golden set: is_negative looks only at relevant, so it
    reports True even though expect_found says the answer should be found."""
    item = GoldenItem(id="q1", query="что-то", relevant=[], expect_found=True)
    assert item.is_negative is True
    assert item.expect_found is True


def test_golden_item_defaults():
    """Fields other than id/query fall back to the values eval relies on."""
    item = GoldenItem(id="q1", query="как приготовить стейк")
    assert item.kind == "exact"
    assert item.expect_found is True
    assert item.note == ""
    assert item.relevant == []


# --- load_golden --------------------------------------------------------
def _write_jsonl(tmp_path, rows: list[dict]):
    path = tmp_path / "golden.jsonl"
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
    )
    return path


ROWS = [
    {"id": "q1", "query": "стейк", "kind": "exact", "relevant": ["v1"]},
    {"id": "q2", "query": "мясо на огне", "kind": "paraphrase", "relevant": ["v1"]},
    {"id": "q3", "query": "штука без рецепта", "kind": "negative", "relevant": []},
]


def test_load_golden_returns_items_in_file_order(tmp_path):
    """A plain jsonl file becomes a list of GoldenItem in the same order."""
    path = _write_jsonl(tmp_path, ROWS)
    result = load_golden(path)
    assert [r.id for r in result] == ["q1", "q2", "q3"]
    assert all(isinstance(r, GoldenItem) for r in result)


def test_load_golden_skips_blank_and_whitespace_only_lines(tmp_path):
    """Blank lines and whitespace-only lines are filtered out before parsing."""
    path = tmp_path / "golden.jsonl"
    path.write_text(
        '{"id": "q1", "query": "a"}\n\n   \n{"id": "q2", "query": "b"}\n\t\n',
        encoding="utf-8",
    )
    result = load_golden(path)
    assert [r.id for r in result] == ["q1", "q2"]


def test_load_golden_empty_file_returns_empty_list(tmp_path):
    """An empty file has no non-blank lines, so the result is []."""
    path = tmp_path / "golden.jsonl"
    path.write_text("", encoding="utf-8")
    assert load_golden(path) == []


def test_load_golden_kinds_none_means_no_filtering(tmp_path):
    """kinds=None (the default) keeps every row regardless of kind."""
    path = _write_jsonl(tmp_path, ROWS)
    result = load_golden(path, kinds=None)
    assert [r.id for r in result] == ["q1", "q2", "q3"]


def test_load_golden_kinds_filters_and_keeps_order(tmp_path):
    """When kinds is given, only rows whose kind is in the list survive, in
    their original relative order."""
    path = _write_jsonl(tmp_path, ROWS)
    result = load_golden(path, kinds=["negative", "exact"])
    assert [r.id for r in result] == ["q1", "q3"]


def test_load_golden_kinds_with_absent_value_returns_empty_list(tmp_path):
    """A kinds filter that matches no row in the file yields []."""
    path = _write_jsonl(tmp_path, ROWS)
    result = load_golden(path, kinds=["descriptive"])
    assert result == []


def test_load_golden_reads_cyrillic_text_as_utf8(tmp_path):
    """The file is decoded with an explicit encoding="utf-8", so Russian
    queries round-trip correctly even though Windows' default locale
    encoding is not UTF-8."""
    row = {"id": "q1", "query": "как приготовить утку по-пекински"}
    path = _write_jsonl(tmp_path, [row])
    result = load_golden(path)
    assert result[0].query == "как приготовить утку по-пекински"


def test_load_golden_invalid_json_raises_validation_error(tmp_path):
    """A malformed JSON line is not swallowed: pydantic's ValidationError
    propagates to the caller instead of being skipped or defaulted."""
    path = tmp_path / "golden.jsonl"
    path.write_text('{"id": "q1", "query": "a"}\n{not valid json}\n', encoding="utf-8")
    with pytest.raises(ValidationError):
        load_golden(path)


def test_load_golden_accepts_str_path(tmp_path):
    """path may be passed as a plain string, not only as a Path."""
    path = _write_jsonl(tmp_path, ROWS[:1])
    result = load_golden(str(path))
    assert [r.id for r in result] == ["q1"]


# --- dump_report ----------------------------------------------------------
def test_dump_report_writes_readable_json(tmp_path):
    """The written file, read back and parsed, equals the given payload."""
    path = tmp_path / "report.json"
    payload = {"precision": 0.75, "n": 4}
    dump_report(path, payload)
    assert json.loads(path.read_text(encoding="utf-8")) == payload


def test_dump_report_creates_missing_parent_directories(tmp_path):
    """dump_report mkdir(parents=True)s the target's parent directory, so
    a nested path that does not exist yet still gets written."""
    path = tmp_path / "reports" / "nested" / "report.json"
    dump_report(path, {"ok": True})
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8")) == {"ok": True}


def test_dump_report_overwrites_existing_file(tmp_path):
    """Calling dump_report a second time replaces the previous content."""
    path = tmp_path / "report.json"
    dump_report(path, {"n": 1})
    dump_report(path, {"n": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"n": 2}


def test_dump_report_keeps_cyrillic_as_is_not_escaped(tmp_path):
    """ensure_ascii=False means Cyrillic is written as literal characters in
    the raw file text, not as \\uXXXX escapes."""
    path = tmp_path / "report.json"
    dump_report(path, {"note": "стейк готов"})
    raw = path.read_text(encoding="utf-8")
    assert "стейк готов" in raw
    assert "\\u" not in raw


def test_dump_report_uses_two_space_indent(tmp_path):
    """The file is pretty-printed with indent=2, checked against the raw text."""
    path = tmp_path / "report.json"
    dump_report(path, {"a": 1})
    raw = path.read_text(encoding="utf-8")
    assert '{\n  "a": 1\n}' == raw


def test_load_golden_empty_kinds_list_does_not_filter(tmp_path):
    """kinds=[] is falsy, so it means "no filter", not "keep nothing".

    It is a reachable value: run.py builds the list by splitting the --kinds
    flag, and an empty flag yields an empty list.
    """
    path = tmp_path / "golden.jsonl"
    path.write_text(
        '{"id": "q1", "query": "борщ", "relevant": ["v1"], "kind": "exact"}\n'
        '{"id": "q2", "query": "плов", "relevant": ["v2"], "kind": "paraphrase"}\n',
        encoding="utf-8",
    )

    assert [item.id for item in load_golden(path, kinds=[])] == ["q1", "q2"]


def test_load_golden_missing_file_raises(tmp_path):
    """A missing golden set fails loudly rather than evaluating on nothing."""
    with pytest.raises(FileNotFoundError):
        load_golden(tmp_path / "no-such-file.jsonl")


def test_dump_report_rejects_a_payload_it_cannot_serialise(tmp_path):
    """Unlike the ETL dumpers, dump_report passes no default= to json.dumps, so
    a non-serialisable value raises instead of being coerced to a string.

    Pinned because the two behaviours look interchangeable from the outside and
    a report that silently stringified objects would be worse than one that
    refuses to be written.
    """
    out = tmp_path / "report.json"

    with pytest.raises(TypeError):
        dump_report(out, {"when": object()})
