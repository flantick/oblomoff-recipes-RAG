"""Tests for src/generation/schemas.py: the structured LLM output schema,
the source reference schema, and the pipeline's final answer schema."""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from src.generation.schemas import LLMRecipe, RecipeAnswer, SourceRef


# --- _coerce_source_n: values that resolve to an int ----------------------
@pytest.mark.parametrize(
    "raw, expected",
    [
        pytest.param(3, 3, id="plain-int"),
        pytest.param("[1]", 1, id="bracketed-string"),
        pytest.param("фрагмент 2", 2, id="text-with-number"),
        pytest.param("3", 3, id="digit-string"),
        pytest.param(3.7, 3, id="float-truncates-towards-zero-not-rounds"),
        pytest.param([3, 4], 3, id="list-takes-first-element"),
        pytest.param((3, 4), 3, id="tuple-takes-first-element"),
        pytest.param({3}, 3, id="single-element-set"),
        pytest.param([[5]], 5, id="nested-list-recurses"),
        pytest.param("фрагменты 2 и 5", 2, id="several-numbers-takes-first-found"),
        # \d+ has no sign handling, so the leading "-" is simply not part of
        # the match: the regex finds "3", not "-3". This is the actual
        # behaviour of the shipped code, not a bug being documented here.
        pytest.param("-3", 3, id="negative-number-loses-its-sign"),
    ],
)
def test_llm_recipe_coerces_source_n_to_int(raw, expected):
    """source_n accepts the various shapes an LLM citation comes back as and
    normalizes every one of them to a plain int."""
    recipe = LLMRecipe(found=True, source_n=raw)
    assert recipe.source_n == expected


# --- _coerce_source_n: values that resolve to None -------------------------
@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(None, id="none"),
        pytest.param("не знаю", id="string-without-digits"),
        pytest.param([], id="empty-list"),
        pytest.param((), id="empty-tuple"),
        pytest.param("", id="empty-string"),
        # bool is a subclass of int in Python: without the isinstance(v, bool)
        # guard True/False would be coerced to 1/0 by the plain int branch.
        pytest.param(True, id="bool-true-is-not-1"),
        pytest.param(False, id="bool-false-is-not-0"),
        # The model sometimes answers with an object or a nested structure
        # instead of a number; anything the branches above do not recognise
        # falls through to the final `return None` rather than raising.
        pytest.param({"n": 1}, id="dict-is-not-a-citation"),
        pytest.param({"a": 1}.items(), id="unrecognised-iterable"),
    ],
)
def test_llm_recipe_coerces_source_n_to_none(raw):
    """Garbage or absent citations fall back to None instead of raising or
    being coerced to a misleading number."""
    recipe = LLMRecipe(found=True, source_n=raw)
    assert recipe.source_n is None


# --- LLMRecipe: defaults and required fields --------------------------------
def test_llm_recipe_defaults_when_only_found_is_given():
    """Every field but found has a default, and source_n/notes default to
    None rather than being required."""
    recipe = LLMRecipe(found=True)
    assert recipe.dish == ""
    assert recipe.ingredients == []
    assert recipe.steps == []
    assert recipe.source_n is None
    assert recipe.notes is None


def test_llm_recipe_found_is_required():
    """Constructing LLMRecipe without found raises, because unlike every
    other field it carries no default."""
    with pytest.raises(ValidationError) as exc_info:
        LLMRecipe()
    assert exc_info.value.errors()[0]["loc"] == ("found",)


def test_llm_recipe_mutable_defaults_are_independent_per_instance():
    """ingredients/steps use Field(default_factory=list): two instances must
    not share the same underlying list object."""
    first = LLMRecipe(found=True)
    second = LLMRecipe(found=True)

    first.ingredients.append("соль")
    first.steps.append("нарезать лук")

    assert first.ingredients is not second.ingredients
    assert first.steps is not second.steps
    assert second.ingredients == []
    assert second.steps == []


# --- SourceRef: every field is required -------------------------------------
_VALID_SOURCE_REF = dict(
    n=1,
    video_id="vid1",
    title="Как приготовить стейк",
    url="https://youtu.be/vid1?t=60",
    timecode="01:00",
)


@pytest.mark.parametrize("missing_field", list(_VALID_SOURCE_REF))
def test_source_ref_field_is_required(missing_field):
    """Dropping any single field from SourceRef raises a ValidationError
    naming that exact field."""
    kwargs = {k: v for k, v in _VALID_SOURCE_REF.items() if k != missing_field}
    with pytest.raises(ValidationError) as exc_info:
        SourceRef(**kwargs)
    assert exc_info.value.errors()[0]["loc"] == (missing_field,)


def test_source_ref_accepts_all_fields():
    """All fields round-trip unchanged when every one is supplied."""
    ref = SourceRef(**_VALID_SOURCE_REF)
    assert ref.n == 1
    assert ref.video_id == "vid1"
    assert ref.title == "Как приготовить стейк"
    assert ref.url == "https://youtu.be/vid1?t=60"
    assert ref.timecode == "01:00"


# --- RecipeAnswer: defaults and required fields -----------------------------
def test_recipe_answer_defaults_when_only_query_and_found_are_given():
    """query and found have no defaults; every other field falls back to its
    documented default value."""
    answer = RecipeAnswer(query="как приготовить борщ", found=True)
    assert answer.dish == ""
    assert answer.ingredients == []
    assert answer.steps == []
    assert answer.notes is None
    assert answer.source is None
    assert answer.sources == []
    assert answer.model == ""
    assert answer.used_reranker is False


def test_recipe_answer_query_is_required():
    """Constructing RecipeAnswer without query raises, because it carries no
    default (unlike dish/ingredients/steps/...)."""
    with pytest.raises(ValidationError) as exc_info:
        RecipeAnswer(found=True)
    assert exc_info.value.errors()[0]["loc"] == ("query",)


# --- RecipeAnswer: JSON round-trip keeps nested SourceRef data --------------
def test_recipe_answer_model_dump_json_keeps_nested_source_ref_fields():
    """model_dump_json must not drop or flatten the nested SourceRef data
    held in source/sources."""
    main_source = SourceRef(
        n=1, video_id="vid1", title="Стейк", url="https://youtu.be/vid1?t=60",
        timecode="01:00",
    )
    extra_source = SourceRef(
        n=2, video_id="vid2", title="Соус", url="https://youtu.be/vid2?t=30",
        timecode="00:30",
    )
    answer = RecipeAnswer(
        query="как приготовить стейк",
        found=True,
        dish="Стейк",
        source=main_source,
        sources=[main_source, extra_source],
    )

    dumped = json.loads(answer.model_dump_json())

    assert dumped["source"]["video_id"] == "vid1"
    assert dumped["source"]["timecode"] == "01:00"
    assert [s["video_id"] for s in dumped["sources"]] == ["vid1", "vid2"]
    assert dumped["sources"][1]["title"] == "Соус"
