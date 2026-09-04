"""Tests for src/retrieval/filters.py: matching a query against the
PLAYLIST_INTENTS regexes and assembling the Qdrant intent filter."""
from __future__ import annotations

import re

import pytest
from qdrant_client import models

from src.retrieval import filters as filters_mod
from src.retrieval.filters import build_intent_filter, detect_playlists

FISH = "Блюда из рыбы и морепродуктов"
GRILL = "Мангал, гриль, бербекю, печь"
POULTRY = "Блюдо из птицы"
MEAT = "Блюда из мяса"


# --- detect_playlists: one keyword per alternative of every intent ------
@pytest.mark.parametrize(
    "query, expected_title",
    [
        pytest.param("салат", "Салаты", id="салат"),
        pytest.param("суп", "Супы", id="суп"),
        pytest.param("борщ", "Супы", id="борщ"),
        pytest.param("щи", "Супы", id="щи"),
        pytest.param("солянка", "Супы", id="солянка"),
        pytest.param("рыба", FISH, id="рыба"),
        pytest.param("лосось", FISH, id="лосось"),
        pytest.param("форель", FISH, id="форель"),
        pytest.param("треска", FISH, id="треска"),
        pytest.param("креветки", FISH, id="креветки"),
        pytest.param("кальмары", FISH, id="кальмары"),
        pytest.param("морепродукты", FISH, id="морепродукты"),
        pytest.param("закуска", "Закуски", id="закуска"),
        pytest.param("намазка", "Закуски", id="намазка"),
        pytest.param("паштет", "Закуски", id="паштет"),
        pytest.param("брускетта", "Закуски", id="брускетта"),
        pytest.param("курица", POULTRY, id="курица"),
        pytest.param("индейка", POULTRY, id="индейка"),
        pytest.param("утка", POULTRY, id="утка"),
        pytest.param("птица", POULTRY, id="птица"),
        pytest.param("стейк", MEAT, id="стейк"),
        pytest.param("говядина", MEAT, id="говядина"),
        pytest.param("свинина", MEAT, id="свинина"),
        pytest.param("баранина", MEAT, id="баранина"),
        pytest.param("мясо", MEAT, id="мясо"),
        pytest.param("фарш", MEAT, id="фарш"),
        pytest.param("котлеты", MEAT, id="котлеты"),
        pytest.param("шашлык", GRILL, id="шашлык"),
        pytest.param("мангал", GRILL, id="мангал"),
        pytest.param("гриль", GRILL, id="гриль"),
        pytest.param("барбекю", GRILL, id="барбекю"),
        pytest.param("на углях", GRILL, id="na-uglyah"),
        pytest.param("на костре", GRILL, id="na-kostre"),
        pytest.param("бутерброд", "Бутеры", id="бутерброд"),
        pytest.param("сэндвич", "Бутеры", id="сэндвич"),
        pytest.param("сендвич", "Бутеры", id="сендвич"),
        pytest.param("тост", "Бутеры", id="тост"),
        pytest.param("новогодний", "Новогодние рецепты", id="новогодний"),
        pytest.param("на новый год", "Новогодние рецепты", id="na-novy-god"),
        pytest.param("рождественский", "Новогодние рецепты", id="рождественский"),
    ],
)
def test_detect_playlists_matches_keyword_for_intent(query, expected_title):
    """Every keyword of PLAYLIST_INTENTS lands in exactly its own playlist."""
    assert detect_playlists(query) == [expected_title]


# --- case insensitivity --------------------------------------------------
@pytest.mark.parametrize(
    "query",
    ["САЛАТ", "Салат", "СаЛаТ"],
    ids=["upper", "title-case", "mixed-case"],
)
def test_detect_playlists_case_insensitive(query):
    """Matching does not depend on the case of the query."""
    assert detect_playlists(query) == ["Салаты"]


# --- several intents at once: order and dedup ----------------------------
def test_detect_playlists_multiple_intents_follow_declaration_order():
    """A query hitting two intents returns both in PLAYLIST_INTENTS order
    (poultry is declared before the grill)."""
    assert detect_playlists("куриный шашлык на мангале") == [POULTRY, GRILL]


def test_detect_playlists_dedupes_when_two_entries_share_a_title(monkeypatch):
    """Two different patterns yielding the same title produce it once, not twice.

    In the real PLAYLIST_INTENTS all 9 titles are unique (the keywords of one
    intent are joined with | inside a SINGLE pattern), so the 'title not in
    hits' branch is unreachable with the shipped config — it is exercised here
    through a substituted _COMPILED.
    """
    monkeypatch.setattr(
        filters_mod,
        "_COMPILED",
        [
            (re.compile(r"утка"), "Дубль"),
            (re.compile(r"гусь"), "Дубль"),
        ],
    )
    assert detect_playlists("утка и гусь") == ["Дубль"]


# --- no match ------------------------------------------------------------
def test_detect_playlists_returns_empty_list_when_nothing_matches():
    """A query with no culinary keyword lands in no playlist at all."""
    assert detect_playlists("как испечь торт") == []


# --- regex boundaries ----------------------------------------------------
def test_detect_playlists_matches_sup_at_word_start():
    """The soup pattern catches 'суп' and its inflections when it starts a word."""
    assert detect_playlists("грибной суп") == ["Супы"]


def test_detect_playlists_word_boundary_rejects_sup_mid_word():
    """The word boundary before 'суп' keeps 'насупился' out (no other Супы
    alternative is hit there either)."""
    assert detect_playlists("он насупился") == []


def test_detect_playlists_matches_salad_inflections_without_boundaries():
    """The salad pattern deliberately matches inside a word, which is how the
    inflections are caught without listing every ending: 'салатник' (a salad
    bowl) counts as a salad query, while 'сало' and 'салями' do not match."""
    assert detect_playlists("салатник") == ["Салаты"]
    assert detect_playlists("сало и салями") == []


# --- bugs found in PLAYLIST_INTENTS: false POSITIVES ---------------------
# The filter is called "soft", but a Qdrant `should` filter is a hard
# restriction: a point passes only if it matches. A false positive therefore
# silently discards every correct result instead of merely reordering it.
@pytest.mark.parametrize("query", ["овощи", "хрящи", "лещи"])
def test_detect_playlists_does_not_tag_words_ending_in_schi_as_soup(query):
    """Words merely ending in 'щи' must not be routed to the soup playlist."""
    assert "Супы" not in detect_playlists(query)


@pytest.mark.parametrize("query", ["куркума", "курага", "плов с куркумой"])
def test_detect_playlists_does_not_tag_kur_words_as_poultry(query):
    """Culinary words starting with 'кур' that are not poultry must not be
    routed to the poultry playlist."""
    assert POULTRY not in detect_playlists(query)


@pytest.mark.parametrize("query", ["минутку", "шутка"])
def test_detect_playlists_does_not_tag_words_containing_utk_as_poultry(query):
    """Words merely containing 'утк' must not be routed to poultry."""
    assert POULTRY not in detect_playlists(query)


# --- bugs found in PLAYLIST_INTENTS: false NEGATIVES ---------------------
# Worse than the false positives above: with the intent filter on, the query is
# not merely left unfiltered - another pattern routes it to the WRONG playlist.
@pytest.mark.parametrize("query", ["запеки на гриле", "запеки грилем"])
def test_detect_playlists_matches_grill_inflections(query):
    """Inflected forms of 'гриль' must reach the grill playlist."""
    assert detect_playlists(query) == [GRILL]


def test_detect_playlists_matches_mussel_singular():
    """The singular 'мидия' must reach the seafood playlist."""
    assert detect_playlists("мидия") == [FISH]


# --- build_intent_filter --------------------------------------------
def test_build_intent_filter_returns_none_and_empty_list_when_no_match():
    """With no match build_intent_filter builds no filter at all."""
    flt, titles = build_intent_filter("как испечь торт")
    assert flt is None
    assert titles == []


def test_build_intent_filter_builds_should_filter_for_single_match():
    """One matched playlist yields a Filter with exactly one should condition
    on playlist_titles carrying that value."""
    flt, titles = build_intent_filter("салат")
    assert titles == ["Салаты"]
    assert flt == models.Filter(
        should=[
            models.FieldCondition(
                key="playlist_titles", match=models.MatchValue(value="Салаты")
            )
        ]
    )


def test_build_intent_filter_builds_one_should_condition_per_playlist():
    """Two matched playlists yield two should conditions in detect_playlists
    order, each carrying its own playlist_titles value."""
    flt, titles = build_intent_filter("куриный шашлык на мангале")
    assert titles == [POULTRY, GRILL]
    assert flt.should == [
        models.FieldCondition(
            key="playlist_titles", match=models.MatchValue(value=POULTRY)
        ),
        models.FieldCondition(
            key="playlist_titles", match=models.MatchValue(value=GRILL)
        ),
    ]
    assert flt.must is None
    assert flt.must_not is None
