"""Tests for src.generation.prompts.build_messages.

SYSTEM_PROMPT's wording is not asserted here on purpose: a text match on the
prompt would break on the next wording tweak without catching any real
regression (by project convention). Only build_messages's contract is under test.
"""
from __future__ import annotations

from src.generation.prompts import SYSTEM_PROMPT, build_messages


def test_build_messages_returns_system_then_user_roles():
    """The two messages come back in a fixed order: system, then user."""
    messages = build_messages("query", "context")

    assert [m["role"] for m in messages] == ["system", "user"]


def test_build_messages_system_content_matches_constant():
    """The system message is exactly the imported SYSTEM_PROMPT, not a copy."""
    messages = build_messages("query", "context")

    assert messages[0]["content"] is SYSTEM_PROMPT


def test_build_messages_user_content_includes_query_and_context():
    """Both the query and the retrieved context land in the user message."""
    messages = build_messages(
        "Как приготовить борщ?", "[1] Нарежьте свёклу соломкой."
    )

    user_content = messages[1]["content"]
    assert "Как приготовить борщ?" in user_content
    assert "[1] Нарежьте свёклу соломкой." in user_content


def test_build_messages_places_query_before_context():
    """The query appears earlier in the template than the context — this
    ordering affects how the model weighs the instruction vs. the evidence."""
    messages = build_messages("МАРКЕР_ЗАПРОСА", "МАРКЕР_КОНТЕКСТА")

    user_content = messages[1]["content"]
    assert user_content.index("МАРКЕР_ЗАПРОСА") < user_content.index("МАРКЕР_КОНТЕКСТА")


def test_build_messages_handles_curly_braces_in_query_and_context():
    """Transcripts and JSON examples both contain literal braces; .format
    must not treat them as format fields and raise KeyError/IndexError."""
    query = "Что значит {непонятное} слово?"
    context = '[1] Ведущий сказал: {"found": true}'

    messages = build_messages(query, context)

    user_content = messages[1]["content"]
    assert query in user_content
    assert context in user_content


def test_build_messages_empty_query_and_context_do_not_raise():
    """An empty query and empty context are valid inputs, not error cases."""
    messages = build_messages("", "")

    assert messages[1]["content"] == (
        "Вопрос: \n\nФрагменты расшифровок:\n\n\nВерни JSON по схеме из инструкции."
    )


def test_build_messages_independent_across_calls():
    """Two consecutive calls don't share mutable state: the second call's
    result carries none of the first call's data."""
    first = build_messages("первый запрос", "первый контекст")
    second = build_messages("второй запрос", "второй контекст")

    assert "первый запрос" not in second[1]["content"]
    assert "первый контекст" not in second[1]["content"]
    assert first[1]["content"] != second[1]["content"]
