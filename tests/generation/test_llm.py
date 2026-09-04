"""Tests for src/generation/llm.py.

openai.OpenAI is imported lazily inside LLMClient.__init__ (`from openai
import OpenAI`), so a real network client is never created here: a fake
module is installed into sys.modules BEFORE the client is constructed, the
same pattern tests/retrieval/test_rerank.py uses for FlagEmbedding.
"""
from __future__ import annotations

import sys
import types

import pytest

from src.config import LLM_MAX_TOKENS, LLM_TEMPERATURE
from src.generation.llm import LLMClient, _loads_lenient


# --- fake openai module -------------------------------------------------

class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    def __init__(self, client: "FakeOpenAI") -> None:
        self._client = client

    def create(self, **kwargs):
        self._client.create_calls.append(kwargs)
        item = self._client.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return FakeResponse(item)


class FakeChat:
    def __init__(self, client: "FakeOpenAI") -> None:
        self.completions = FakeCompletions(client)


class FakeModels:
    def __init__(self, client: "FakeOpenAI") -> None:
        self._client = client

    def list(self):
        if self._client.models_error is not None:
            raise self._client.models_error
        return {"data": []}


class FakeOpenAI:
    """Fake of openai.OpenAI.

    Records the connection kwargs it was constructed with. A test queues up
    responses to chat.completions.create() in `responses`: each item is
    either the text content of the next reply, or an exception instance to
    raise on that call. `models_error`, when set, is raised by models.list().
    """

    def __init__(self, *, base_url=None, api_key=None, timeout=None) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.create_calls: list[dict] = []
        self.responses: list = []
        self.models_error: Exception | None = None
        self.chat = FakeChat(self)
        self.models = FakeModels(self)


@pytest.fixture
def fake_openai_module(monkeypatch):
    """Replaces sys.modules["openai"] with a fake module BEFORE
    LLMClient.__init__ runs its lazy `from openai import OpenAI`."""
    fake_module = types.ModuleType("openai")
    fake_module.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    return fake_module


def make_client(fake_openai_module, **kwargs) -> LLMClient:
    defaults: dict = dict(
        base_url="http://fake-host/v1", api_key="k", model="test-model", timeout=5.0,
    )
    defaults.update(kwargs)
    return LLMClient(**defaults)


# --- LLMClient.__init__ ---------------------------------------------------

def test_init_passes_connection_args_to_openai_client(fake_openai_module):
    """base_url/api_key/timeout given to LLMClient reach the OpenAI client."""
    client = make_client(
        fake_openai_module, base_url="http://host/v1", api_key="secret", timeout=42.0,
    )
    assert client.client.base_url == "http://host/v1"
    assert client.client.api_key == "secret"
    assert client.client.timeout == 42.0


def test_init_keeps_the_model_name_on_the_client(fake_openai_module):
    """The model name stays on LLMClient and is sent per request, rather than
    being baked into the OpenAI connection."""
    client = make_client(fake_openai_module, model="qwen2.5-14b")
    assert client.model == "qwen2.5-14b"


# --- LLMClient.chat ---------------------------------------------------------

def test_chat_sends_model_messages_temperature_max_tokens(fake_openai_module):
    """The request carries the client's model plus the messages and sampling
    settings given to chat(); the reply's content is what comes back."""
    client = make_client(fake_openai_module, model="qwen-test")
    client.client.responses = ["ответ"]
    messages = [{"role": "user", "content": "привет"}]

    result = client.chat(messages, temperature=0.7, max_tokens=99)

    call = client.client.create_calls[0]
    assert call["model"] == "qwen-test"
    assert call["messages"] == messages
    assert call["temperature"] == 0.7
    assert call["max_tokens"] == 99
    assert result == "ответ"


def test_chat_uses_config_defaults_when_sampling_args_are_omitted(fake_openai_module):
    """Called without temperature/max_tokens, chat() sends the configured
    LLM_TEMPERATURE and LLM_MAX_TOKENS, not numbers hardcoded in the method."""
    client = make_client(fake_openai_module)
    client.client.responses = ["ok"]

    client.chat([{"role": "user", "content": "x"}])

    call = client.client.create_calls[0]
    assert call["temperature"] == pytest.approx(LLM_TEMPERATURE)
    assert call["max_tokens"] == LLM_MAX_TOKENS


def test_chat_json_mode_true_adds_response_format(fake_openai_module):
    """json_mode=True asks the backend for a JSON object."""
    client = make_client(fake_openai_module)
    client.client.responses = ["ok"]

    client.chat([{"role": "user", "content": "x"}], json_mode=True)

    assert client.client.create_calls[0]["response_format"] == {"type": "json_object"}


def test_chat_json_mode_false_omits_response_format(fake_openai_module):
    """json_mode=False leaves the key out entirely rather than sending null."""
    client = make_client(fake_openai_module)
    client.client.responses = ["ok"]

    client.chat([{"role": "user", "content": "x"}], json_mode=False)

    assert "response_format" not in client.client.create_calls[0]


def test_chat_falls_back_and_retries_without_response_format(fake_openai_module):
    """The backend rejects response_format -> the key is dropped and the
    request is retried once; the retried reply is what comes back."""
    client = make_client(fake_openai_module)
    client.client.responses = [
        RuntimeError("Unrecognized request argument supplied: response_format"),
        "второй ответ",
    ]

    result = client.chat([{"role": "user", "content": "x"}], json_mode=True)

    assert result == "второй ответ"
    assert len(client.client.create_calls) == 2
    assert "response_format" in client.client.create_calls[0]
    assert "response_format" not in client.client.create_calls[1]


def test_chat_reraises_exception_without_response_format_in_text(fake_openai_module):
    """An unrelated failure is not treated as a response_format problem: it
    propagates immediately, with no retry."""
    client = make_client(fake_openai_module)
    client.client.responses = [ConnectionError("сервер недоступен")]

    with pytest.raises(ConnectionError):
        client.chat([{"role": "user", "content": "x"}], json_mode=True)

    assert len(client.client.create_calls) == 1


def test_chat_reraises_response_format_error_when_json_mode_false(fake_openai_module):
    """The exception text happens to mention response_format, but
    json_mode=False means the client never sent that key -> no retry, the
    exception just propagates."""
    client = make_client(fake_openai_module)
    client.client.responses = [RuntimeError("response_format is not supported here")]

    with pytest.raises(RuntimeError):
        client.chat([{"role": "user", "content": "x"}], json_mode=False)

    assert len(client.client.create_calls) == 1


def test_chat_returns_empty_string_when_content_is_none(fake_openai_module):
    """A reply whose content is null becomes "" rather than propagating None."""
    client = make_client(fake_openai_module)
    client.client.responses = [None]

    result = client.chat([{"role": "user", "content": "x"}])

    assert result == ""


# --- LLMClient.chat_json -----------------------------------------------------

def test_chat_json_returns_dict_on_first_valid_response(fake_openai_module):
    """A valid JSON reply is parsed and returned without a second request, and
    the request itself went out in JSON mode."""
    client = make_client(fake_openai_module)
    client.client.responses = ['{"found": true, "answer": "да"}']

    result = client.chat_json([{"role": "user", "content": "x"}])

    assert result == {"found": True, "answer": "да"}
    assert len(client.client.create_calls) == 1
    assert client.client.create_calls[0]["response_format"] == {"type": "json_object"}


def test_chat_json_forwards_extra_kwargs_to_the_request(fake_openai_module):
    """chat_json passes **kw (temperature and the like) down through chat() to
    the API call — this is how answer()'s temperature argument reaches vLLM."""
    client = make_client(fake_openai_module)
    client.client.responses = ['{"ok": true}']

    client.chat_json([{"role": "user", "content": "x"}], temperature=0.9)

    assert client.client.create_calls[0]["temperature"] == pytest.approx(0.9)


def test_chat_json_retries_once_on_invalid_json_then_succeeds(fake_openai_module):
    """An unparseable reply triggers exactly one stricter retry, which carries
    the original messages plus the bad answer and a correction."""
    client = make_client(fake_openai_module)
    client.client.responses = ["это не json", '{"found": false}']
    messages = [{"role": "user", "content": "исходный вопрос"}]

    result = client.chat_json(messages)

    assert result == {"found": False}
    assert len(client.client.create_calls) == 2
    assert client.client.create_calls[1]["response_format"] == {"type": "json_object"}
    retry_messages = client.client.create_calls[1]["messages"]
    assert retry_messages[: len(messages)] == messages
    assert retry_messages[-2] == {"role": "assistant", "content": "это не json"}
    assert retry_messages[-1] == {
        "role": "user",
        "content": "Это был невалидный JSON. Верни ТОЛЬКО корректный JSON-объект по схеме.",
    }


def test_chat_json_raises_value_error_when_both_attempts_invalid(fake_openai_module):
    """Two unparseable replies in a row give up with a ValueError instead of
    looping or returning None."""
    client = make_client(fake_openai_module)
    client.client.responses = ["мусор раз", "мусор два"]

    with pytest.raises(ValueError):
        client.chat_json([{"role": "user", "content": "x"}])

    assert len(client.client.create_calls) == 2


def test_chat_json_truncates_raw_response_in_retry_message(fake_openai_module):
    """A raw reply longer than 2000 chars is cut to raw[:2000] before being
    echoed back as the assistant turn of the retry."""
    client = make_client(fake_openai_module)
    long_raw = "x" * 2500  # no braces at all -> _loads_lenient returns None
    client.client.responses = [long_raw, '{"ok": true}']

    client.chat_json([{"role": "user", "content": "x"}])

    assistant_turn = client.client.create_calls[1]["messages"][-2]
    assert assistant_turn["content"] == long_raw[:2000]
    assert len(assistant_turn["content"]) == 2000


# --- LLMClient.health ---------------------------------------------------------

def test_health_true_when_models_list_succeeds(fake_openai_module):
    """A reachable backend reports healthy."""
    client = make_client(fake_openai_module)
    assert client.health() is True


def test_health_false_when_models_list_raises(fake_openai_module):
    """An unreachable backend reports unhealthy instead of raising."""
    client = make_client(fake_openai_module)
    client.client.models_error = ConnectionError("недоступен")

    assert client.health() is False


# --- _loads_lenient ------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        pytest.param('{"a": 1}', {"a": 1}, id="plain_json_object"),
        pytest.param('```json\n{"a": 1}\n```', {"a": 1}, id="json_fenced"),
        pytest.param('```\n{"a": 1}\n```', {"a": 1}, id="bare_fenced"),
        pytest.param('Вот ответ: {"a": 1} спасибо', {"a": 1}, id="object_inside_garbage"),
        pytest.param("просто текст без скобок", None, id="no_braces"),
        pytest.param('текст {"a": 1,} конец', None, id="malformed_json_inside_braces"),
        pytest.param("", None, id="empty_string"),
        pytest.param(None, None, id="none_input"),
        pytest.param("} раньше { позже", None, id="closing_brace_before_opening"),
        # The annotation says dict | None, but json.loads returns whatever the
        # payload holds and nothing narrows it. answer() survives this because
        # LLMRecipe.model_validate([1, 2]) raises and is downgraded to
        # found=false, so this pins the real contract rather than the hint.
        pytest.param("[1, 2]", [1, 2], id="json_array_is_returned_despite_dict_hint"),
        pytest.param("123", 123, id="bare_number_is_returned_despite_dict_hint"),
    ],
)
def test_loads_lenient(raw, expected):
    """Parsing tolerates fences and surrounding chatter, and gives up with None
    rather than raising when there is no usable JSON."""
    assert _loads_lenient(raw) == expected
