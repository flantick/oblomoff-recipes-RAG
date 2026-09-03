"""Tests for src.etl.ytdlp_common.ytdlp_network_opts.

The function re-reads src.config on every call (by design, so CLI flags can
override the module-level defaults at runtime — see the module docstring).
Every test therefore pins all four inputs explicitly via monkeypatch instead
of relying on whatever is in the environment: PROXY in particular is derived
from HTTPS_PROXY/HTTP_PROXY, which may well be set on a developer's machine.
"""
from __future__ import annotations

import pytest

from src import config
from src.etl.ytdlp_common import ytdlp_network_opts


@pytest.fixture(autouse=True)
def _blank_network_config(monkeypatch):
    """Start every test from a clean slate: no lang, proxy or cookies."""
    monkeypatch.setattr(config, "YTDLP_LANG", None, raising=False)
    monkeypatch.setattr(config, "PROXY", None, raising=False)
    monkeypatch.setattr(config, "YTDLP_COOKIES_FILE", None, raising=False)
    monkeypatch.setattr(config, "YTDLP_COOKIES_FROM_BROWSER", None, raising=False)


def test_all_settings_empty_returns_empty_dict():
    """With every knob unset, the function returns {} rather than None."""
    assert ytdlp_network_opts() == {}


def test_lang_is_wrapped_in_a_single_element_list(monkeypatch):
    """extractor_args.lang must be a list, not a bare string (yt-dlp's schema)."""
    monkeypatch.setattr(config, "YTDLP_LANG", "ru")

    opts = ytdlp_network_opts()

    assert opts == {"extractor_args": {"youtube": {"lang": ["ru"]}}}


@pytest.mark.parametrize("lang", [None, ""], ids=["none", "empty-string"])
def test_falsy_lang_omits_extractor_args_key(monkeypatch, lang):
    monkeypatch.setattr(config, "YTDLP_LANG", lang)

    opts = ytdlp_network_opts()

    assert "extractor_args" not in opts


def test_proxy_set_adds_proxy_key(monkeypatch):
    monkeypatch.setattr(config, "PROXY", "http://proxy.example:3128")

    opts = ytdlp_network_opts()

    assert opts == {"proxy": "http://proxy.example:3128"}


def test_cookies_file_adds_cookiefile_key_only(monkeypatch):
    monkeypatch.setattr(config, "YTDLP_COOKIES_FILE", "/tmp/cookies.txt")

    opts = ytdlp_network_opts()

    assert opts == {"cookiefile": "/tmp/cookies.txt"}
    assert "cookiesfrombrowser" not in opts


def test_cookies_from_browser_adds_tuple(monkeypatch):
    monkeypatch.setattr(config, "YTDLP_COOKIES_FROM_BROWSER", "chrome")

    opts = ytdlp_network_opts()

    assert opts == {"cookiesfrombrowser": ("chrome", None, None, None)}


def test_cookies_file_wins_over_browser_when_both_set(monkeypatch):
    """The elif branch means a configured cookie FILE silently shadows the
    browser-cookies setting instead of combining with it."""
    monkeypatch.setattr(config, "YTDLP_COOKIES_FILE", "/tmp/cookies.txt")
    monkeypatch.setattr(config, "YTDLP_COOKIES_FROM_BROWSER", "chrome")

    opts = ytdlp_network_opts()

    assert opts == {"cookiefile": "/tmp/cookies.txt"}


def test_no_cookies_configured_omits_both_keys():
    opts = ytdlp_network_opts()

    assert "cookiefile" not in opts
    assert "cookiesfrombrowser" not in opts


def test_all_settings_together_produce_exactly_three_keys(monkeypatch):
    monkeypatch.setattr(config, "YTDLP_LANG", "ru")
    monkeypatch.setattr(config, "PROXY", "http://proxy.example:3128")
    monkeypatch.setattr(config, "YTDLP_COOKIES_FILE", "/tmp/cookies.txt")
    monkeypatch.setattr(config, "YTDLP_COOKIES_FROM_BROWSER", "chrome")

    opts = ytdlp_network_opts()

    assert opts == {
        "extractor_args": {"youtube": {"lang": ["ru"]}},
        "proxy": "http://proxy.example:3128",
        "cookiefile": "/tmp/cookies.txt",
    }


def test_missing_getattr_based_attributes_do_not_raise(monkeypatch):
    """YTDLP_LANG/YTDLP_COOKIES_FILE/YTDLP_COOKIES_FROM_BROWSER are read via
    getattr(..., None), so an environment without them at all must not crash
    the call (PROXY is read directly and is kept set to a falsy value here)."""
    monkeypatch.delattr(config, "YTDLP_LANG", raising=False)
    monkeypatch.delattr(config, "YTDLP_COOKIES_FILE", raising=False)
    monkeypatch.delattr(config, "YTDLP_COOKIES_FROM_BROWSER", raising=False)

    opts = ytdlp_network_opts()

    assert opts == {}
