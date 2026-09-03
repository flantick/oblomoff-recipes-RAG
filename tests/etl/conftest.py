"""Fixtures shared by the ETL tests.

The ETL modules are the ones that talk to the outside world, so this is where
the guardrails live.
"""
from __future__ import annotations

import time

import pytest


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """Turns a real sleep into a test failure.

    fetch_via_ytdlp backs off for 30*attempt seconds on HTTP 429. The tests
    that exercise it install their own spy, but a future test written without
    one would quietly hang the suite for half a minute instead of failing —
    and a hang reads as "CI is slow today", not as "this test is wrong".

    Sub-100ms sleeps are left alone: those come from libraries, not from the
    retry policy under test.
    """
    real_sleep = time.sleep

    def guarded(seconds: float) -> None:
        if seconds > 0.1:
            raise AssertionError(
                f"a unit test tried to really sleep for {seconds}s — "
                "patch the sleep in the module under test instead"
            )
        real_sleep(seconds)

    monkeypatch.setattr(time, "sleep", guarded)
