"""shared pytest configuration.

Skips the live-API test files when MISTRAL_API_KEY is unset, so the offline
test suite (`make test-all`) is green on machines without a key configured.
The skip is targeted at modules that actually need the key -- the other
unit tests (router rules, output cleaning, etc.) continue to run.
"""
from __future__ import annotations

import os

import pytest


_LIVE_API_MODULES = (
    "tests.test_adaptive_v2",
    "tests.test_fixes_phase3",
)


def pytest_collection_modifyitems(config, items):  # noqa: D401  -- pytest hook
    """Mark live-API tests as skipped when no Mistral key is available."""
    if os.getenv("MISTRAL_API_KEY"):
        return
    skip_marker = pytest.mark.skip(reason="MISTRAL_API_KEY not set -- live-API test")
    for item in items:
        if item.module.__name__ in _LIVE_API_MODULES:
            item.add_marker(skip_marker)
