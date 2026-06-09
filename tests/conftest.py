"""Shared pytest configuration for the goggles test suite."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _in_process_host_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Host in-process unless a test opts into the dedicated host.

    Running the host in a dedicated subprocess (see ``GOGGLES_DEDICATED_HOST``)
    is goggles' runtime default, but spawning a subprocess per test would be
    slow, non-deterministic, and would share the default socket across tests.
    The suite therefore hosts in-process by default; the tests that exercise
    the dedicated host re-enable it explicitly (see
    ``tests/core/test_dedicated_host.py``).
    """
    monkeypatch.setenv("GOGGLES_DEDICATED_HOST", "0")
