"""Shared pytest configuration for the goggles test suite."""

from __future__ import annotations

import os

import pytest

# MLflow 3.x ships usage telemetry enabled by default, which spawns a
# background ``MLflowTelemetryConsumer`` daemon thread on first client use.
# That thread intermittently allocates short-lived cyclic objects, which
# pollutes the strict process-global ``gc.collect()`` measurement in
# ``tests/core/test_logger.py::test_caller_id_does_not_leak_frame_cycles``
# (the count is process-wide, so any concurrent allocator perturbs it).
# CI already auto-disables MLflow telemetry; do the same for local runs so
# the suite is deterministic regardless of where it runs. Set at import time
# (before any test module imports MLflow) and via ``setdefault`` so an
# explicit developer override is preserved.
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
os.environ.setdefault("DO_NOT_TRACK", "true")


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
