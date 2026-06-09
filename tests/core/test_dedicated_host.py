"""Tests for the dedicated host process (``GOGGLES_DEDICATED_HOST``).

When the env var is set, :func:`goggles._core.routing.get_bus` spawns a
dedicated :mod:`goggles._core.host` subprocess to own the EventBus + handlers,
so the application is a client and handler work (e.g. W&B uploads) runs off the
application's interpreter. :func:`goggles.finish` reaps the subprocess.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

import goggles as gg
from goggles import Event, LocalStorageHandler
from goggles._core import host as host_mod
from goggles._core import routing
from goggles._core.transport import LocalTransport
from goggles._core.transport._frames import _IS_WINDOWS


def _unique_socket() -> str:
    base = tempfile.gettempdir() if _IS_WINDOWS else "/tmp"
    return os.path.join(base, f"gg-ded-{uuid.uuid4().hex[:12]}.sock")


def _wait_until(
    cond: Callable[[], bool], timeout: float = 5.0, interval: float = 0.02
) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(interval)
    return cond()


@pytest.fixture
def dedicated_socket(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    path = _unique_socket()
    monkeypatch.setenv("GOGGLES_DEDICATED_HOST", "1")
    monkeypatch.setenv("GOGGLES_SOCKET", path)
    # Bound the host's graceful-shutdown budget so tests never hang.
    monkeypatch.setenv("GOGGLES_SHUTDOWN_TIMEOUT", "10")
    routing.reset_bus()
    try:
        yield path
    finally:
        routing._terminate_dedicated_host(timeout=10.0)
        routing.reset_bus()
        for suffix in ("", ".lock"):
            try:
                os.unlink(path + suffix)
            except OSError:
                pass


@pytest.fixture
def plain_socket(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    path = _unique_socket()
    monkeypatch.delenv("GOGGLES_DEDICATED_HOST", raising=False)
    monkeypatch.setenv("GOGGLES_SOCKET", path)
    routing.reset_bus()
    try:
        yield path
    finally:
        routing.reset_bus()
        try:
            os.unlink(path)
        except OSError:
            pass


def test_disabled_by_default_hosts_in_process(plain_socket: str) -> None:
    bus = gg.get_bus()
    try:
        assert bus.is_host, "without the env var the caller hosts in-process"
        assert routing._dedicated_host_process() is None
    finally:
        bus.shutdown(timeout=5.0)
        routing.reset_bus()


def test_spawns_host_and_runs_handler_in_subprocess(
    dedicated_socket: str, tmp_path: Path
) -> None:
    storage_dir = tmp_path / "logs"
    gg.attach(LocalStorageHandler(path=storage_dir), scopes=["global"])

    bus = gg.get_bus()
    # The application is a CLIENT; the dedicated subprocess is the host.
    assert not bus.is_host
    proc = routing._dedicated_host_process()
    assert proc is not None
    assert proc.poll() is None, "host subprocess should be alive"

    for i in range(5):
        bus.emit(
            Event(
                kind="log",
                scope="global",
                payload=f"event-{i}",
                filepath="test.py",
                lineno=i,
            )
        )

    # finish() flushes this client AND drains the host before reaping it, so
    # every event lands in the jsonl even though we emit + finish immediately.
    # The LocalStorageHandler was reconstructed and runs INSIDE the host
    # subprocess (we are a client), so this also proves the handler ran there.
    gg.finish(timeout=10.0)

    assert proc.poll() is not None, "host subprocess should be reaped"
    log_file = storage_dir / "log.jsonl"
    assert log_file.exists() and len(log_file.read_text().splitlines()) == 5, (
        "finish() should flush every event through the host before reaping"
    )


def test_does_not_spawn_when_a_host_already_listens(
    dedicated_socket: str,
) -> None:
    # A host is already bound to the socket (e.g. an externally managed one).
    existing = LocalTransport(socket_path=os.environ["GOGGLES_SOCKET"])
    bus = None
    try:
        assert existing.is_host
        routing.reset_bus()
        bus = gg.get_bus()
        assert not bus.is_host, "should connect to the existing host"
        assert routing._dedicated_host_process() is None, (
            "must not spawn a second host when one already listens"
        )
    finally:
        if bus is not None:
            bus.shutdown(timeout=5.0)
        existing.shutdown(timeout=5.0)
        routing.reset_bus()


def test_falls_back_to_in_process_host_when_spawn_not_ready(
    dedicated_socket: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _DummyProc:
        def __init__(self) -> None:
            self._alive = True

        def poll(self) -> int | None:
            return None if self._alive else 0

        def terminate(self) -> None:
            self._alive = False

        def wait(self, timeout: float | None = None) -> int:
            self._alive = False
            return 0

        def kill(self) -> None:
            self._alive = False

    dummy = _DummyProc()
    monkeypatch.setattr(routing.subprocess, "Popen", lambda *a, **k: dummy)
    monkeypatch.setattr(routing, "_wait_for_ready", lambda proc, path: False)

    bus = gg.get_bus()
    try:
        # Spawn never became ready -> we fall back to an in-process host.
        assert bus.is_host
        assert routing._dedicated_host_process() is None
        assert not dummy._alive, "the unready child must be killed"
    finally:
        bus.shutdown(timeout=5.0)
        routing.reset_bus()


def test_terminate_dedicated_host_is_idempotent() -> None:
    # No host spawned by this process -> both calls are safe no-ops.
    routing._terminate_dedicated_host()
    routing._terminate_dedicated_host(timeout=1.0)
    assert routing._dedicated_host_process() is None


# ----- host module helpers -------------------------------------------------


def test_host_shutdown_timeout_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOGGLES_SHUTDOWN_TIMEOUT", "5")
    assert host_mod._shutdown_timeout() == 5.0
    monkeypatch.setenv("GOGGLES_SHUTDOWN_TIMEOUT", "0")
    assert host_mod._shutdown_timeout() is None
    monkeypatch.setenv("GOGGLES_SHUTDOWN_TIMEOUT", "not-a-number")
    assert host_mod._shutdown_timeout() is None
    monkeypatch.delenv("GOGGLES_SHUTDOWN_TIMEOUT", raising=False)
    assert host_mod._shutdown_timeout() is None


def test_host_imports_named_modules(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    marker = f"GG_IMPORTED_{uuid.uuid4().hex[:8]}"
    module_name = f"gg_himport_{uuid.uuid4().hex[:8]}"
    (tmp_path / f"{module_name}.py").write_text(
        f"import os\nos.environ[{marker!r}] = 'yes'\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delenv(marker, raising=False)

    monkeypatch.setenv(
        "GOGGLES_HOST_IMPORTS", f"{module_name}, nonexistent_xyz"
    )
    # Imports the good module and skips the bad one without raising.
    host_mod._import_host_modules()
    assert os.environ.get(marker) == "yes"
    monkeypatch.delenv(marker, raising=False)


def test_kill_reaps_a_live_process() -> None:
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"]
    )
    routing._kill(proc)
    assert proc.poll() is not None
    # Calling again on an already-dead process is a safe no-op.
    routing._kill(proc)
