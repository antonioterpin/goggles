"""Tests for the dedicated host process (default; ``GOGGLES_DEDICATED_HOST``).

By default :func:`goggles._core.routing.get_bus` spawns a dedicated
:mod:`goggles._core.host` subprocess to own the EventBus + handlers, so the
application is a client and handler work (e.g. W&B uploads) runs off the
application's interpreter. ``GOGGLES_DEDICATED_HOST=0`` opts out (in-process
host). :func:`goggles.finish` drains and reaps the subprocess.

The suite-wide autouse fixture in ``tests/conftest.py`` disables the dedicated
host by default; these tests re-enable it (or assert the disabled path)
explicitly.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
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
from goggles._core.transport._local import _host_idle_timeout_s


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
def default_host_socket(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """A unique socket with the dedicated host left at its default (on)."""
    path = _unique_socket()
    # Clear the suite-wide opt-out so we exercise the real default (on).
    monkeypatch.delenv("GOGGLES_DEDICATED_HOST", raising=False)
    monkeypatch.setenv("GOGGLES_SOCKET", path)
    # Bound the host's graceful-shutdown budget so tests never hang.
    monkeypatch.setenv("GOGGLES_SHUTDOWN_TIMEOUT", "10")
    # Long idle grace by default so the host never self-reaps mid-test; tests
    # that exercise the self-reap path override this to a small value.
    monkeypatch.setenv("GOGGLES_HOST_IDLE_TIMEOUT", "60")
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
def disabled_host_socket(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """A unique socket with the dedicated host explicitly disabled."""
    path = _unique_socket()
    monkeypatch.setenv("GOGGLES_DEDICATED_HOST", "0")
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


def test_dedicated_host_enabled_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Default (unset) is ON.
    monkeypatch.delenv("GOGGLES_DEDICATED_HOST", raising=False)
    assert routing._dedicated_host_enabled() is True
    # Only explicit falsy values disable it.
    for value in ("0", "false", "no", "off", "OFF", "False"):
        monkeypatch.setenv("GOGGLES_DEDICATED_HOST", value)
        assert routing._dedicated_host_enabled() is False, value
    for value in ("1", "true", "yes", "on", "anything"):
        monkeypatch.setenv("GOGGLES_DEDICATED_HOST", value)
        assert routing._dedicated_host_enabled() is True, value


def test_default_runs_host_in_a_subprocess(default_host_socket: str) -> None:
    bus = gg.get_bus()
    # The application is a CLIENT by default; the subprocess is the host.
    assert not bus.is_host
    proc = routing._dedicated_host_process()
    assert proc is not None
    assert proc.poll() is None, "host subprocess should be alive"


def test_disabled_hosts_in_process(disabled_host_socket: str) -> None:
    bus = gg.get_bus()
    try:
        assert bus.is_host, "GOGGLES_DEDICATED_HOST=0 hosts in-process"
        assert routing._dedicated_host_process() is None
    finally:
        bus.shutdown(timeout=5.0)
        routing.reset_bus()


def test_host_self_reaps_and_flushes_after_last_client(
    default_host_socket: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Short idle grace so the host self-reaps promptly once its client leaves.
    monkeypatch.setenv("GOGGLES_HOST_IDLE_TIMEOUT", "1")
    # Capture the host's output to a file (exercises the GOGGLES_HOST_LOG path).
    monkeypatch.setenv("GOGGLES_HOST_LOG", str(tmp_path / "host.log"))
    storage_dir = tmp_path / "logs"
    gg.attach(LocalStorageHandler(path=storage_dir), scopes=["global"])

    bus = gg.get_bus()
    assert not bus.is_host
    proc = routing._dedicated_host_process()
    assert proc is not None

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

    # finish() ships this client's events to the host and disconnects; it does
    # NOT reap the shared host. With its last client gone, the host self-reaps
    # after the idle grace -- draining the queue and closing the handler (where
    # the jsonl is flushed), proving the reconstructed handler ran in the
    # subprocess.
    gg.finish(timeout=10.0)

    assert _wait_until(lambda: proc.poll() is not None, timeout=10.0), (
        "host should self-reap after its last client disconnects"
    )
    log_file = storage_dir / "log.jsonl"
    assert log_file.exists() and len(log_file.read_text().splitlines()) == 5, (
        "the host should flush every event when it self-reaps"
    )


# A second, independent client process: connects to the host, signals it is
# up, then stays connected until a stop file appears. It builds the transport
# directly so it is unambiguously a CLIENT (never spawns/owns a host).
_CLIENT_B_SRC = """
import os, time
from goggles._core.transport import LocalTransport

t = LocalTransport(socket_path=os.environ["GG_SOCKET"])
if t.is_host:
    raise SystemExit("client B unexpectedly became the host")
open(os.environ["GG_READY"], "w").close()
stop = os.environ["GG_STOP"]
while not os.path.exists(stop):
    time.sleep(0.02)
t.shutdown(timeout=5.0)
"""


def test_host_survives_spawner_finish_until_last_client(
    default_host_socket: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The host the spawner reaped on ``finish()`` is exactly what fragmented
    multi-process runs; it must now outlive the spawner while any client is
    connected, never respawn, and self-reap only after the last client leaves.
    """
    monkeypatch.setenv("GOGGLES_HOST_IDLE_TIMEOUT", "1")
    socket_path = os.environ["GOGGLES_SOCKET"]

    # This process spawns and "owns" the host.
    bus = gg.get_bus()
    assert not bus.is_host
    proc = routing._dedicated_host_process()
    assert proc is not None

    # A second, independent client connects and stays connected.
    ready = tmp_path / "b_ready"
    stop = tmp_path / "b_stop"
    script = tmp_path / "client_b.py"
    script.write_text(_CLIENT_B_SRC)
    env = dict(
        os.environ,
        GG_SOCKET=socket_path,
        GG_READY=str(ready),
        GG_STOP=str(stop),
    )
    client_b = subprocess.Popen([sys.executable, str(script)], env=env)
    try:
        assert _wait_until(ready.exists, timeout=10.0), (
            "client B never connected"
        )

        # The spawner finishes. The host must NOT die -- B is still connected.
        gg.finish(timeout=10.0)

        # Past the idle grace, the host is still the same live process: B kept
        # it alive, finish() did not reap it, and nobody respawned a second one.
        time.sleep(2.0)  # > GOGGLES_HOST_IDLE_TIMEOUT
        assert proc.poll() is None, "host must stay alive while a client is on"
        assert routing._host_is_listening(socket_path)
        assert routing._dedicated_host_process() is proc, "no respawn"

        # Release B: the host's last client is gone -> it self-reaps.
        stop.touch()
        assert _wait_until(lambda: proc.poll() is not None, timeout=10.0), (
            "host should self-reap once its last client disconnects"
        )
    finally:
        stop.touch()  # ensure B exits even if an assertion failed
        try:
            client_b.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            client_b.kill()


def test_does_not_spawn_when_a_host_already_listens(
    default_host_socket: str,
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
    default_host_socket: str, monkeypatch: pytest.MonkeyPatch
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


def test_configure_replaces_console_under_dedicated_host(
    default_host_socket: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Under the dedicated host the local bus is a client (empty), so
    # configure() must not read it to decide whether to replace the console
    # handler -- it detaches the target scope unconditionally (over the wire)
    # before re-attaching, so a second configure() actually wins.
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(gg, "detach", lambda n, s: calls.append((n, s)))
    try:
        gg.configure(enable_console=True, scopes=["global"])
        assert (gg.ConsoleHandler.name, "global") in calls
    finally:
        gg.finish(timeout=10.0)


def test_falls_back_to_in_process_host_when_spawn_raises(
    default_host_socket: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*_a: object, **_k: object) -> object:
        raise OSError("cannot spawn (restricted environment)")

    monkeypatch.setattr(routing.subprocess, "Popen", _boom)

    # A spawn failure must NOT crash the caller's first log; it falls back to
    # an in-process host.
    bus = gg.get_bus()
    try:
        assert bus.is_host
        assert routing._dedicated_host_process() is None
    finally:
        bus.shutdown(timeout=5.0)
        routing.reset_bus()


# --- self-reap internals -----------------------------------------------------
# The real dedicated host runs the self-reap inside a subprocess coverage can't
# see, so exercise the LocalTransport machinery against an in-process host here.


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, 5.0), ("2.5", 2.5), ("bad", 5.0), ("0", 5.0), ("-1", 5.0)],
)
def test_host_idle_timeout_parsing(
    monkeypatch: pytest.MonkeyPatch, value: str | None, expected: float
) -> None:
    if value is None:
        monkeypatch.delenv("GOGGLES_HOST_IDLE_TIMEOUT", raising=False)
    else:
        monkeypatch.setenv("GOGGLES_HOST_IDLE_TIMEOUT", value)
    assert _host_idle_timeout_s() == expected


@pytest.fixture
def inproc_host(monkeypatch: pytest.MonkeyPatch) -> Iterator[LocalTransport]:
    """An in-process host bound to a unique socket (no subprocess)."""
    monkeypatch.setenv("GOGGLES_DEDICATED_HOST", "0")
    path = _unique_socket()
    monkeypatch.setenv("GOGGLES_SOCKET", path)
    host = LocalTransport(socket_path=path)
    assert host.is_host
    try:
        yield host
    finally:
        host.shutdown(timeout=5.0)
        for suffix in ("", ".lock"):
            try:
                os.unlink(path + suffix)
            except OSError:
                pass


def test_idle_callback_fires_when_no_clients(
    inproc_host: LocalTransport,
) -> None:
    inproc_host._idle_timeout_s = 0.1
    fired = threading.Event()
    inproc_host.set_idle_callback(lambda: None)  # arms a timer
    inproc_host.set_idle_callback(fired.set)  # re-arms (cancels the first)
    assert fired.wait(timeout=2.0)


def test_idle_timer_rearms_on_last_client_disconnect(
    inproc_host: LocalTransport,
) -> None:
    client = LocalTransport(socket_path=inproc_host._socket_path)
    try:
        assert not client.is_host
        assert _wait_until(
            lambda: len(inproc_host._client_sockets) == 1, timeout=3.0
        )
        inproc_host._idle_timeout_s = 0.2
        fired = threading.Event()
        inproc_host.set_idle_callback(fired.set)  # client present -> no arm
        assert not fired.wait(timeout=0.5)
    finally:
        client.shutdown(timeout=3.0)  # last client gone -> re-arms -> fires
    assert fired.wait(timeout=3.0)


def test_idle_timer_cancelled_by_new_client(
    inproc_host: LocalTransport,
) -> None:
    inproc_host._idle_timeout_s = 1.0
    fired = threading.Event()
    inproc_host.set_idle_callback(fired.set)  # no clients -> arms
    client = LocalTransport(socket_path=inproc_host._socket_path)
    try:
        # The accept loop cancels the armed timer, so it never fires.
        assert not fired.wait(timeout=1.5)
    finally:
        client.shutdown(timeout=3.0)


def test_reap_if_idle_is_safe_without_a_callback(
    inproc_host: LocalTransport,
) -> None:
    # No callback installed and no clients -> the defensive None branch.
    inproc_host._reap_if_idle()


def test_reap_if_idle_is_noop_while_a_client_is_connected(
    inproc_host: LocalTransport,
) -> None:
    client = LocalTransport(socket_path=inproc_host._socket_path)
    try:
        assert _wait_until(
            lambda: len(inproc_host._client_sockets) == 1, timeout=3.0
        )
        fired = threading.Event()
        inproc_host._idle_callback = fired.set
        inproc_host._reap_if_idle()  # a client is connected -> no-op
        assert not fired.is_set()
    finally:
        client.shutdown(timeout=3.0)


def test_open_host_log_defaults_to_devnull(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GOGGLES_HOST_LOG", raising=False)
    assert routing._open_host_log() is subprocess.DEVNULL


def test_open_host_log_opens_the_configured_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log_path = tmp_path / "host.log"
    monkeypatch.setenv("GOGGLES_HOST_LOG", str(log_path))
    target = routing._open_host_log()
    assert not isinstance(target, int)  # a real file, not DEVNULL
    try:
        target.write("hi\n")
    finally:
        target.close()
    assert log_path.exists()


def test_open_host_log_falls_back_when_path_unopenable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "GOGGLES_HOST_LOG", str(tmp_path / "no" / "such" / "dir.log")
    )
    assert routing._open_host_log() is subprocess.DEVNULL


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
