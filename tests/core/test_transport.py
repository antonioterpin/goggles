"""Tests for the LocalTransport (cross-platform same-machine transport)."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import ClassVar, cast

import numpy as np
import pytest

import goggles as gg
from goggles import Event, Kind
from goggles._core.transport import (
    _IS_WINDOWS,
    LocalTransport,
    _default_socket_path,
    _pack_small,
    _try_unlink_shm,
    _unpack_small,
    _user_tag,
)


@pytest.fixture
def socket_path(tmp_path: Path) -> Iterator[str]:
    """Allocate a unique path for the transport endpoint.

    On Unix this is an AF_UNIX socket file; AF_UNIX paths are capped at
    ~104 chars on macOS, so ``tmp_path`` (under ``/private/var/folders``)
    is too long. We use a short path under the platform tempdir with a
    tmp_path-derived suffix for uniqueness, and clean up afterwards.
    On Windows the path is a plain discovery file; the length cap does
    not apply.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Yields:
        str: Absolute path to use as the transport endpoint.
    """
    suffix = os.path.basename(tmp_path)[-20:]
    base = tempfile.gettempdir() if _IS_WINDOWS else "/tmp"
    path = os.path.join(base, f"gg-{suffix}-{os.getpid()}.sock")
    try:
        yield path
    finally:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


class _CollectingHandler:
    """Minimal Handler protocol impl collecting events for assertion.

    Attributes:
        name: Handler identifier used by the bus's dedup logic.
        capabilities: Event kinds this handler claims to handle.
    """

    name = "collector"
    capabilities: ClassVar[frozenset[Kind]] = frozenset(
        {
            "log",
            "metric",
            "image",
            "video",
            "artifact",
            "histogram",
            "vector",
            "vector_field",
        }
    )

    def __init__(self) -> None:
        self.events: list[Event] = []
        self.lock = threading.Lock()

    def can_handle(self, kind: Kind) -> bool:
        del kind
        return True

    def handle(self, event: Event) -> None:
        with self.lock:
            self.events.append(event)

    def open(self) -> None: ...

    def close(self) -> None: ...

    def to_dict(self) -> dict:
        return {"cls": "_CollectingHandler", "data": {}}

    @classmethod
    def from_dict(cls, serialized: dict) -> _CollectingHandler:
        del serialized
        return cls()


def _wait_until(
    cond: Callable[[], bool],
    timeout: float = 2.0,
    interval: float = 0.01,
) -> bool:
    """Poll cond until true or timeout elapses.

    Args:
        cond: Callable returning True once the awaited condition is met.
        timeout: Total seconds to wait.
        interval: Seconds between polls.

    Returns:
        The final value of ``cond()`` (True on success, False on timeout).
    """
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(interval)
    return cond()


def _install_collector(
    transport: LocalTransport, collector: _CollectingHandler
) -> None:
    """Install a handler directly into a host-mode bus (bypassing attach).

    Args:
        transport: Host-mode transport that owns the bus.
        collector: Handler to wire under the ``global`` scope.
    """
    with transport._bus._lock:  # type: ignore[attr-defined]
        transport._bus.handlers[collector.name] = collector  # type: ignore[attr-defined]
        transport._bus.scopes["global"] = {collector.name}  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Portable helpers
# ---------------------------------------------------------------------------


def test_user_tag_is_portable() -> None:
    tag = _user_tag()
    assert tag
    assert isinstance(tag, str)


def test_default_socket_path_uses_tempdir_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GOGGLES_SOCKET", raising=False)
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    path = _default_socket_path()
    assert path.endswith(".sock")
    assert os.path.isabs(path)


# ---------------------------------------------------------------------------
# Framing unit tests
# ---------------------------------------------------------------------------


def test_pack_unpack_small_roundtrip_scalar() -> None:
    event = Event(
        kind="metric",
        scope="global",
        payload={"loss": 0.42},
        filepath="test.py",
        lineno=1,
        step=7,
    )
    data = _pack_small(event)
    restored = _unpack_small(data)
    assert restored.kind == "metric"
    assert restored.payload == {"loss": 0.42}
    assert restored.step == 7
    assert restored.scope == "global"


def test_pack_unpack_small_roundtrip_numpy() -> None:
    arr = np.arange(16 * 16, dtype=np.uint8).reshape(16, 16)
    event = Event(
        kind="image",
        scope="global",
        payload=arr,
        filepath="test.py",
        lineno=2,
    )
    data = _pack_small(event)
    restored = _unpack_small(data)
    assert isinstance(restored.payload, np.ndarray)
    assert restored.payload.shape == (16, 16)
    np.testing.assert_array_equal(restored.payload, arr)


# ---------------------------------------------------------------------------
# Host-mode tests (single process)
# ---------------------------------------------------------------------------


def test_host_mode_emit_dispatches_to_handler(socket_path: str) -> None:
    gg.register_handler(_CollectingHandler)
    transport = LocalTransport(socket_path=socket_path)
    try:
        assert transport.is_host, "first transport should become host"
        assert transport.is_running
        transport.attach(
            handlers=[_CollectingHandler().to_dict()], scopes=["global"]
        )
        collector = cast(
            _CollectingHandler,
            transport._bus.handlers["collector"],  # type: ignore[attr-defined]
        )

        event = Event(
            kind="log",
            scope="global",
            payload="hello",
            filepath="t.py",
            lineno=1,
        )
        transport.emit(event)

        assert _wait_until(lambda: len(collector.events) == 1), (
            "host-mode emit should dispatch to attached handler"
        )
        assert collector.events[0].payload == "hello"
    finally:
        transport.shutdown(timeout=2.0)
        assert not transport.is_running
        assert not Path(socket_path).exists(), (
            "host should clean up endpoint on shutdown"
        )


def test_shutdown_is_idempotent(socket_path: str) -> None:
    transport = LocalTransport(socket_path=socket_path)
    transport.shutdown(timeout=2.0)
    # A second shutdown must be a no-op, not raise.
    transport.shutdown(timeout=2.0)
    assert not transport.is_running


def test_host_emit_sync_dispatches_inline(socket_path: str) -> None:
    transport = LocalTransport(socket_path=socket_path)
    try:
        collector = _CollectingHandler()
        _install_collector(transport, collector)

        event = Event(
            kind="log",
            scope="global",
            payload="sync",
            filepath="t.py",
            lineno=1,
        )
        transport.emit_sync(event)
        # No wait needed: sync path must dispatch before returning.
        assert len(collector.events) == 1
        assert collector.events[0].payload == "sync"
    finally:
        transport.shutdown(timeout=2.0)


def test_handler_exception_does_not_kill_drain(socket_path: str) -> None:
    transport = LocalTransport(socket_path=socket_path)
    try:

        class Flaky(_CollectingHandler):
            @classmethod
            def from_dict(cls, serialized: dict) -> Flaky:
                del serialized
                return cls()

            def handle(self, event: Event) -> None:
                with self.lock:
                    self.events.append(event)
                    if len(self.events) == 1:
                        raise RuntimeError("boom")

        flaky = Flaky()
        _install_collector(transport, flaky)

        for i in range(3):
            transport.emit(
                Event(
                    kind="log",
                    scope="global",
                    payload=f"m{i}",
                    filepath="t.py",
                    lineno=i,
                )
            )

        assert _wait_until(lambda: len(flaky.events) == 3), (
            "drain thread must survive handler exceptions"
        )
    finally:
        transport.shutdown(timeout=2.0)


def test_concurrent_attach_detach_with_emit(socket_path: str) -> None:
    """EventBus must not corrupt state under concurrent attach+emit.

    Args:
        socket_path: Endpoint path (via fixture).
    """
    transport = LocalTransport(socket_path=socket_path)
    gg.register_handler(_CollectingHandler)
    try:
        stop = threading.Event()

        def emitter() -> None:
            i = 0
            while not stop.is_set():
                transport.emit(
                    Event(
                        kind="log",
                        scope="global",
                        payload=f"m{i}",
                        filepath="t.py",
                        lineno=i,
                    )
                )
                i += 1

        def attacher() -> None:
            name_idx = 0
            while not stop.is_set():
                scope = f"scope_{name_idx % 3}"
                transport.attach(
                    handlers=[_CollectingHandler().to_dict()],
                    scopes=[scope],
                )
                name_idx += 1
                try:
                    transport.detach("collector", scope)
                except ValueError:
                    pass

        threads = [
            threading.Thread(target=emitter, daemon=True),
            threading.Thread(target=attacher, daemon=True),
        ]
        for t in threads:
            t.start()
        time.sleep(0.3)
        stop.set()
        for t in threads:
            t.join(timeout=2.0)
            assert not t.is_alive()
    finally:
        transport.shutdown(timeout=2.0)


# ---------------------------------------------------------------------------
# Multi-process tests (host + client, same Python process)
# ---------------------------------------------------------------------------


def test_second_transport_connects_as_client(socket_path: str) -> None:
    host = LocalTransport(socket_path=socket_path)
    try:
        assert host.is_host
        client = LocalTransport(socket_path=socket_path)
        try:
            assert not client.is_host, (
                "second transport must connect as client, not rebind"
            )

            collector = _CollectingHandler()
            _install_collector(host, collector)

            event = Event(
                kind="log",
                scope="global",
                payload="from-client",
                filepath="t.py",
                lineno=1,
            )
            client.emit(event)

            assert _wait_until(lambda: len(collector.events) == 1), (
                "host should receive event from connected client"
            )
            assert collector.events[0].payload == "from-client"
        finally:
            client.shutdown(timeout=2.0)
    finally:
        host.shutdown(timeout=2.0)


def test_client_numpy_payload_roundtrip(socket_path: str) -> None:
    host = LocalTransport(socket_path=socket_path, shm_threshold=10**9)
    # Large threshold → force small-path (inline pickle) even for ndarray.
    try:
        client = LocalTransport(socket_path=socket_path, shm_threshold=10**9)
        try:
            collector = _CollectingHandler()
            _install_collector(host, collector)

            arr = np.arange(32 * 32, dtype=np.uint8).reshape(32, 32)
            client.emit(
                Event(
                    kind="image",
                    scope="global",
                    payload=arr,
                    filepath="t.py",
                    lineno=1,
                )
            )

            assert _wait_until(lambda: len(collector.events) == 1)
            got = collector.events[0].payload
            assert isinstance(got, np.ndarray)
            np.testing.assert_array_equal(got, arr)
        finally:
            client.shutdown(timeout=2.0)
    finally:
        host.shutdown(timeout=2.0)


def test_shm_side_channel_for_large_payload(socket_path: str) -> None:
    host = LocalTransport(socket_path=socket_path, shm_threshold=1024)
    try:
        client = LocalTransport(socket_path=socket_path, shm_threshold=1024)
        try:
            collector = _CollectingHandler()
            _install_collector(host, collector)

            arr = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
            assert arr.nbytes > 1024
            client.emit(
                Event(
                    kind="image",
                    scope="global",
                    payload=arr,
                    filepath="t.py",
                    lineno=1,
                )
            )

            assert _wait_until(lambda: len(collector.events) == 1)
            got = collector.events[0].payload
            assert isinstance(got, np.ndarray)
            assert got.shape == arr.shape
            assert got.dtype == arr.dtype
            np.testing.assert_array_equal(got, arr)
            # Client must have released its pending-shm tracking.
            assert _wait_until(
                lambda: not client._pending_shm,  # type: ignore[attr-defined]
                timeout=1.0,
            )
        finally:
            client.shutdown(timeout=2.0)
    finally:
        host.shutdown(timeout=2.0)


def test_shutdown_flushes_pending_events(socket_path: str) -> None:
    """Every enqueued event must reach the host before shutdown returns.

    Regresses the "77/1500 video logs arrived" bug: BYE was sent out-of-band
    during shutdown, so any frames still in the client's send queue were
    discarded when the send thread was force-closed.

    Args:
        socket_path: Endpoint path (via fixture).
    """
    host = LocalTransport(socket_path=socket_path)
    try:
        client = LocalTransport(socket_path=socket_path)
        try:
            collector = _CollectingHandler()
            _install_collector(host, collector)

            n = 500
            for i in range(n):
                client.emit(
                    Event(
                        kind="log",
                        scope="global",
                        payload=f"m{i}",
                        filepath="t.py",
                        lineno=i,
                    )
                )
            client.shutdown(timeout=10.0)
            # After client.shutdown() returns, every frame must have been
            # handed to the host. Give the drain thread a moment to finish.
            assert _wait_until(
                lambda: len(collector.events) == n, timeout=5.0
            ), f"host got {len(collector.events)}/{n}"
        finally:
            # client already shut down
            pass
    finally:
        host.shutdown(timeout=5.0)


def test_shutdown_flushes_shm_frames(socket_path: str) -> None:
    """LARGE frames must also flush on graceful shutdown (no shm leak).

    Args:
        socket_path: Endpoint path (via fixture).
    """
    host = LocalTransport(socket_path=socket_path, shm_threshold=1024)
    try:
        client = LocalTransport(socket_path=socket_path, shm_threshold=1024)
        try:
            collector = _CollectingHandler()
            _install_collector(host, collector)

            n = 20
            arr = np.zeros((64, 64), dtype=np.float32)  # 16 KiB > threshold
            for i in range(n):
                client.emit(
                    Event(
                        kind="image",
                        scope="global",
                        payload=arr + i,
                        filepath="t.py",
                        lineno=i,
                    )
                )
            client.shutdown(timeout=10.0)
            assert _wait_until(
                lambda: len(collector.events) == n, timeout=5.0
            ), f"host got {len(collector.events)}/{n}"
            # No leaked shm references on the client side.
            assert not client._pending_shm  # type: ignore[attr-defined]
        finally:
            pass
    finally:
        host.shutdown(timeout=5.0)


def test_shm_unlinked_on_client_send_failure(socket_path: str) -> None:
    """If the send socket dies mid-stream, the client reaps its shm.

    Recovery happens in two phases: the send thread unlinks everything
    currently in the queue when sendall fails, and ``shutdown`` sweeps
    anything that was emitted concurrently with the send thread dying.
    Together the two phases must reap every shm block the client allocated.

    Args:
        socket_path: Endpoint path (via fixture).
    """
    host = LocalTransport(socket_path=socket_path, shm_threshold=1024)
    client = LocalTransport(socket_path=socket_path, shm_threshold=1024)
    host.shutdown(timeout=2.0)
    time.sleep(0.1)

    arr = np.zeros((64, 64), dtype=np.float32)  # > 1024 bytes
    for i in range(5):
        client.emit(
            Event(
                kind="image",
                scope="global",
                payload=arr + i,
                filepath="t.py",
                lineno=i,
            )
        )

    client.shutdown(timeout=3.0)
    assert not client._pending_shm, (  # type: ignore[attr-defined]
        f"leaked shm names after shutdown: {client._pending_shm}"  # type: ignore[attr-defined]
    )


# ---------------------------------------------------------------------------
# Host election: stale endpoint cleanup
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    _IS_WINDOWS,
    reason="Unix-only: stale AF_UNIX file is a filesystem concern",
)
def test_stale_socket_is_cleaned_up(socket_path: str) -> None:
    Path(socket_path).write_bytes(b"stale")
    transport = LocalTransport(socket_path=socket_path)
    try:
        assert transport.is_host, (
            "transport should unlink stale file and become host"
        )
    finally:
        transport.shutdown(timeout=2.0)


@pytest.mark.skipif(
    _IS_WINDOWS,
    reason="Unix-only: socket permissions rely on POSIX file modes",
)
def test_unix_socket_is_owner_only(socket_path: str) -> None:
    """AF_UNIX socket file must be 0o600 (no world/group access).

    Args:
        socket_path: Endpoint path (via fixture).
    """
    transport = LocalTransport(socket_path=socket_path)
    try:
        assert transport.is_host
        mode = os.stat(socket_path).st_mode & 0o777
        # Under a restrictive umask we expect exactly 0o600; tolerate
        # environments that narrow it further (0o600 or less-permissive).
        assert mode & 0o077 == 0, f"socket too open: {oct(mode)}"
    finally:
        transport.shutdown(timeout=2.0)


# ---------------------------------------------------------------------------
# Independent-process verification
# ---------------------------------------------------------------------------


HOST_WORKER_SRC = """
import os
import sys
import time
from pathlib import Path

from goggles import Event
from goggles._core.transport import LocalTransport


class Sink:
    name = "sink"
    capabilities = frozenset([
        "log", "metric", "image", "video", "artifact",
        "histogram", "vector", "vector_field"
    ])

    def __init__(self, out_path):
        self.out = Path(out_path)
        self.count = 0

    def can_handle(self, _k):
        return True

    def handle(self, event):
        self.count += 1
        self.out.write_text(str(self.count))

    def open(self): ...
    def close(self): ...
    def to_dict(self): return {"cls": "Sink", "data": {}}

    @classmethod
    def from_dict(cls, _d): return cls(out_path=os.environ["OUT_PATH"])


socket_path = os.environ["GOGGLES_SOCKET"]
out_path = os.environ["OUT_PATH"]
expected = int(os.environ.get("EXPECTED", "1"))
transport = LocalTransport(socket_path=socket_path)
assert transport.is_host
sink = Sink(out_path=out_path)
with transport._bus._lock:
    transport._bus.handlers[sink.name] = sink
    transport._bus.scopes["global"] = {sink.name}
Path(os.environ["READY_PATH"]).write_text("ready")
deadline = time.time() + 30
while time.time() < deadline:
    if sink.count >= expected:
        break
    time.sleep(0.05)
transport.shutdown(timeout=5.0)
print(sink.count)
"""


def _launch_host_subprocess(
    socket_path: str, tmp_path: Path, expected: int
) -> tuple[subprocess.Popen, Path]:
    out_path = tmp_path / "count.txt"
    ready_path = tmp_path / "ready.txt"
    worker_script = tmp_path / "worker.py"
    worker_script.write_text(HOST_WORKER_SRC)

    env = os.environ.copy()
    env["GOGGLES_SOCKET"] = socket_path
    env["OUT_PATH"] = str(out_path)
    env["READY_PATH"] = str(ready_path)
    env["EXPECTED"] = str(expected)

    proc = subprocess.Popen(
        [sys.executable, str(worker_script)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert _wait_until(ready_path.exists, timeout=15.0), (
        "host subprocess failed to initialize"
    )
    return proc, out_path


def test_two_independent_processes_share_host(
    socket_path: str, tmp_path: Path
) -> None:
    proc, out_path = _launch_host_subprocess(socket_path, tmp_path, expected=1)
    try:
        client = LocalTransport(socket_path=socket_path)
        try:
            assert not client.is_host
            client.emit(
                Event(
                    kind="log",
                    scope="global",
                    payload="cross-proc",
                    filepath="t.py",
                    lineno=1,
                )
            )
            assert _wait_until(
                lambda: out_path.exists() and out_path.read_text() == "1",
                timeout=10.0,
            ), "host process should have recorded 1 event"
        finally:
            client.shutdown(timeout=5.0)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def test_cross_process_no_loss_at_bulk_send(
    socket_path: str, tmp_path: Path
) -> None:
    """Client emits N events then shuts down; host process must see N.

    This is the multi-process regression test for the "77/1500" bug.

    Args:
        socket_path: Endpoint path (via fixture).
        tmp_path: pytest's per-test temporary directory.
    """
    n = 500
    proc, out_path = _launch_host_subprocess(socket_path, tmp_path, expected=n)
    try:
        client = LocalTransport(socket_path=socket_path)
        try:
            assert not client.is_host
            for i in range(n):
                client.emit(
                    Event(
                        kind="log",
                        scope="global",
                        payload=f"m{i}",
                        filepath="t.py",
                        lineno=i,
                    )
                )
        finally:
            client.shutdown(timeout=15.0)

        # By the time shutdown returns, BYE has been flushed; give the
        # host subprocess a beat to drain and record the final count.
        assert _wait_until(
            lambda: out_path.exists() and out_path.read_text() == str(n),
            timeout=15.0,
        ), (
            f"host recorded "
            f"{out_path.read_text() if out_path.exists() else '<no file>'}/"
            f"{n}"
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def test_cross_process_multiple_concurrent_clients(
    socket_path: str, tmp_path: Path
) -> None:
    """Two independent client transports sharing one host subprocess.

    Args:
        socket_path: Endpoint path (via fixture).
        tmp_path: pytest's per-test temporary directory.
    """
    n_per_client = 100
    total = n_per_client * 2
    proc, out_path = _launch_host_subprocess(
        socket_path, tmp_path, expected=total
    )
    try:
        c1 = LocalTransport(socket_path=socket_path)
        c2 = LocalTransport(socket_path=socket_path)
        try:
            for c in (c1, c2):
                assert not c.is_host

            def emit_many(client: LocalTransport, tag: str) -> None:
                for i in range(n_per_client):
                    client.emit(
                        Event(
                            kind="log",
                            scope="global",
                            payload=f"{tag}-{i}",
                            filepath="t.py",
                            lineno=i,
                        )
                    )

            t1 = threading.Thread(target=emit_many, args=(c1, "a"))
            t2 = threading.Thread(target=emit_many, args=(c2, "b"))
            t1.start()
            t2.start()
            t1.join()
            t2.join()
        finally:
            c1.shutdown(timeout=10.0)
            c2.shutdown(timeout=10.0)

        assert _wait_until(
            lambda: out_path.exists() and out_path.read_text() == str(total),
            timeout=15.0,
        ), (
            f"host recorded "
            f"{out_path.read_text() if out_path.exists() else '<no file>'}/"
            f"{total}"
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


# ---------------------------------------------------------------------------
# Shared-memory helpers
# ---------------------------------------------------------------------------


def test_try_unlink_shm_tolerates_missing_name() -> None:
    # Should be a no-op, not raise.
    _try_unlink_shm("this-name-does-not-exist-12345")
