"""Tests for the LocalTransport (Unix-socket-based transport)."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import cast

from typing import ClassVar

import numpy as np
import pytest

import goggles as gg
from goggles import Event, Kind
from goggles._core.transport import (
    LocalTransport,
    _pack_small,
    _unpack_small,
)


@pytest.fixture
def socket_path(tmp_path: Path) -> "Iterator[str]":
    """Allocate a unique Unix socket path short enough for AF_UNIX.

    macOS limits AF_UNIX paths to ~104 chars, so ``tmp_path`` (which nests
    under ``/private/var/folders/...``) is too long. Use a short path under
    ``/tmp`` with a tmp_path-derived suffix for uniqueness; ensure cleanup.
    """
    suffix = os.path.basename(tmp_path)[-20:]
    path = f"/tmp/gg-{suffix}-{os.getpid()}.sock"
    try:
        yield path
    finally:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


class _CollectingHandler:
    """Minimal Handler protocol impl collecting events for assertion."""

    name = "collector"
    capabilities: ClassVar[frozenset[Kind]] = frozenset(
        {"log", "metric", "image", "video", "artifact", "histogram",
         "vector", "vector_field"}
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
    def from_dict(cls, serialized: dict) -> "_CollectingHandler":
        del serialized
        return cls()


def _wait_until(
    cond: "Callable[[], bool]",
    timeout: float = 2.0,
    interval: float = 0.01,
) -> bool:
    """Poll cond until true or timeout elapses."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(interval)
    return cond()


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
        transport.attach(
            handlers=[_CollectingHandler().to_dict()], scopes=["global"]
        )
        # Reach inside to get the exact instance kept by the bus.
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
        assert not Path(socket_path).exists(), (
            "host should unlink socket on shutdown"
        )


def test_host_emit_sync_dispatches_inline(socket_path: str) -> None:
    transport = LocalTransport(socket_path=socket_path)
    try:
        collector = _CollectingHandler()
        transport._bus.handlers[collector.name] = collector  # type: ignore[attr-defined]
        transport._bus.scopes["global"] = {collector.name}  # type: ignore[attr-defined]

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


def test_handler_exception_does_not_kill_reader(socket_path: str) -> None:
    transport = LocalTransport(socket_path=socket_path)
    try:
        class Flaky(_CollectingHandler):
            @classmethod
            def from_dict(cls, serialized: dict) -> "Flaky":
                del serialized
                return cls()

            def handle(self, event: Event) -> None:
                with self.lock:
                    self.events.append(event)
                    if len(self.events) == 1:
                        raise RuntimeError("boom")

        flaky = Flaky()
        transport._bus.handlers[flaky.name] = flaky  # type: ignore[attr-defined]
        transport._bus.scopes["global"] = {flaky.name}  # type: ignore[attr-defined]

        for i in range(3):
            transport.emit(Event(
                kind="log", scope="global", payload=f"m{i}",
                filepath="t.py", lineno=i,
            ))

        assert _wait_until(lambda: len(flaky.events) == 3), (
            "drain thread must survive handler exceptions"
        )
    finally:
        transport.shutdown(timeout=2.0)


# ---------------------------------------------------------------------------
# Multi-process tests (host + client)
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
            host._bus.handlers[collector.name] = collector  # type: ignore[attr-defined]
            host._bus.scopes["global"] = {collector.name}  # type: ignore[attr-defined]

            event = Event(
                kind="log", scope="global", payload="from-client",
                filepath="t.py", lineno=1,
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
            host._bus.handlers[collector.name] = collector  # type: ignore[attr-defined]
            host._bus.scopes["global"] = {collector.name}  # type: ignore[attr-defined]

            arr = np.arange(32 * 32, dtype=np.uint8).reshape(32, 32)
            client.emit(Event(
                kind="image", scope="global", payload=arr,
                filepath="t.py", lineno=1,
            ))

            assert _wait_until(lambda: len(collector.events) == 1)
            got = collector.events[0].payload
            assert isinstance(got, np.ndarray)
            np.testing.assert_array_equal(got, arr)
        finally:
            client.shutdown(timeout=2.0)
    finally:
        host.shutdown(timeout=2.0)


def test_shm_side_channel_for_large_payload(socket_path: str) -> None:
    # Use a small threshold so even modestly-sized arrays take the shm path.
    host = LocalTransport(socket_path=socket_path, shm_threshold=1024)
    try:
        client = LocalTransport(socket_path=socket_path, shm_threshold=1024)
        try:
            collector = _CollectingHandler()
            host._bus.handlers[collector.name] = collector  # type: ignore[attr-defined]
            host._bus.scopes["global"] = {collector.name}  # type: ignore[attr-defined]

            arr = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
            assert arr.nbytes > 1024
            client.emit(Event(
                kind="image", scope="global", payload=arr,
                filepath="t.py", lineno=1,
            ))

            assert _wait_until(lambda: len(collector.events) == 1)
            got = collector.events[0].payload
            assert isinstance(got, np.ndarray)
            assert got.shape == arr.shape
            assert got.dtype == arr.dtype
            np.testing.assert_array_equal(got, arr)
            # Give the host a beat to release the shm block after handling.
            time.sleep(0.1)
        finally:
            client.shutdown(timeout=2.0)
    finally:
        host.shutdown(timeout=2.0)


# ---------------------------------------------------------------------------
# Host election: stale socket file cleanup
# ---------------------------------------------------------------------------


def test_stale_socket_is_cleaned_up(
    socket_path: str, tmp_path: Path
) -> None:
    # Create a stale file at the socket path (not an actual listening socket).
    Path(socket_path).write_bytes(b"stale")
    transport = LocalTransport(socket_path=socket_path)
    try:
        assert transport.is_host, (
            "transport should unlink stale file and become host"
        )
    finally:
        transport.shutdown(timeout=2.0)


# ---------------------------------------------------------------------------
# Independent-process verification
# ---------------------------------------------------------------------------


HOST_WORKER_SRC = """
import json
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
transport = LocalTransport(socket_path=socket_path)
assert transport.is_host
sink = Sink(out_path=out_path)
transport._bus.handlers[sink.name] = sink
transport._bus.scopes["global"] = {sink.name}
Path(os.environ["READY_PATH"]).write_text("ready")
deadline = time.time() + 20
while time.time() < deadline:
    if sink.count >= 1:
        break
    time.sleep(0.05)
transport.shutdown(timeout=2.0)
print(sink.count)
"""


def test_two_independent_processes_share_host(
    socket_path: str, tmp_path: Path
) -> None:
    out_path = tmp_path / "count.txt"
    ready_path = tmp_path / "ready.txt"
    worker_script = tmp_path / "worker.py"
    worker_script.write_text(HOST_WORKER_SRC)

    env = os.environ.copy()
    env["GOGGLES_SOCKET"] = socket_path
    env["OUT_PATH"] = str(out_path)
    env["READY_PATH"] = str(ready_path)

    proc = subprocess.Popen(
        [sys.executable, str(worker_script)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        assert _wait_until(ready_path.exists, timeout=10.0), (
            "host subprocess failed to initialize"
        )

        client = LocalTransport(socket_path=socket_path)
        try:
            assert not client.is_host
            client.emit(Event(
                kind="log", scope="global", payload="cross-proc",
                filepath="t.py", lineno=1,
            ))
            assert _wait_until(
                lambda: out_path.exists() and out_path.read_text() == "1",
                timeout=5.0,
            ), "host process should have recorded 1 event"
        finally:
            client.shutdown(timeout=2.0)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
