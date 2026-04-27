"""Tests for the LocalTransport (cross-platform same-machine transport)."""

from __future__ import annotations

import contextlib
import os
import struct
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from multiprocessing import shared_memory
from pathlib import Path
from typing import ClassVar, cast

import numpy as np
import pytest

import goggles as gg
from goggles import Event, Kind
from goggles._core.transport import (
    _DEFAULT_SHM_THRESHOLD,
    _HEADER_FMT,
    _HEADER_SIZE,
    _IS_WINDOWS,
    _MSG_SMALL,
    _SHM_NAME_PREFIX,
    LocalTransport,
    _default_shm_threshold,
    _default_socket_path,
    _next_shm_name,
    _pack_large,
    _pack_small_frame,
    _reap_orphan_shm,
    _try_unlink_shm,
    _unpack_large,
    _unpack_small,
    _user_tag,
)


def _frame_body(event: Event) -> bytes:
    """Return only the body half of a SMALL wire frame for unit tests.

    Args:
        event: Event to pack into a SMALL frame.

    Returns:
        The frame body (everything after the 5-byte header).
    """
    return bytes(_pack_small_frame(event)[_HEADER_SIZE:])


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


def test_default_shm_threshold_env_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GOGGLES_SHM_THRESHOLD should support overrides and safe fallback.

    Args:
        monkeypatch: pytest helper for environment isolation.
    """
    monkeypatch.delenv("GOGGLES_SHM_THRESHOLD", raising=False)
    assert _default_shm_threshold() == _DEFAULT_SHM_THRESHOLD

    monkeypatch.setenv("GOGGLES_SHM_THRESHOLD", "1234")
    assert _default_shm_threshold() == 1234

    monkeypatch.setenv("GOGGLES_SHM_THRESHOLD", "-1")
    assert _default_shm_threshold() == 0

    monkeypatch.setenv("GOGGLES_SHM_THRESHOLD", "not-an-int")
    assert _default_shm_threshold() == _DEFAULT_SHM_THRESHOLD


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
    restored = _unpack_small(_frame_body(event))
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
    restored = _unpack_small(_frame_body(event))
    assert isinstance(restored.payload, np.ndarray)
    assert restored.payload.shape == (16, 16)
    np.testing.assert_array_equal(restored.payload, arr)


def test_pack_small_frame_header_layout() -> None:
    """Frame layout must be ``[1-byte kind][4-byte body_len][body]``
    so the host's reader (which pulls the header with ``_recvall``)
    can decode without changes."""
    arr = np.arange(8 * 8, dtype=np.uint8).reshape(8, 8)
    event = Event(
        kind="image",
        scope="global",
        payload=arr,
        filepath="test.py",
        lineno=3,
    )
    frame = _pack_small_frame(event)
    assert isinstance(frame, bytearray)
    kind, body_len = struct.unpack_from(_HEADER_FMT, frame, 0)
    assert kind == _MSG_SMALL
    assert body_len == len(frame) - _HEADER_SIZE


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


@pytest.mark.parametrize(
    "kind, payload, extra",
    [
        ("log", "", None),
        (
            "artifact",
            {"empty": "", "non_empty": "x"},
            {"name": "d", "format": "json"},
        ),
    ],
    ids=["log_empty_string", "artifact_with_empty_string_value"],
)
def test_empty_string_payloads_roundtrip_through_framing(
    kind: str, payload: object, extra: dict | None
) -> None:
    """Regression for #79: empty-string payloads must survive the wire layer.

    Portal's non-empty-buffer assertion used to blow up on ``""``. The new
    transport goes through ``_pack_small_frame`` → socket → ``_unpack_small``,
    so we assert the framing round-trip directly (covers the failure mode
    regardless of host/client dispatch mode).

    Args:
        kind: Event kind to round-trip.
        payload: Event payload (string or mapping containing empty strings).
        extra: Optional event ``extra`` dict; when provided, ``step=0`` is
            also set.
    """
    event_kwargs: dict = {
        "kind": kind,
        "scope": "global",
        "payload": payload,
        "filepath": "t.py",
        "lineno": 1,
    }
    if extra is not None:
        event_kwargs["step"] = 0
        event_kwargs["extra"] = extra
    event = Event(**event_kwargs)
    restored = _unpack_small(_frame_body(event))
    assert restored.kind == kind
    assert restored.payload == payload
    if extra is not None:
        assert restored.extra == extra


@pytest.mark.parametrize(
    "kind, payload, extra",
    [
        ("log", "", None),
        (
            "artifact",
            {"empty": "", "non_empty": "x"},
            {"name": "d", "format": "json"},
        ),
    ],
    ids=["log_empty_string", "artifact_with_empty_string_value"],
)
def test_empty_string_payloads_survive_host_emit_sync(
    socket_path: str, kind: str, payload: object, extra: dict | None
) -> None:
    """Integration-level smoke for #79: host-mode emit_sync preserves data.

    This exercises dispatch (not framing), to confirm the inline path
    doesn't eat empty strings. The wire-level regression lives in
    :func:`test_empty_string_payloads_roundtrip_through_framing`.

    Args:
        socket_path: Endpoint path (via fixture).
        kind: Event kind to dispatch.
        payload: Event payload (string or mapping containing empty strings).
        extra: Optional event ``extra`` dict; when provided, ``step=0`` is
            also set.
    """
    transport = LocalTransport(socket_path=socket_path)
    try:
        collector = _CollectingHandler()
        _install_collector(transport, collector)
        event_kwargs: dict = {
            "kind": kind,
            "scope": "global",
            "payload": payload,
            "filepath": "t.py",
            "lineno": 1,
        }
        if extra is not None:
            event_kwargs["step"] = 0
            event_kwargs["extra"] = extra
        transport.emit_sync(Event(**event_kwargs))
        assert len(collector.events) == 1
        assert collector.events[0].payload == payload
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


@pytest.mark.parametrize(
    ("shm_threshold", "path_label"),
    [
        (10**9, "SMALL inline pickle"),
        (1024, "LARGE shared memory"),
    ],
)
def test_client_numpy_payload_is_snapshotted_before_mutation(
    socket_path: str,
    shm_threshold: int,
    path_label: str,
) -> None:
    """Client emits must snapshot ndarray bytes before returning.

    The benchmark hot path reuses payload buffers across calls; correctness
    depends on the transport copying the array into its wire representation
    before user code can mutate the original buffer.

    Args:
        socket_path: Endpoint path (via fixture).
        shm_threshold: Threshold that selects SMALL or LARGE encoding.
        path_label: Human-readable encoding path for assertion messages.
    """
    host = LocalTransport(socket_path=socket_path, shm_threshold=shm_threshold)
    try:
        client = LocalTransport(
            socket_path=socket_path,
            shm_threshold=shm_threshold,
        )
        try:
            collector = _CollectingHandler()
            _install_collector(host, collector)

            arr = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
            expected = arr.copy()
            client.emit(
                Event(
                    kind="image",
                    scope="global",
                    payload=arr,
                    filepath="t.py",
                    lineno=1,
                )
            )
            arr.fill(-1.0)

            assert _wait_until(lambda: len(collector.events) == 1), (
                f"{path_label} client event should be delivered"
            )
            got = collector.events[0].payload
            assert isinstance(got, np.ndarray)
            np.testing.assert_array_equal(got, expected)
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


def test_shm_threshold_zero_disables_side_channel(
    socket_path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A threshold of 0 should force ndarray payloads onto SMALL frames.

    Args:
        socket_path: Endpoint path (via fixture).
        monkeypatch: pytest helper used to fail fast if shm allocation starts.
    """

    def fail_next_shm_name() -> str:
        raise AssertionError("threshold=0 must not allocate shared memory")

    monkeypatch.setattr(
        "goggles._core.transport._next_shm_name",
        fail_next_shm_name,
    )
    host = LocalTransport(socket_path=socket_path, shm_threshold=0)
    try:
        client = LocalTransport(socket_path=socket_path, shm_threshold=0)
        try:
            collector = _CollectingHandler()
            _install_collector(host, collector)

            arr = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
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


def test_client_attach_and_detach_control_frames(socket_path: str) -> None:
    """Client-mode ATTACH/DETACH frames must update the host EventBus.

    Args:
        socket_path: Endpoint path (via fixture).
    """
    gg.register_handler(_CollectingHandler)
    host = LocalTransport(socket_path=socket_path)
    try:
        client = LocalTransport(socket_path=socket_path)
        try:
            assert not client.is_host
            bus = host._bus  # type: ignore[attr-defined]
            client.attach(
                handlers=[_CollectingHandler().to_dict()],
                scopes=["global"],
            )
            assert _wait_until(
                lambda: (
                    "collector" in bus.handlers
                    and "collector" in bus.scopes.get("global", set())
                )
            ), "client ATTACH should install handler on host"
            collector = cast(_CollectingHandler, bus.handlers["collector"])

            client.emit(
                Event(
                    kind="log",
                    scope="global",
                    payload="before-detach",
                    filepath="t.py",
                    lineno=1,
                )
            )
            assert _wait_until(lambda: len(collector.events) == 1)
            assert collector.events[0].payload == "before-detach"

            client.detach("collector", "global")
            assert _wait_until(lambda: "collector" not in bus.handlers), (
                "client DETACH should remove handler from host"
            )

            client.emit(
                Event(
                    kind="log",
                    scope="global",
                    payload="after-detach",
                    filepath="t.py",
                    lineno=2,
                )
            )
            assert not _wait_until(
                lambda: len(collector.events) > 1,
                timeout=0.2,
            )
        finally:
            client.shutdown(timeout=2.0)
    finally:
        host.shutdown(timeout=2.0)


def test_shutdown_flushes_pending_events(socket_path: str) -> None:
    """Every enqueued event must reach the host before shutdown returns.

    Regresses the "77/1500 video logs arrived" bug: BYE was sent out-of-band
    during shutdown, so any frames still in the client's send queue were
    discarded when the send thread was force-closed. Full retrospective:
    ``docs/retrospectives/2026-04-shutdown-bye-flush.md``.

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
deadline_s = float(os.environ.get("DEADLINE", "30"))
grace_s = float(os.environ.get("GRACE", "0.5"))
transport = LocalTransport(socket_path=socket_path)
assert transport.is_host
sink = Sink(out_path=out_path)
with transport._bus._lock:
    transport._bus.handlers[sink.name] = sink
    transport._bus.scopes["global"] = {sink.name}
Path(os.environ["READY_PATH"]).write_text("ready")
deadline = time.time() + deadline_s
while time.time() < deadline:
    if sink.count >= expected:
        # Stay up for a short grace period so any client still in the
        # middle of its own shutdown finishes draining before we close
        # the server socket.
        time.sleep(grace_s)
        break
    time.sleep(0.05)
transport.shutdown(timeout=10.0)
print(sink.count)
"""


CLIENT_WORKER_SRC = """
import os
import sys

from goggles import Event
from goggles._core.transport import LocalTransport


socket_path = os.environ["GOGGLES_SOCKET"]
tag = os.environ["CLIENT_TAG"]
n = int(os.environ["N_EVENTS"])

client = LocalTransport(socket_path=socket_path)
try:
    assert not client.is_host, "client subprocess must not become host"
    for i in range(n):
        client.emit(Event(
            kind="log", scope="global", payload=f"{tag}-{i}",
            filepath="client.py", lineno=i,
        ))
finally:
    client.shutdown(timeout=30.0)
"""


def _launch_host_subprocess(
    socket_path: str,
    tmp_path: Path,
    expected: int,
    deadline: float = 30.0,
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
    env["DEADLINE"] = str(deadline)

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


def _terminate_subprocess(
    proc: subprocess.Popen, *, timeout: float = 5.0
) -> None:
    """SIGTERM, then SIGKILL on timeout. Always wait for the child to exit.

    Tolerates a process that has already exited (e.g. the host worker
    naturally finished after recording its expected events) — without
    that guard, ``proc.terminate()`` would raise ``ProcessLookupError``
    on POSIX and fail tests during teardown. Also closes the stdout/
    stderr pipes opened by ``_launch_host_subprocess`` so we don't leak
    file descriptors.

    Args:
        proc: Process to terminate.
        timeout: Seconds to wait for graceful exit before killing.
    """
    try:
        if proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            proc.wait()
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.stderr is not None:
            proc.stderr.close()


@contextlib.contextmanager
def _host_subprocess(
    socket_path: str,
    tmp_path: Path,
    *,
    expected: int,
    deadline: float = 30.0,
    teardown_timeout: float = 5.0,
) -> Iterator[tuple[subprocess.Popen, Path]]:
    """Run a host worker for the duration of the ``with`` block.

    Yields the launched ``Popen`` and the count file the worker writes
    to. Always terminates (and kills, if needed) the subprocess on
    exit so a hung host doesn't survive the test.

    Args:
        socket_path: Endpoint path forwarded as ``GOGGLES_SOCKET``.
        tmp_path: pytest's per-test temp dir; the worker script and
            count/ready files are placed here.
        expected: Number of events the host should record before
            considering itself done.
        deadline: Seconds the host worker waits for ``expected``
            events before exiting on its own.
        teardown_timeout: Seconds to wait for graceful termination on
            block exit.

    Yields:
        tuple[subprocess.Popen, Path]: ``(proc, out_path)``; ``out_path``
            is the file the host worker increments as events arrive.
    """
    proc, out_path = _launch_host_subprocess(
        socket_path, tmp_path, expected=expected, deadline=deadline
    )
    try:
        yield proc, out_path
    finally:
        _terminate_subprocess(proc, timeout=teardown_timeout)


def test_two_independent_processes_share_host(
    socket_path: str, tmp_path: Path
) -> None:
    """Client in this process emits; host subprocess records exactly 1.

    Args:
        socket_path: Endpoint path (via fixture).
        tmp_path: pytest's per-test temporary directory.
    """
    with _host_subprocess(socket_path, tmp_path, expected=1) as (_, out_path):
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
    with _host_subprocess(socket_path, tmp_path, expected=n) as (_, out_path):
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
    with _host_subprocess(socket_path, tmp_path, expected=total) as (
        _,
        out_path,
    ):
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


@pytest.mark.resilience
def test_cross_process_twenty_clients(socket_path: str, tmp_path: Path) -> None:
    """One host subprocess + 19 concurrent client subprocesses.

    Each client is an independent OS process that opens its own
    ``LocalTransport`` in client mode, emits a batch of events, then
    runs a graceful shutdown. Total = 19 * 50 = 950 events. We bring
    up a host subprocess (count=950) first so the in-test process is
    not competing to become host.

    This stresses:
      * accept() and reader-thread-per-client fan-in (20 concurrent
        reader threads on the host),
      * the SHM-less happy path end-to-end across many short-lived
        clients,
      * graceful shutdown ordering: every client's BYE is ordered
        after its queued frames, so no frame is lost.

    Slow-marked because spawning 20 Python interpreters is ~3-5 s.

    Args:
        socket_path: Endpoint path (via fixture).
        tmp_path: pytest's per-test temporary directory.
    """
    n_clients = 19
    n_per_client = 50
    total = n_clients * n_per_client

    client_script = tmp_path / "client_worker.py"
    client_script.write_text(CLIENT_WORKER_SRC)

    with _host_subprocess(
        socket_path,
        tmp_path,
        expected=total,
        deadline=120.0,
        teardown_timeout=10.0,
    ) as (_, out_path):
        clients: list[subprocess.Popen] = []
        for i in range(n_clients):
            env = os.environ.copy()
            env["GOGGLES_SOCKET"] = socket_path
            env["CLIENT_TAG"] = f"client{i:02d}"
            env["N_EVENTS"] = str(n_per_client)
            clients.append(
                subprocess.Popen(
                    [sys.executable, str(client_script)],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            )

        try:
            for c in clients:
                try:
                    c.wait(timeout=90.0)
                except subprocess.TimeoutExpired:
                    c.kill()
                    c.wait()
                    err = c.stderr.read().decode() if c.stderr else ""
                    pytest.fail(
                        f"client did not exit within 90 s; stderr: {err}"
                    )
                if c.returncode != 0:
                    stderr = c.stderr.read().decode() if c.stderr else ""
                    pytest.fail(
                        f"client exited with code {c.returncode}: {stderr}"
                    )
        finally:
            for c in clients:
                if c.stdout is not None:
                    c.stdout.close()
                if c.stderr is not None:
                    c.stderr.close()

        assert _wait_until(
            lambda: out_path.exists() and out_path.read_text() == str(total),
            timeout=30.0,
        ), (
            f"host recorded "
            f"{out_path.read_text() if out_path.exists() else '<no file>'}/"
            f"{total}"
        )


# ---------------------------------------------------------------------------
# Long-form soak (resilience-marked, off the default test path)
# ---------------------------------------------------------------------------


PACED_CLIENT_WORKER_SRC = """
import os
import sys
import time

from goggles import Event
from goggles._core.transport import LocalTransport


socket_path = os.environ["GOGGLES_SOCKET"]
tag = os.environ["CLIENT_TAG"]
n = int(os.environ["N_EVENTS"])
duration = float(os.environ["DURATION_S"])
period = duration / max(1, n)

client = LocalTransport(socket_path=socket_path)
try:
    assert not client.is_host, "client subprocess must not become host"
    next_tick = time.monotonic()
    for i in range(n):
        client.emit(Event(
            kind="log", scope="global", payload=f"{tag}-{i}",
            filepath="client.py", lineno=i,
        ))
        next_tick += period
        sleep_for = next_tick - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)
finally:
    client.shutdown(timeout=60.0)
"""


@pytest.mark.resilience
@pytest.mark.skipif(
    not os.environ.get("GOGGLES_SOAK"),
    reason="opt-in via GOGGLES_SOAK=1; this test runs for ~60 s",
)
def test_cross_process_sustained_soak(socket_path: str, tmp_path: Path) -> None:
    """Sustained multi-minute load across host + 4 paced clients.

    Each client emits at ~250 Hz for 60 s (~15 000 events / client,
    60 000 total). Catches slow leaks and steady-state regressions
    that the bulk-send tests miss because they finish in seconds.
    Opt-in via ``GOGGLES_SOAK=1`` so it doesn't slow the default
    ``pytest`` run.

    Args:
        socket_path: Endpoint path (via fixture).
        tmp_path: pytest's per-test temporary directory.
    """
    duration_s = 60.0
    rate_hz = 250
    n_clients = 4
    n_per_client = int(duration_s * rate_hz)
    total = n_clients * n_per_client
    host_deadline = duration_s + 60.0

    client_script = tmp_path / "paced_client.py"
    client_script.write_text(PACED_CLIENT_WORKER_SRC)

    with _host_subprocess(
        socket_path,
        tmp_path,
        expected=total,
        deadline=host_deadline,
        teardown_timeout=15.0,
    ) as (_, out_path):
        clients: list[subprocess.Popen] = []
        for i in range(n_clients):
            env = os.environ.copy()
            env["GOGGLES_SOCKET"] = socket_path
            env["CLIENT_TAG"] = f"paced{i:02d}"
            env["N_EVENTS"] = str(n_per_client)
            env["DURATION_S"] = str(duration_s)
            clients.append(
                subprocess.Popen(
                    [sys.executable, str(client_script)],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            )

        try:
            for c in clients:
                try:
                    c.wait(timeout=duration_s + 60.0)
                except subprocess.TimeoutExpired:
                    c.kill()
                    c.wait()
                    err = c.stderr.read().decode() if c.stderr else ""
                    pytest.fail(
                        f"paced client did not exit in time; stderr: {err}"
                    )
                if c.returncode != 0:
                    stderr = c.stderr.read().decode() if c.stderr else ""
                    pytest.fail(
                        f"client exited with code {c.returncode}: {stderr}"
                    )
        finally:
            for c in clients:
                if c.stdout is not None:
                    c.stdout.close()
                if c.stderr is not None:
                    c.stderr.close()

        assert _wait_until(
            lambda: out_path.exists() and out_path.read_text() == str(total),
            timeout=60.0,
        ), (
            f"host recorded "
            f"{out_path.read_text() if out_path.exists() else '<no file>'}/"
            f"{total}"
        )


# ---------------------------------------------------------------------------
# Shared-memory helpers
# ---------------------------------------------------------------------------


def test_try_unlink_shm_tolerates_missing_name() -> None:
    # Should be a no-op, not raise.
    _try_unlink_shm("this-name-does-not-exist-12345")


def test_next_shm_name_is_unique_and_prefixed() -> None:
    a = _next_shm_name()
    b = _next_shm_name()
    assert a.startswith(_SHM_NAME_PREFIX)
    assert b.startswith(_SHM_NAME_PREFIX)
    assert a != b


def test_pack_unpack_large_roundtrip_and_unlinks_shm() -> None:
    """LARGE frame metadata should preserve Event fields and reap shm."""
    arr = np.arange(10 * 10, dtype=np.float32).reshape(10, 10)
    event = Event(
        kind="image",
        scope="global",
        payload=arr,
        filepath="test.py",
        lineno=9,
        step=3,
        time=12.5,
        extra={"name": "image"},
    )
    shm_name = _next_shm_name()
    shm = shared_memory.SharedMemory(
        create=True,
        name=shm_name,
        size=arr.nbytes,
    )
    try:
        view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        view[...] = arr

        restored = _unpack_large(_pack_large(event, shm_name))

        assert restored.kind == event.kind
        assert restored.scope == event.scope
        assert restored.filepath == event.filepath
        assert restored.lineno == event.lineno
        assert restored.step == event.step
        assert restored.time == event.time
        assert restored.extra == event.extra
        assert isinstance(restored.payload, np.ndarray)
        np.testing.assert_array_equal(restored.payload, arr)
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=shm_name)
    finally:
        try:
            shm.close()
        except OSError:
            pass
        _try_unlink_shm(shm_name)


@pytest.mark.skipif(
    not Path("/dev/shm").is_dir(),
    reason="Linux-only: /dev/shm is the visible POSIX-shm mount",
)
def test_reap_orphan_shm_removes_only_old_goggles_segments(
    tmp_path: Path,
) -> None:
    del tmp_path  # unused; pytest fixture kept for parity with neighbours
    fresh = shared_memory.SharedMemory(
        create=True, name=f"goggles_{os.getpid()}_freshxxx", size=8
    )
    stale = shared_memory.SharedMemory(
        create=True, name=f"goggles_{os.getpid()}_stalexxx", size=8
    )
    other = shared_memory.SharedMemory(
        create=True, name=f"not_goggles_{os.getpid()}_xxx", size=8
    )
    try:
        # Backdate the stale one well past the cutoff.
        stale_path = Path("/dev/shm") / stale.name
        old = time.time() - 10_000
        os.utime(stale_path, (old, old))

        reaped = _reap_orphan_shm(max_age_s=600.0)
        assert reaped >= 1
        assert not stale_path.exists(), "stale segment must be unlinked"
        assert (Path("/dev/shm") / fresh.name).exists(), (
            "fresh segment must not be touched"
        )
        assert (Path("/dev/shm") / other.name).exists(), (
            "non-goggles segment must not be touched"
        )
    finally:
        for shm in (fresh, other):
            try:
                shm.close()
                shm.unlink()
            except (FileNotFoundError, OSError):
                pass
        try:
            stale.close()
        except OSError:
            pass
