"""Transport workers release delivered payloads and disconnected clients."""

from __future__ import annotations

import gc
import socket
import tempfile
import threading
import time
import uuid
import weakref
from collections.abc import Callable, Iterator
from multiprocessing import shared_memory

import numpy as np
import pytest

from goggles import Event
from goggles._core.transport import LocalTransport, _local


def _wait_until(predicate: Callable[[], bool]) -> bool:
    """Wait briefly for an asynchronous lifetime transition."""
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


@pytest.fixture
def host() -> Iterator[LocalTransport]:
    """Create an isolated host and always close its threads."""
    path = f"{tempfile.gettempdir()}/gg-life-{uuid.uuid4().hex[:12]}.sock"
    transport = LocalTransport(socket_path=path)
    try:
        yield transport
    finally:
        transport.shutdown(timeout=3.0)


def test_idle_drain_releases_delivered_payload(
    host: LocalTransport, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An idle worker must not hold the last delivered image until next emit."""
    delivered = threading.Event()
    monkeypatch.setattr(host._bus, "emit", lambda _: delivered.set())
    payload = np.zeros((1024, 1024), dtype=np.uint8)
    ref = weakref.ref(payload)
    host.emit(Event("image", "global", payload, __file__, 0))
    del payload
    assert delivered.wait(3.0), "The payload must reach the handler"
    assert _wait_until(lambda: ref() is None), (
        "The idle drain worker must release a delivered payload"
    )


def test_disconnected_reader_threads_are_released(host: LocalTransport) -> None:
    """Repeated short-lived clients must not grow the host's thread registry."""
    refs = []
    for _ in range(5):
        client = LocalTransport(socket_path=host._socket_path)
        assert _wait_until(lambda: bool(host._reader_threads)), (
            "The host must register the connected client's reader"
        )
        with host._client_sockets_lock:
            reader = host._reader_threads[-1]
        refs.append(weakref.ref(reader))
        client.shutdown(timeout=3.0)
        reader.join(timeout=3.0)
        assert not reader.is_alive(), "Disconnect must finish the reader"
        del client, reader
    gc.collect()
    assert all(ref() is None for ref in refs), (
        "Disconnected client threads must not remain rooted by the host"
    )


def test_host_queue_limit_preserves_control_operations(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A saturated host drops new events while retaining ordered controls."""
    monkeypatch.setenv("GOGGLES_MAX_PENDING_EVENTS", "2")
    path = f"{tempfile.gettempdir()}/gg-limit-{uuid.uuid4().hex[:12]}.sock"
    transport = LocalTransport(socket_path=path)
    entered, release = threading.Event(), threading.Event()
    seen = []

    def handle(event: Event) -> None:
        entered.set()
        assert release.wait(3.0), "Test must release the handler"
        seen.append(event.step)

    monkeypatch.setattr(transport._bus, "emit", handle)
    monkeypatch.setattr(
        transport._bus, "detach", lambda *args: seen.append("detach")
    )
    try:
        transport.emit(Event("metric", "global", 0, __file__, 0, step=0))
        assert entered.wait(3.0), "First event must be in flight"
        for step in range(1, 11):
            transport.emit(Event("metric", "global", 0, __file__, 0, step=step))
        transport.detach("handler", "global")
        release.set()
        transport.shutdown(timeout=3.0)
        assert seen == [0, 1, 2, "detach"], (
            "Keep queued events and ordered detach; drop only excess events"
        )
        diagnostics = capsys.readouterr().err
        assert (
            "Goggles host queue full (limit=2); dropped 8 events" in diagnostics
        ), "Drops must be visible without configuring Python logging"
        assert "shutdown: dropped 8 events" in diagnostics, (
            "Shutdown must report the final drop count"
        )
    finally:
        release.set()
        transport.shutdown(timeout=3.0)


@pytest.mark.parametrize("shm", [False, True])
def test_client_queue_limit_releases_rejected_shared_memory(
    host: LocalTransport, monkeypatch: pytest.MonkeyPatch, shm: bool
) -> None:
    """Client overload bounds frames and releases rejected SHM segments."""
    monkeypatch.setenv("GOGGLES_MAX_PENDING_EVENTS", "2")
    with monkeypatch.context() as pause:
        pause.setattr(LocalTransport, "_start_send_worker", lambda _: None)
        client = LocalTransport(
            socket_path=host._socket_path, shm_threshold=1 if shm else 0
        )
    names = []
    original = client._encode_event

    def encode(event: Event):
        item = original(event)
        if item.shm_name is not None:
            names.append(item.shm_name)
        return item

    monkeypatch.setattr(client, "_encode_event", encode)
    seen = []
    monkeypatch.setattr(
        host._bus, "emit", lambda event: seen.append(event.step)
    )
    try:
        for step in range(10):
            client.emit(
                Event(
                    "image",
                    "global",
                    np.zeros((32, 32)),
                    __file__,
                    0,
                    step=step,
                )
            )
        for name in names[2:]:
            with pytest.raises(FileNotFoundError):
                shared_memory.SharedMemory(name=name)
        client._start_send_worker()
        client.shutdown(timeout=3.0)
        assert _wait_until(lambda: len(seen) >= 2), (
            "Accepted events must arrive"
        )
        host.shutdown(timeout=3.0)
        assert seen == [0, 1], (
            "Only the two accepted events should be delivered"
        )
    finally:
        client.shutdown(timeout=3.0)


@pytest.mark.parametrize("value", ["-1", "not-an-integer"])
def test_invalid_queue_limit_fails_before_connecting(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """A mistyped queue limit must not silently leave queues unbounded."""
    monkeypatch.setenv("GOGGLES_MAX_PENDING_EVENTS", value)
    with pytest.raises(ValueError, match="GOGGLES_MAX_PENDING_EVENTS"):
        LocalTransport()


class _TrackedFrame(bytearray):
    """Wire buffer with weak-reference support for worker lifetime tests."""


@pytest.mark.parametrize("worker", ["sender", "reader"])
def test_idle_socket_worker_releases_last_frame(
    host: LocalTransport, monkeypatch: pytest.MonkeyPatch, worker: str
) -> None:
    """An open, idle connection must release its last transmitted frame."""
    refs: list[weakref.ReferenceType[_TrackedFrame]] = []
    delivered = threading.Event()
    monkeypatch.setattr(host._bus, "emit", lambda _: delivered.set())
    client = LocalTransport(socket_path=host._socket_path, shm_threshold=0)
    if worker == "sender":
        encode = client._encode_event

        def tracked_encode(event: Event) -> _local._SendItem:
            item = encode(event)
            item.frame = _TrackedFrame(item.frame)
            refs.append(weakref.ref(item.frame))
            return item

        monkeypatch.setattr(client, "_encode_event", tracked_encode)
    else:
        receive = _local._recvall

        def tracked_receive(
            conn: socket.socket, size: int
        ) -> bytes | _TrackedFrame | None:
            frame = receive(conn, size)
            if frame is not None and size > _local._HEADER_SIZE:
                tracked = _TrackedFrame(frame)
                refs.append(weakref.ref(tracked))
                return tracked
            return frame

        monkeypatch.setattr(_local, "_recvall", tracked_receive)
    try:
        client.emit(Event("image", "global", np.zeros((128, 128)), __file__, 0))
        assert delivered.wait(3.0), "The frame must reach the handler"
        assert _wait_until(lambda: bool(refs) and refs[0]() is None), (
            "Release the idle worker frame before another event arrives"
        )
    finally:
        client.shutdown(timeout=3.0)


@pytest.mark.parametrize("shm", [False, True])
@pytest.mark.parametrize("synchronous", [False, True])
@pytest.mark.parametrize("allocate_before_shutdown", [False, True])
def test_shutdown_rejects_an_inflight_encoded_frame(
    host: LocalTransport,
    monkeypatch: pytest.MonkeyPatch,
    shm: bool,
    synchronous: bool,
    allocate_before_shutdown: bool,
) -> None:
    """Finishing encoding after shutdown must not orphan a queued frame."""
    client = LocalTransport(
        socket_path=host._socket_path, shm_threshold=1 if shm else 0
    )
    encoding, release = threading.Event(), threading.Event()
    encode = client._encode_event
    names = []

    def delayed_encode(event: Event) -> _local._SendItem:
        if not allocate_before_shutdown:
            encoding.set()
            assert release.wait(3.0), "Test must release the producer"
        item = encode(event)
        if item.shm_name is not None:
            names.append(item.shm_name)
        if allocate_before_shutdown:
            encoding.set()
            assert release.wait(3.0), "Test must release the producer"
        return item

    monkeypatch.setattr(client, "_encode_event", delayed_encode)
    emit = client.emit_sync if synchronous else client.emit
    producer = threading.Thread(
        target=emit,
        args=(Event("image", "global", np.zeros((128, 128)), __file__, 0),),
    )
    try:
        producer.start()
        assert encoding.wait(3.0), (
            "Producer must enter encoding before shutdown"
        )
        client.shutdown(timeout=3.0)
        release.set()
        producer.join(timeout=3.0)
        assert not producer.is_alive(), "Do not wait on a stopped sender"
        assert client._send_queue.empty(), (
            "Shutdown must not leave frames behind its consumed sentinel"
        )
        assert not client._pending_shm, "Release all encoded shared memory"
        for name in names:
            with pytest.raises(FileNotFoundError):
                shared_memory.SharedMemory(name=name)
    finally:
        release.set()
        producer.join(timeout=6.0)
        client.shutdown(timeout=3.0)


def test_closed_host_rejects_control_operations(host: LocalTransport) -> None:
    """Controls sent after shutdown must not accumulate without a drainer."""
    host.shutdown(timeout=3.0)
    host.attach([], ["global"])
    host.detach("missing", "global")
    assert host._drain_queue.empty(), "A stopped host cannot consume controls"
