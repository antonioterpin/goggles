"""iceoryx2-backed Goggles transport.

Replaces the previous LocalTransport (Unix-socket / Windows-TCP+token
endpoint split, host-elected broker, ATTACH/DETACH wire frames,
hand-rolled pickle proto-5 OOB framing, 64 KiB shared-memory
side-channel) with a thin wrapper over iceoryx2's publish-subscribe
service.

Every process owns:

- one iceoryx2 IPC node and a single ``goggles/events`` service;
- one publisher loaning dynamic-length byte slices in SHM;
- one subscriber + dispatch thread that decodes inbound events and
  forwards them to a local :class:`EventBus`.

``emit`` dispatches synchronously to the local bus *and* publishes on
the service so peer processes also see the event. Each frame carries
a 16-byte publisher tag so each process's own subscriber can drop
self-published messages and avoid double dispatch.
"""

from __future__ import annotations

import ctypes
import logging
import pickle
import threading
import time
import uuid
from typing import Any

import iceoryx2 as iox2
from iceoryx2 import Slice

from goggles.types import Event

_log = logging.getLogger(__name__)

_TAG_BYTES = 16
_LEN_BYTES = 8
_HEADER_BYTES = _TAG_BYTES + _LEN_BYTES
_SERVICE_NAME = "goggles/events"
_INITIAL_CAPACITY = 4096
_RECV_POLL_S = 1e-5


def _open_service(node: Any) -> Any:
    return (
        node.service_builder(iox2.ServiceName.new(_SERVICE_NAME))
        .publish_subscribe(Slice[ctypes.c_uint8])
        .open_or_create()
    )


class LocalTransport:
    """Cross-process transport on top of iceoryx2.

    The class name is preserved for backward compatibility with
    callers that did ``isinstance(t, LocalTransport)`` against the
    legacy implementation; the wire and topology model are different.
    """

    def __init__(self) -> None:
        """Open the iceoryx2 service and start the dispatch thread."""
        from goggles import EventBus  # noqa: PLC0415

        self._tag = uuid.uuid4().bytes  # 16-byte process-local identity
        self._bus: EventBus = EventBus()
        self._iox_node = iox2.NodeBuilder.new().create(iox2.ServiceType.Ipc)
        self._service = _open_service(self._iox_node)
        self._capacity = _INITIAL_CAPACITY
        self._publisher = self._build_publisher(self._capacity)
        self._publisher_lock = threading.Lock()
        self._subscriber = self._service.subscriber_builder().create()

        self._running = True
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop,
            name="goggles-iox-recv",
            daemon=True,
        )
        self._thread.start()

    @property
    def is_running(self) -> bool:
        """Whether the transport is accepting new events."""
        return self._running

    def _build_publisher(self, capacity: int) -> Any:
        return (
            self._service.publisher_builder()
            .initial_max_slice_len(capacity)
            .create()
        )

    def emit(self, event: Event) -> None:
        """Dispatch ``event`` locally and publish it to peer processes.

        Local dispatch happens first and synchronously, so emits made
        immediately before :meth:`shutdown` are still observed by
        local handlers. Cross-process delivery is best-effort: errors
        are logged but never raised.
        """
        if not self._running:
            return
        try:
            self._bus.emit(event)
        except Exception:  # noqa: BLE001
            _log.exception("EventBus.emit raised on local emit")
        try:
            body = pickle.dumps(event, protocol=5)
        except Exception:  # noqa: BLE001
            _log.exception("Failed to pickle event for cross-process emit")
            return
        total = _HEADER_BYTES + len(body)
        try:
            with self._publisher_lock:
                if total > self._capacity:
                    self._capacity = max(total, self._capacity * 2)
                    self._publisher = self._build_publisher(self._capacity)
                sample = self._publisher.loan_slice_uninit(total)
                base = sample.payload_ptr
                ctypes.memmove(base, self._tag, _TAG_BYTES)
                ctypes.memmove(
                    base + _TAG_BYTES,
                    len(body).to_bytes(_LEN_BYTES, "little"),
                    _LEN_BYTES,
                )
                if body:
                    ctypes.memmove(
                        base + _HEADER_BYTES, body, len(body)
                    )
                sample.assume_init().send()
        except Exception:  # noqa: BLE001
            _log.exception("iceoryx2 publish failed")

    def emit_sync(self, event: Event) -> None:
        """Synchronous emit; identical to :meth:`emit` in this transport.

        With iceoryx2 there is no cross-process ack and ``emit``
        already dispatches to the local bus before publishing, so the
        ``_sync`` variant has no extra work to do.
        """
        self.emit(event)

    def attach(self, handlers: list[dict], scopes: list[str]) -> None:
        """Attach handlers under the given scopes (local process only)."""
        self._bus.attach(handlers, scopes)

    def detach(self, handler_name: str, scope: str) -> None:
        """Detach a handler from the given scope (local process only)."""
        self._bus.detach(handler_name, scope)

    def shutdown(self, timeout: float | None = None) -> None:
        """Stop the dispatch thread and shut the local bus down."""
        if not self._running:
            return
        self._running = False
        self._stop.set()
        self._thread.join(timeout=1.0)
        try:
            self._bus.shutdown(timeout=timeout)
        except Exception:  # noqa: BLE001
            _log.exception("EventBus.shutdown raised")
        with self._publisher_lock:
            self._publisher = None
        self._subscriber = None
        self._service = None
        self._iox_node = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            sample = self._subscriber.receive()
            if sample is None:
                time.sleep(_RECV_POLL_S)
                continue
            sl = sample.payload()
            n = sl.number_of_elements
            buf = (ctypes.c_uint8 * n).from_address(sl.data_ptr)
            tag = bytes(buf[:_TAG_BYTES])
            if tag == self._tag:
                # we published this; the local bus already saw it in emit().
                continue
            length = int.from_bytes(
                bytes(buf[_TAG_BYTES:_HEADER_BYTES]), "little"
            )
            try:
                event = pickle.loads(
                    bytes(buf[_HEADER_BYTES : _HEADER_BYTES + length])
                )
            except Exception:  # noqa: BLE001
                _log.exception("Failed to unpickle inbound event")
                continue
            try:
                self._bus.emit(event)
            except Exception:  # noqa: BLE001
                _log.exception("EventBus.emit raised on inbound event")


__all__ = ["LocalTransport"]
