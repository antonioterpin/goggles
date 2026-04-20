"""LocalTransport: Unix-socket based transport for same-machine logging.

This module replaces the previous portal-based RPC with a direct Unix-domain-
socket transport. The first process to bind the configured socket path becomes
the host (owns the `EventBus` and dispatches events to attached handlers).
Subsequent processes on the same machine connect as clients; their events are
serialized with pickle protocol 5 and forwarded to the host.

Payloads above :data:`_DEFAULT_SHM_THRESHOLD` bytes take a zero-copy
shared-memory side-channel: the client writes the numpy buffer into a
``multiprocessing.shared_memory.SharedMemory`` block and sends only metadata
over the socket; the host maps the same block as a view and passes it to
handlers before unlinking it.

Only same-machine multi-process routing is supported. Cross-machine logging is
out of scope; add a new implementation of :class:`Transport` if needed.
"""

from __future__ import annotations

import logging
import os
import pickle
import queue
import socket
import struct
import threading
import time
from multiprocessing import shared_memory
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from goggles.types import Event

if TYPE_CHECKING:
    from goggles import EventBus

_log = logging.getLogger(__name__)


# --- Framing ---------------------------------------------------------------

# Every message on the wire is prefixed with a 1-byte kind and a 4-byte
# big-endian length. The payload format depends on kind.
_MSG_SMALL = 1  # inline pickle protocol 5 with out-of-band buffers
_MSG_LARGE = 2  # shared-memory side-channel; payload is pickled metadata
_MSG_ATTACH = 3
_MSG_DETACH = 4
_MSG_BYE = 5

_HEADER_FMT = "!BI"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)

_DEFAULT_SHM_THRESHOLD = 65536


def _default_socket_path() -> str:
    """Default socket path, overridable via ``GOGGLES_SOCKET``.

    Returns:
        Absolute filesystem path to the Unix domain socket.
    """
    override = os.getenv("GOGGLES_SOCKET")
    if override:
        return override
    runtime_dir = os.getenv("XDG_RUNTIME_DIR") or "/tmp"
    return os.path.join(runtime_dir, f"goggles-{os.getuid()}.sock")


def _default_shm_threshold() -> int:
    """Default shared-memory threshold in bytes.

    Returns:
        Minimum payload size (bytes) that triggers the shm side-channel.
    """
    raw = os.getenv("GOGGLES_SHM_THRESHOLD")
    if raw is None:
        return _DEFAULT_SHM_THRESHOLD
    return max(0, int(raw))


def _recvall(sock: socket.socket, n: int) -> bytes | None:
    """Read exactly ``n`` bytes from ``sock``.

    Args:
        sock: Connected stream socket.
        n: Number of bytes to read.

    Returns:
        The read bytes, or None if the peer closed the connection.
    """
    buf = bytearray()
    while len(buf) < n:
        try:
            chunk = sock.recv(n - len(buf))
        except (ConnectionResetError, OSError):
            return None
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _pack_small(event: Event) -> bytes:
    """Serialize an Event for the small inline path.

    Uses pickle protocol 5 with ``buffer_callback`` to avoid copying numpy
    array bytes into the main pickle stream.

    Args:
        event: Event to serialize.

    Returns:
        A single byte string containing the main pickle + out-of-band
        buffers, framed so :func:`_unpack_small` can reverse the operation.
    """
    buffers: list[pickle.PickleBuffer] = []
    main = pickle.dumps(event, protocol=5, buffer_callback=buffers.append)
    parts: list[bytes] = [
        struct.pack("!I", len(main)),
        main,
        struct.pack("!I", len(buffers)),
    ]
    for buf in buffers:
        mv = buf.raw()
        parts.append(struct.pack("!I", len(mv)))
        parts.append(bytes(mv))
    return b"".join(parts)


def _unpack_small(payload: bytes) -> Event:
    """Reverse :func:`_pack_small`.

    Args:
        payload: Bytes produced by :func:`_pack_small`.

    Returns:
        The deserialized Event.
    """
    offset = 0
    (main_len,) = struct.unpack_from("!I", payload, offset)
    offset += 4
    main = payload[offset : offset + main_len]
    offset += main_len
    (num_bufs,) = struct.unpack_from("!I", payload, offset)
    offset += 4
    buffers: list[bytes] = []
    for _ in range(num_bufs):
        (blen,) = struct.unpack_from("!I", payload, offset)
        offset += 4
        buffers.append(payload[offset : offset + blen])
        offset += blen
    return pickle.loads(main, buffers=buffers)


def _pack_large(event: Event, shm_name: str) -> bytes:
    """Serialize a LARGE-mode event.

    The numpy payload has already been copied into shared memory named
    ``shm_name``; the wire payload carries metadata needed to reconstruct
    the array on the host side.

    Args:
        event: Event whose ``payload`` is a numpy ndarray.
        shm_name: Name of the shared memory block holding the array bytes.

    Returns:
        Pickled metadata + a stripped event (payload replaced by None).
    """
    arr = event.payload
    if not isinstance(arr, np.ndarray):
        raise TypeError(
            "LARGE path requires numpy.ndarray payload, got "
            f"{type(arr).__name__}"
        )
    # Build a stripped event: same fields, payload=None. The host fills it in.
    stripped = Event(
        kind=event.kind,
        scope=event.scope,
        payload=None,
        filepath=event.filepath,
        lineno=event.lineno,
        level=event.level,
        step=event.step,
        time=event.time,
        extra=event.extra,
    )
    meta: dict[str, Any] = {
        "shm_name": shm_name,
        "dtype": str(arr.dtype),
        "shape": tuple(arr.shape),
        "nbytes": int(arr.nbytes),
        "event": stripped,
    }
    return pickle.dumps(meta, protocol=5)


def _unpack_large(payload: bytes) -> Event:
    """Reconstruct an Event from a LARGE frame and the shared memory it names.

    The shared-memory block is opened, its bytes are viewed as a numpy array
    (zero-copy), the view is copied into a private array, and the block is
    closed and unlinked.

    Args:
        payload: Pickled metadata produced by :func:`_pack_large`.

    Returns:
        The reconstructed Event with its numpy payload materialized.
    """
    meta = pickle.loads(payload)
    shm_name: str = meta["shm_name"]
    dtype = np.dtype(meta["dtype"])
    shape: tuple[int, ...] = tuple(meta["shape"])
    stripped: Event = meta["event"]

    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        view = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        # Copy out before unlinking so the Event owns its payload.
        arr = np.array(view, copy=True)
    finally:
        shm.close()
        try:
            shm.unlink()
        except FileNotFoundError:
            pass
    return Event(
        kind=stripped.kind,
        scope=stripped.scope,
        payload=arr,
        filepath=stripped.filepath,
        lineno=stripped.lineno,
        level=stripped.level,
        step=stripped.step,
        time=stripped.time,
        extra=stripped.extra,
    )


# --- Transport Protocol ----------------------------------------------------


@runtime_checkable
class Transport(Protocol):
    """Contract for routing Goggles events to the EventBus."""

    def emit(self, event: Event) -> None:
        """Route an event asynchronously (fire-and-forget).

        Args:
            event: Event to route.
        """
        ...

    def emit_sync(self, event: Event) -> None:
        """Route an event, blocking until it has reached the bus.

        Args:
            event: Event to route.
        """
        ...

    def attach(self, handlers: list[dict], scopes: list[str]) -> None:
        """Attach handlers under the given scopes.

        Args:
            handlers: Serialized handlers to attach.
            scopes: Scopes under which to attach.
        """
        ...

    def detach(self, handler_name: str, scope: str) -> None:
        """Detach a handler from the given scope.

        Args:
            handler_name: Name of the handler to detach.
            scope: Scope to detach from.
        """
        ...

    def shutdown(self, timeout: float | None = None) -> None:
        """Flush pending events and release resources.

        Args:
            timeout: Maximum time to wait for pending events, in seconds.
        """
        ...


# --- LocalTransport --------------------------------------------------------


_SENTINEL = object()


class LocalTransport:
    """Unix-socket based transport with auto-elected host.

    On construction, the transport either:
      1. Connects to an existing host at ``socket_path`` (client mode), or
      2. Binds the socket and becomes host (running an accept loop that
         spawns one reader thread per connected client).

    Host mode also owns an ``EventBus`` instance; its own ``emit`` calls
    dispatch directly to the bus via a background drain thread (no socket,
    no serialization). Client-mode ``emit`` calls pickle the event and send
    it to the host; payloads above ``shm_threshold`` bytes travel through
    shared memory instead of the socket.
    """

    def __init__(
        self,
        socket_path: str | None = None,
        shm_threshold: int | None = None,
    ) -> None:
        """Initialize a LocalTransport.

        Args:
            socket_path: Path of the Unix domain socket.
                Defaults to the value of ``GOGGLES_SOCKET`` or a
                per-user path under ``$XDG_RUNTIME_DIR`` / ``/tmp``.
            shm_threshold: Payload size threshold (bytes) at or above which
                numpy payloads travel through shared memory. Defaults to
                ``GOGGLES_SHM_THRESHOLD`` or 64 KiB. ``0`` disables shm.
        """
        # Lazy import to avoid a cycle with goggles/__init__.py.
        from goggles import EventBus  # noqa: PLC0415

        self._socket_path = socket_path or _default_socket_path()
        self._shm_threshold = (
            shm_threshold
            if shm_threshold is not None
            else _default_shm_threshold()
        )
        self._bus: EventBus = EventBus()
        self._is_host = False
        self._running = True

        # Host-side state
        self._server_sock: socket.socket | None = None
        self._accept_thread: threading.Thread | None = None
        self._reader_threads: list[threading.Thread] = []
        self._drain_queue: queue.SimpleQueue[Any] = queue.SimpleQueue()
        self._drain_thread: threading.Thread | None = None

        # Client-side state
        self._client_sock: socket.socket | None = None
        self._send_queue: queue.SimpleQueue[Any] = queue.SimpleQueue()
        self._send_thread: threading.Thread | None = None
        self._send_lock = threading.Lock()

        self._connect_or_host()

    # ----- introspection ---------------------------------------------------

    @property
    def is_host(self) -> bool:
        """Whether this transport is the host for its socket path."""
        return self._is_host

    # ----- setup -----------------------------------------------------------

    def _connect_or_host(self) -> None:
        """Try to connect to the socket; otherwise bind and become host."""
        # Attempt to join an existing host first.
        if self._try_connect():
            self._start_send_worker()
            return

        # Clean up any stale socket file and bind.
        parent = os.path.dirname(self._socket_path)
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
        try:
            os.unlink(self._socket_path)
        except FileNotFoundError:
            pass

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            server.bind(self._socket_path)
            server.listen(64)
        except OSError:
            # Race: another process bound between our connect and bind.
            server.close()
            if self._try_connect():
                self._start_send_worker()
                return
            raise

        self._server_sock = server
        self._is_host = True
        self._start_host_workers()

    def _try_connect(self, retries: int = 3, backoff: float = 0.01) -> bool:
        """Try to connect to an existing host at ``self._socket_path``.

        Args:
            retries: Number of connect attempts.
            backoff: Initial delay between attempts (doubles each retry).

        Returns:
            True iff a connection was established and stored on the instance.
        """
        for attempt in range(retries):
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                sock.connect(self._socket_path)
                self._client_sock = sock
                return True
            except OSError:
                # Includes ConnectionRefusedError (no listener),
                # FileNotFoundError (no socket file), and ENOTSOCK
                # (stale regular file at the path).
                sock.close()
                if attempt < retries - 1:
                    time.sleep(backoff * (2**attempt))
        return False

    def _start_host_workers(self) -> None:
        """Start the accept thread and the drain thread on the host side."""
        self._drain_thread = threading.Thread(
            target=self._drain_loop,
            daemon=True,
            name="goggles-drain",
        )
        self._drain_thread.start()

        self._accept_thread = threading.Thread(
            target=self._accept_loop,
            daemon=True,
            name="goggles-accept",
        )
        self._accept_thread.start()

    def _start_send_worker(self) -> None:
        """Start the background sender thread on the client side."""
        self._send_thread = threading.Thread(
            target=self._send_loop,
            daemon=True,
            name="goggles-send",
        )
        self._send_thread.start()

    # ----- host loops ------------------------------------------------------

    def _accept_loop(self) -> None:
        """Accept client connections and spawn a reader thread per client."""
        assert self._server_sock is not None
        while self._running:
            try:
                conn, _ = self._server_sock.accept()
            except OSError:
                return
            reader = threading.Thread(
                target=self._reader_loop,
                args=(conn,),
                daemon=True,
                name=f"goggles-reader-{conn.fileno()}",
            )
            reader.start()
            self._reader_threads.append(reader)

    def _reader_loop(self, conn: socket.socket) -> None:
        """Read framed messages from ``conn`` and route them to the bus.

        Args:
            conn: Connected client socket.
        """
        try:
            while self._running:
                header = _recvall(conn, _HEADER_SIZE)
                if header is None:
                    return
                kind, length = struct.unpack(_HEADER_FMT, header)
                body = _recvall(conn, length) if length else b""
                if body is None:
                    return
                self._dispatch_incoming(kind, body)
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def _dispatch_incoming(self, kind: int, body: bytes) -> None:
        """Route an incoming framed message to the appropriate handler.

        Args:
            kind: Message kind from the frame header.
            body: Frame payload bytes.
        """
        if kind == _MSG_SMALL:
            try:
                event = _unpack_small(body)
            except Exception:
                _log.exception("Failed to unpack small frame")
                return
            self._drain_queue.put(event)
        elif kind == _MSG_LARGE:
            try:
                event = _unpack_large(body)
            except Exception:
                _log.exception("Failed to unpack large frame")
                return
            self._drain_queue.put(event)
        elif kind == _MSG_ATTACH:
            try:
                handlers, scopes = pickle.loads(body)
                self._bus.attach(handlers, scopes)
            except Exception:
                _log.exception("Failed to handle ATTACH frame")
        elif kind == _MSG_DETACH:
            try:
                handler_name, scope = pickle.loads(body)
                self._bus.detach(handler_name, scope)
            except Exception:
                _log.exception("Failed to handle DETACH frame")
        elif kind == _MSG_BYE:
            # Client announced disconnect; reader will hit EOF and exit.
            return
        else:
            _log.warning("Unknown frame kind %d", kind)

    def _drain_loop(self) -> None:
        """Drain queued events on the host and call ``EventBus.emit``."""
        while True:
            item = self._drain_queue.get()
            if item is _SENTINEL:
                return
            try:
                self._bus.emit(item)
            except Exception:
                _log.exception("Handler raised while dispatching event")

    # ----- client loop -----------------------------------------------------

    def _send_loop(self) -> None:
        """Drain queued outbound frames and send them over the socket."""
        assert self._client_sock is not None
        while True:
            item = self._send_queue.get()
            if item is _SENTINEL:
                return
            try:
                with self._send_lock:
                    self._client_sock.sendall(item)
            except (BrokenPipeError, ConnectionResetError, OSError):
                _log.warning(
                    "Goggles client socket closed; dropping queued events"
                )
                return

    # ----- framing helpers -------------------------------------------------

    @staticmethod
    def _frame(kind: int, body: bytes) -> bytes:
        """Prepend a fixed-size header to a frame body.

        Args:
            kind: Message kind byte.
            body: Payload bytes.

        Returns:
            Header + body bytes, ready to send.
        """
        return struct.pack(_HEADER_FMT, kind, len(body)) + body

    def _encode_event(self, event: Event) -> bytes:
        """Encode an Event as a single framed wire message.

        Chooses between the small inline path and the shm side-channel
        based on ``self._shm_threshold`` and the payload's byte size.

        Args:
            event: Event to encode.

        Returns:
            Framed bytes ready to send over the socket.
        """
        payload = event.payload
        if (
            self._shm_threshold > 0
            and isinstance(payload, np.ndarray)
            and payload.nbytes >= self._shm_threshold
        ):
            # Allocate shm and copy the array bytes in once.
            shm = shared_memory.SharedMemory(
                create=True, size=max(1, payload.nbytes)
            )
            try:
                view = np.ndarray(
                    payload.shape, dtype=payload.dtype, buffer=shm.buf
                )
                view[...] = payload
                body = _pack_large(event, shm_name=shm.name)
            finally:
                shm.close()
            return self._frame(_MSG_LARGE, body)
        return self._frame(_MSG_SMALL, _pack_small(event))

    # ----- public API ------------------------------------------------------

    def emit(self, event: Event) -> None:
        """Route an event asynchronously.

        Host mode enqueues the event for the drain thread. Client mode
        enqueues a framed wire message for the send thread.

        Args:
            event: Event to route.
        """
        if not self._running:
            return
        if self._is_host:
            self._drain_queue.put(event)
        else:
            try:
                frame = self._encode_event(event)
            except Exception:
                _log.exception("Failed to encode event for transport")
                return
            self._send_queue.put(frame)

    def emit_sync(self, event: Event) -> None:
        """Route an event and block until the bus has seen it.

        In host mode the event is dispatched inline on the caller thread.
        In client mode this currently behaves like :meth:`emit`; there is
        no cross-process ack protocol (tracked as future work).

        Args:
            event: Event to route.
        """
        if self._is_host:
            try:
                self._bus.emit(event)
            except Exception:
                _log.exception("Handler raised in emit_sync")
            return
        self.emit(event)

    def attach(self, handlers: list[dict], scopes: list[str]) -> None:
        """Attach handlers under the given scopes.

        Args:
            handlers: Serialized handler dicts (see ``Handler.to_dict``).
            scopes: Scopes under which to attach.
        """
        if self._is_host:
            self._bus.attach(handlers, scopes)
            return
        body = pickle.dumps((handlers, scopes), protocol=5)
        with self._send_lock:
            assert self._client_sock is not None
            self._client_sock.sendall(self._frame(_MSG_ATTACH, body))

    def detach(self, handler_name: str, scope: str) -> None:
        """Detach a handler from the given scope.

        Args:
            handler_name: Name of the handler to detach.
            scope: Scope to detach from.
        """
        if self._is_host:
            self._bus.detach(handler_name, scope)
            return
        body = pickle.dumps((handler_name, scope), protocol=5)
        with self._send_lock:
            assert self._client_sock is not None
            self._client_sock.sendall(self._frame(_MSG_DETACH, body))

    def shutdown(self, timeout: float | None = None) -> None:
        """Flush pending work and release resources.

        Args:
            timeout: Max seconds to wait per background thread for a clean
                exit. ``None`` means wait indefinitely.
        """
        if not self._running:
            return
        self._running = False

        if self._is_host:
            self._shutdown_host(timeout)
        else:
            self._shutdown_client(timeout)

    def _shutdown_host(self, timeout: float | None) -> None:
        """Host-side shutdown: close listen socket, drain queue, close bus.

        Args:
            timeout: Max seconds to wait for each background thread.
        """
        if self._server_sock is not None:
            try:
                self._server_sock.close()
            except OSError:
                pass
            self._server_sock = None
        try:
            os.unlink(self._socket_path)
        except FileNotFoundError:
            pass

        if self._accept_thread is not None:
            self._accept_thread.join(timeout=timeout)

        for t in self._reader_threads:
            t.join(timeout=timeout)

        self._drain_queue.put(_SENTINEL)
        if self._drain_thread is not None:
            self._drain_thread.join(timeout=timeout)

        try:
            self._bus.shutdown()
        except Exception:
            _log.exception("EventBus.shutdown raised")

    def _shutdown_client(self, timeout: float | None) -> None:
        """Client-side shutdown: announce BYE, flush sends, close socket.

        Args:
            timeout: Max seconds to wait for the send thread.
        """
        try:
            if self._client_sock is not None:
                with self._send_lock:
                    try:
                        self._client_sock.sendall(self._frame(_MSG_BYE, b""))
                    except OSError:
                        pass
        finally:
            self._send_queue.put(_SENTINEL)
            if self._send_thread is not None:
                self._send_thread.join(timeout=timeout)
            if self._client_sock is not None:
                try:
                    self._client_sock.close()
                except OSError:
                    pass
                self._client_sock = None
