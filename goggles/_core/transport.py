"""LocalTransport: cross-platform local-machine transport for logging.

The first process to bind the configured socket path becomes the host (owns
the :class:`EventBus` and dispatches events to attached handlers); subsequent
processes on the same machine connect as clients.

On Unix (Linux, macOS) the transport uses ``AF_UNIX`` streams with the
socket file protected at ``0o600``. On Windows, ``AF_UNIX`` is unreliable
across Python versions, so the transport binds a TCP loopback socket on
``127.0.0.1`` and writes the chosen port to a sidecar discovery file at the
same logical ``socket_path``. Both paths share the same framing protocol.

Payloads above :data:`_DEFAULT_SHM_THRESHOLD` bytes take a zero-copy
shared-memory side-channel: the client writes the numpy buffer into a
``multiprocessing.shared_memory.SharedMemory`` block and sends only metadata
over the socket; the host maps the same block, copies the view out, and
unlinks the block. The client tracks outstanding block names and unlinks
any that never made it to the host (e.g. because shutdown dropped them).
"""

from __future__ import annotations

import errno
import logging
import os
import pickle
import queue
import secrets
import socket
import struct
import sys
import tempfile
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

_IS_WINDOWS = sys.platform == "win32"


def _user_tag() -> str:
    """Portable per-user identifier for default socket-path naming.

    Returns:
        A short string unique per local user.
    """
    getuid = getattr(os, "getuid", None)
    if getuid is not None:
        return str(getuid())
    return os.environ.get("USERNAME") or os.environ.get("USER") or "default"


def _default_socket_path() -> str:
    """Default socket path, overridable via ``GOGGLES_SOCKET``.

    On Unix this is the Unix-domain-socket path. On Windows it is the
    sidecar discovery file that records the TCP port the host is listening
    on; the file itself is not a socket.

    Returns:
        Absolute filesystem path.
    """
    override = os.getenv("GOGGLES_SOCKET")
    if override:
        return override
    runtime_dir = os.getenv("XDG_RUNTIME_DIR") or tempfile.gettempdir()
    return os.path.join(runtime_dir, f"goggles-{_user_tag()}.sock")


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
    array bytes into the main pickle stream. The out-of-band buffers are
    concatenated into the returned ``bytes`` with a single intermediate
    ``bytearray`` (one memcpy per buffer rather than two).

    Args:
        event: Event to serialize.

    Returns:
        A single byte string containing the main pickle + out-of-band
        buffers, framed so :func:`_unpack_small` can reverse the operation.
    """
    buffers: list[pickle.PickleBuffer] = []
    main = pickle.dumps(event, protocol=5, buffer_callback=buffers.append)
    out = bytearray()
    out += struct.pack("!I", len(main))
    out += main
    out += struct.pack("!I", len(buffers))
    for buf in buffers:
        mv = buf.raw()
        out += struct.pack("!I", len(mv))
        out += mv  # bytearray += memoryview is a single memcpy
    return bytes(out)


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

    Raises:
        TypeError: If ``event.payload`` is not a ``numpy.ndarray``.
    """
    arr = event.payload
    if not isinstance(arr, np.ndarray):
        raise TypeError(
            "LARGE path requires numpy.ndarray payload, got "
            f"{type(arr).__name__}"
        )
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
    """Reconstruct an Event from a LARGE frame and unlink the named shm.

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
        arr = np.array(view, copy=True)
    finally:
        shm.close()
        _try_unlink_shm(shm_name)
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


_SHM_NAME_PREFIX = "goggles_"
_SHM_REAP_AGE_S = 300.0
_LINUX_SHM_DIR = "/dev/shm"


def _next_shm_name() -> str:
    """Return a uniquely-prefixed shared-memory name for this process.

    The ``goggles_`` prefix lets the host opportunistically reap blocks
    that survived a crash without the consumer ever processing them.

    Returns:
        A name suitable for ``SharedMemory(create=True, name=...)``.
    """
    return f"{_SHM_NAME_PREFIX}{os.getpid()}_{secrets.token_hex(8)}"


def _reap_orphan_shm(max_age_s: float = _SHM_REAP_AGE_S) -> int:
    """Best-effort sweep of stale ``goggles_*`` shared-memory blocks.

    The :mod:`multiprocessing.shared_memory` resource tracker covers
    in-process leaks, but a host that crashes between receiving a
    LARGE frame and unlinking the named segment leaves the block
    behind. This sweep runs at host startup and unlinks any
    ``goggles_*`` segment older than ``max_age_s`` seconds. Linux only
    (``/dev/shm`` is the visible mount); a no-op elsewhere.

    Args:
        max_age_s: Reap blocks whose mtime is older than this many
            seconds. The default leaves enough headroom that a busy
            client's just-allocated segments are never touched.

    Returns:
        Count of segments unlinked (zero on non-Linux or empty dir).
    """
    if not os.path.isdir(_LINUX_SHM_DIR):
        return 0
    try:
        names = os.listdir(_LINUX_SHM_DIR)
    except OSError:
        return 0
    cutoff = time.time() - max_age_s
    reaped = 0
    for name in names:
        if not name.startswith(_SHM_NAME_PREFIX):
            continue
        path = os.path.join(_LINUX_SHM_DIR, name)
        try:
            if os.path.getmtime(path) >= cutoff:
                continue
        except OSError:
            continue
        try:
            os.unlink(path)
            reaped += 1
        except OSError:
            pass
    return reaped


def _try_unlink_shm(name: str) -> None:
    """Best-effort unlink of a named shm block, tolerant to already-gone.

    Args:
        name: Name of the shared-memory block to remove.
    """
    try:
        shm = shared_memory.SharedMemory(name=name)
    except FileNotFoundError:
        return
    except OSError:
        return
    try:
        shm.close()
    except OSError:
        pass
    try:
        shm.unlink()
    except (FileNotFoundError, OSError):
        pass


# --- Cross-platform endpoint ----------------------------------------------


class _Endpoint(Protocol):
    """Platform abstraction for binding / connecting to a logical path."""

    @staticmethod
    def connect(path: str, timeout: float = 1.0) -> socket.socket | None:
        """Attempt to connect to a host at ``path``.

        Args:
            path: Logical endpoint identifier.
            timeout: Connect timeout in seconds.

        Returns:
            Connected stream socket, or None if no host is listening.
        """
        ...

    @staticmethod
    def bind(path: str) -> socket.socket:
        """Bind and listen at ``path``.

        Args:
            path: Logical endpoint identifier.

        Returns:
            The bound server socket.
        """
        ...

    @staticmethod
    def cleanup(path: str) -> None:
        """Remove filesystem artifacts associated with ``path``.

        Best-effort; callers should not rely on failure modes.

        Args:
            path: Logical endpoint identifier.
        """
        ...

    @staticmethod
    def accept_address_hint() -> str:
        """Return a human-readable label for logs.

        Returns:
            A short string identifying the endpoint family.
        """
        ...


class _UnixEndpoint:
    """AF_UNIX-based endpoint for Linux and macOS."""

    @staticmethod
    def connect(path: str, timeout: float = 1.0) -> socket.socket | None:
        if not os.path.exists(path):
            return None
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            sock.connect(path)
        except OSError:
            sock.close()
            return None
        sock.settimeout(None)
        return sock

    @staticmethod
    def bind(path: str) -> socket.socket:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
            try:
                os.chmod(parent, 0o700)
            except OSError:
                # Parent may be a shared dir (e.g. /tmp); don't fail here.
                pass

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        old_umask = os.umask(0o077)
        try:
            server.bind(path)
        finally:
            os.umask(old_umask)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        server.listen(64)
        return server

    @staticmethod
    def cleanup(path: str) -> None:
        try:
            os.unlink(path)
        except (FileNotFoundError, IsADirectoryError):
            pass
        except OSError:
            _log.exception("Failed to unlink socket file at %s", path)

    @staticmethod
    def accept_address_hint() -> str:
        return "AF_UNIX"


class _WindowsEndpoint:
    """TCP loopback endpoint for Windows.

    The "socket path" becomes a sidecar file recording the port the host
    chose; clients read it to find the host.
    """

    _LOOPBACK = "127.0.0.1"

    @staticmethod
    def _read_port(path: str) -> int | None:
        try:
            with open(path, encoding="utf-8") as f:
                data = f.read().strip()
        except OSError:
            return None
        try:
            return int(data)
        except ValueError:
            return None

    @staticmethod
    def connect(path: str, timeout: float = 1.0) -> socket.socket | None:
        port = _WindowsEndpoint._read_port(path)
        if port is None:
            return None
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            sock.connect((_WindowsEndpoint._LOOPBACK, port))
        except OSError:
            sock.close()
            return None
        sock.settimeout(None)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return sock

    @staticmethod
    def bind(path: str) -> socket.socket:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((_WindowsEndpoint._LOOPBACK, 0))
        server.listen(64)
        port = server.getsockname()[1]
        # Write port atomically: write to tmp, then rename.
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(str(port))
        os.replace(tmp, path)
        return server

    @staticmethod
    def cleanup(path: str) -> None:
        try:
            os.unlink(path)
        except (FileNotFoundError, IsADirectoryError):
            pass
        except OSError:
            _log.exception("Failed to remove discovery file at %s", path)

    @staticmethod
    def accept_address_hint() -> str:
        return "TCP loopback"


def _endpoint() -> type[_Endpoint]:
    if _IS_WINDOWS:
        return _WindowsEndpoint  # type: ignore[return-value]
    return _UnixEndpoint  # type: ignore[return-value]


# --- Transport Protocol ----------------------------------------------------


@runtime_checkable
class Transport(Protocol):
    """Contract for routing Goggles events to the EventBus."""

    @property
    def is_running(self) -> bool:
        """Whether the transport is accepting new events.

        Returns:
            True while the transport is live, False after shutdown or an
            unrecoverable send failure.
        """
        ...

    def emit(self, event: Event) -> None:
        """Route an event asynchronously (fire-and-forget).

        Args:
            event: Event to route.
        """
        ...

    def emit_sync(self, event: Event) -> None:
        """Route an event, blocking until it has reached the bus.

        In client mode this is best-effort: there is no cross-process ack,
        so it behaves like :meth:`emit` plus a local queue flush.

        Args:
            event: Event to route.
        """
        ...

    def attach(self, handlers: list[dict], scopes: list[str]) -> None:
        """Attach handlers under the given scopes.

        Args:
            handlers: Serialized handler dicts (see ``Handler.to_dict``).
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
            timeout: Max seconds per background thread for clean exit.
                ``None`` means wait indefinitely.
        """
        ...


# --- LocalTransport --------------------------------------------------------


_SENTINEL = object()  # "stop draining / sending" marker


class _SendItem:
    """Tuple-like wrapper for queued send items.

    Carrying the shm name (if any) alongside the frame bytes lets the
    sender free the shm block if sending fails, so the segment doesn't
    leak when shutdown drops the queue.
    """

    __slots__ = ("frame", "shm_name")

    def __init__(self, frame: bytes, shm_name: str | None = None) -> None:
        """Initialise the send item.

        Args:
            frame: Framed bytes to send on the wire.
            shm_name: Name of the associated shared-memory block, if any.
        """
        self.frame = frame
        self.shm_name = shm_name


class LocalTransport:
    """Cross-platform local transport with auto-elected host.

    On construction, the transport either:
      1. Connects to an existing host at ``socket_path`` (client mode), or
      2. Binds the endpoint and becomes host (running an accept loop that
         spawns one reader thread per connected client).

    Host mode owns an :class:`EventBus`; its own ``emit`` calls dispatch
    directly to the bus via a background drain thread. Client-mode ``emit``
    calls pickle the event and send it to the host; payloads above
    ``shm_threshold`` bytes travel through shared memory instead of the
    socket.
    """

    def __init__(
        self,
        socket_path: str | None = None,
        shm_threshold: int | None = None,
    ) -> None:
        """Initialize a LocalTransport.

        Args:
            socket_path: Path of the Unix domain socket (Unix) or
                discovery file (Windows). Defaults to the value of
                ``GOGGLES_SOCKET`` or a per-user path under
                ``$XDG_RUNTIME_DIR`` / the platform temp dir.
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
        self._endpoint = _endpoint()
        self._bus: EventBus = EventBus()
        self._is_host = False
        # ``_running`` gates new emits; it can be flipped to False either
        # by ``shutdown`` (graceful) or by the send loop when it detects
        # a dead socket. ``_shutdown_called`` is only set by ``shutdown``
        # so the cleanup path runs exactly once regardless of who first
        # noticed the transport was done.
        self._running = True
        self._shutdown_called = False
        self._state_lock = threading.Lock()

        # Host-side state
        self._server_sock: socket.socket | None = None
        self._accept_thread: threading.Thread | None = None
        self._reader_threads: list[threading.Thread] = []
        self._client_sockets: list[socket.socket] = []
        self._client_sockets_lock = threading.Lock()
        self._drain_queue: queue.SimpleQueue[Any] = queue.SimpleQueue()
        self._drain_thread: threading.Thread | None = None

        # Client-side state
        self._client_sock: socket.socket | None = None
        self._send_queue: queue.SimpleQueue[Any] = queue.SimpleQueue()
        self._send_thread: threading.Thread | None = None
        self._pending_shm: set[str] = set()
        self._pending_shm_lock = threading.Lock()

        self._connect_or_host()

    # ----- introspection ---------------------------------------------------

    @property
    def is_host(self) -> bool:
        """Whether this transport owns the EventBus (bound the socket)."""
        return self._is_host

    @property
    def is_running(self) -> bool:
        """Whether the transport is accepting new events."""
        return self._running

    # ----- setup -----------------------------------------------------------

    def _connect_or_host(self) -> None:
        """Bind first; fall back to client mode; only unlink stale files.

        On Unix we attempt :meth:`bind` before :meth:`connect`, so a live
        host always wins (its socket path is taken — our bind fails with
        ``EADDRINUSE``) and we never risk clobbering it on a transient
        connect failure (``PermissionError``, accept-queue pressure,
        etc.). Only when both ``bind`` and ``connect`` fail do we treat
        the file as stale and unlink it.

        On Windows the endpoint is a sidecar discovery file, not a
        bound name, so ``bind()`` always succeeds (it acquires a fresh
        ephemeral port). We therefore probe with ``connect`` first and
        bind iff there is no listener.

        Raises:
            OSError: If neither connecting nor binding succeed (for
                example, no write permission on the endpoint path).
        """
        if _IS_WINDOWS:
            self._connect_or_host_windows()
            return

        try:
            server = self._endpoint.bind(self._socket_path)
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                # Anything other than "path already taken" is a real
                # failure (no permission, no parent dir, ...) — don't
                # touch the existing file.
                raise
            # Path exists. A live host? Try to connect.
            if self._try_connect():
                self._start_send_worker()
                return
            # Bind said taken, connect said no listener: the file is
            # stale. Unlink and retry the bind exactly once.
            self._endpoint.cleanup(self._socket_path)
            try:
                server = self._endpoint.bind(self._socket_path)
            except OSError:
                # Another process raced us to the now-empty path.
                if self._try_connect(retries=5, backoff=0.02):
                    self._start_send_worker()
                    return
                raise

        self._server_sock = server
        self._is_host = True
        self._start_host_workers()

    def _connect_or_host_windows(self) -> None:
        """Windows variant: probe the discovery file, then bind if dead.

        Raises:
            OSError: If binding fails for non-recoverable reasons.
        """
        if self._try_connect():
            self._start_send_worker()
            return
        try:
            server = self._endpoint.bind(self._socket_path)
        except OSError:
            # Race: another process became host between our probe and
            # bind. Try one more time as client.
            if self._try_connect(retries=5, backoff=0.02):
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
            sock = self._endpoint.connect(self._socket_path)
            if sock is not None:
                self._client_sock = sock
                return True
            if attempt < retries - 1:
                time.sleep(backoff * (2**attempt))
        return False

    def _start_host_workers(self) -> None:
        """Start the accept thread and the drain thread on the host side.

        Sweeps ``/dev/shm`` for ``goggles_*`` segments left behind by a
        prior crashed host before spawning workers (Linux only; no-op
        elsewhere). The cutoff is conservative so live clients are
        never affected.
        """
        try:
            reaped = _reap_orphan_shm()
            if reaped:
                _log.info(
                    "Reaped %d stale goggles_* shm segment(s) at host startup",
                    reaped,
                )
        except Exception:
            _log.exception("Orphan-shm sweep raised; continuing")

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
        # On Linux, closing a listening AF_UNIX socket from another thread
        # does not wake a blocked accept(); poll so shutdown can break out.
        self._server_sock.settimeout(0.1)
        while self._running:
            try:
                conn, _ = self._server_sock.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            with self._client_sockets_lock:
                self._client_sockets.append(conn)
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
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                conn.close()
            except OSError:
                pass
            with self._client_sockets_lock:
                try:
                    self._client_sockets.remove(conn)
                except ValueError:
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
            # Client announced graceful disconnect; reader will hit EOF
            # after all queued frames have been consumed.
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
            assert isinstance(item, _SendItem)
            try:
                self._client_sock.sendall(item.frame)
            except (BrokenPipeError, ConnectionResetError, OSError) as exc:
                _log.warning(
                    "Goggles client socket closed (%s); dropping queued events",
                    exc,
                )
                self._fail_pending(item)
                return
            else:
                # Frame reached the kernel; host will unlink the shm after
                # it processes the frame. If the frame carried a shm name,
                # we can drop our tracking entry.
                if item.shm_name is not None:
                    with self._pending_shm_lock:
                        self._pending_shm.discard(item.shm_name)

    def _fail_pending(self, failing: _SendItem) -> None:
        """Unlink any shm blocks we created that won't be delivered.

        Args:
            failing: The send item whose sendall just errored; its shm
                needs unlinking too.
        """
        self._running = False
        names: list[str] = []
        if failing.shm_name is not None:
            names.append(failing.shm_name)
        while True:
            try:
                item = self._send_queue.get_nowait()
            except queue.Empty:
                break
            if item is _SENTINEL:
                continue
            if isinstance(item, _SendItem) and item.shm_name is not None:
                names.append(item.shm_name)
        for name in names:
            _try_unlink_shm(name)
        with self._pending_shm_lock:
            for name in names:
                self._pending_shm.discard(name)

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

    def _encode_event(self, event: Event) -> _SendItem:
        """Encode an Event as a framed send item.

        Args:
            event: Event to encode.

        Returns:
            Send item carrying the framed bytes and (for LARGE frames) the
            name of the shared-memory block the client allocated.

        Raises:
            BaseException: Propagates any failure from shm allocation,
                buffer copy, or pickling after reaping the allocated
                segment so it doesn't leak.
        """
        payload = event.payload
        if (
            self._shm_threshold > 0
            and isinstance(payload, np.ndarray)
            and payload.nbytes >= self._shm_threshold
        ):
            shm_name = _next_shm_name()
            shm = shared_memory.SharedMemory(
                create=True, name=shm_name, size=max(1, payload.nbytes)
            )
            with self._pending_shm_lock:
                self._pending_shm.add(shm_name)
            try:
                view = np.ndarray(
                    payload.shape, dtype=payload.dtype, buffer=shm.buf
                )
                view[...] = payload
                body = _pack_large(event, shm_name=shm_name)
            except BaseException:
                # Alloc/copy/pack failed; reap the segment we created.
                try:
                    shm.close()
                finally:
                    _try_unlink_shm(shm_name)
                with self._pending_shm_lock:
                    self._pending_shm.discard(shm_name)
                raise
            finally:
                shm.close()
            return _SendItem(
                frame=self._frame(_MSG_LARGE, body), shm_name=shm_name
            )
        return _SendItem(frame=self._frame(_MSG_SMALL, _pack_small(event)))

    # ----- public API ------------------------------------------------------

    def emit(self, event: Event) -> None:
        """Route an event asynchronously.

        Args:
            event: Event to route.
        """
        if not self._running:
            return
        if self._is_host:
            self._drain_queue.put(event)
            return
        try:
            item = self._encode_event(event)
        except Exception:
            _log.exception("Failed to encode event for transport")
            return
        self._send_queue.put(item)

    def emit_sync(self, event: Event) -> None:
        """Route an event and (host) block until the bus has seen it.

        In host mode the event is dispatched inline on the caller thread.
        In client mode this enqueues the event and then waits for the
        send queue to drain (best-effort; there is no cross-process ack).

        Args:
            event: Event to route.
        """
        if self._is_host:
            try:
                self._bus.emit(event)
            except Exception:
                _log.exception("Handler raised in emit_sync")
            return
        if not self._running:
            return
        try:
            item = self._encode_event(event)
        except Exception:
            _log.exception("Failed to encode event for transport")
            return
        self._send_queue.put(item)
        # Wait until the sender has caught up with this item.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if self._send_queue.empty():
                return
            time.sleep(0.001)
        _log.warning(
            "emit_sync: send queue did not drain within 5s; "
            "continuing without ack"
        )

    def attach(self, handlers: list[dict], scopes: list[str]) -> None:
        """Attach handlers under the given scopes.

        Args:
            handlers: Serialized handler dicts (see ``Handler.to_dict``).
            scopes: Scopes under which to attach.
        """
        if self._is_host:
            self._bus.attach(handlers, scopes)
            return
        if not self._running:
            return
        body = pickle.dumps((handlers, scopes), protocol=5)
        self._send_queue.put(_SendItem(frame=self._frame(_MSG_ATTACH, body)))

    def detach(self, handler_name: str, scope: str) -> None:
        """Detach a handler from the given scope.

        Args:
            handler_name: Name of the handler to detach.
            scope: Scope to detach from.
        """
        if self._is_host:
            self._bus.detach(handler_name, scope)
            return
        if not self._running:
            return
        body = pickle.dumps((handler_name, scope), protocol=5)
        self._send_queue.put(_SendItem(frame=self._frame(_MSG_DETACH, body)))

    def shutdown(self, timeout: float | None = None) -> None:
        """Flush pending work and release resources.

        Args:
            timeout: Max seconds to wait per background thread for a clean
                exit. ``None`` means wait indefinitely.
        """
        with self._state_lock:
            if self._shutdown_called:
                return
            self._shutdown_called = True
            self._running = False

        if self._is_host:
            self._shutdown_host(timeout)
        else:
            self._shutdown_client(timeout)

    def _shutdown_host(self, timeout: float | None) -> None:
        """Host-side shutdown: close listen socket, drain queue, close bus.

        Reader threads are unblocked by shutting down their peer sockets
        so ``recv`` returns immediately.

        Args:
            timeout: Max seconds to wait for each background thread.
        """
        if self._server_sock is not None:
            try:
                self._server_sock.close()
            except OSError:
                pass
            self._server_sock = None
        self._endpoint.cleanup(self._socket_path)

        # Wake up any reader threads blocked in recv().
        with self._client_sockets_lock:
            conns = list(self._client_sockets)
        for conn in conns:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

        if self._accept_thread is not None:
            self._accept_thread.join(timeout=timeout)

        for t in self._reader_threads:
            t.join(timeout=timeout)

        depth = self._drain_queue.qsize()
        if depth > 0:
            _log.info(
                "goggles host shutdown: waiting for %d queued events "
                "to drain (timeout=%s)",
                depth,
                "inf" if timeout is None else f"{timeout:.1f}s",
            )
        self._drain_queue.put(_SENTINEL)
        if self._drain_thread is not None:
            self._drain_thread.join(timeout=timeout)
            if self._drain_thread.is_alive():
                remaining = self._drain_queue.qsize()
                _log.warning(
                    "goggles host shutdown: drain did not finish within "
                    "%.1fs; %d events will not reach handlers",
                    timeout if timeout is not None else -1.0,
                    remaining,
                )

        try:
            self._bus.shutdown()
        except Exception:
            _log.exception("EventBus.shutdown raised")

    def _shutdown_client(self, timeout: float | None) -> None:
        """Client-side shutdown: enqueue BYE, drain, then close socket.

        BYE is placed on the send queue rather than sent out-of-band so
        it is ordered after every previously-enqueued frame, and so the
        send thread is the only one ever touching the socket (no lock
        contention with the caller). The sentinel that follows
        terminates the sender loop *after* it has drained.

        If the host is wedged (e.g. its receive buffer is full and it
        isn't draining), ``sendall`` would otherwise block the send
        thread forever. We install a finite socket timeout right before
        joining so a wedged ``sendall`` raises ``socket.timeout`` (an
        ``OSError``), the send loop's existing error handler treats it
        as a transport failure, and shutdown can complete in bounded
        time.

        Args:
            timeout: Max seconds to wait for the send queue to drain.
                ``None`` means wait indefinitely (no socket timeout).
        """
        # Enqueue a graceful BYE followed by the stop sentinel.
        try:
            self._send_queue.put(_SendItem(frame=self._frame(_MSG_BYE, b"")))
        except Exception:
            _log.exception("Failed to enqueue BYE frame")
        self._send_queue.put(_SENTINEL)

        if (
            self._client_sock is not None
            and timeout is not None
            and timeout > 0
        ):
            try:
                self._client_sock.settimeout(timeout)
            except OSError:
                pass

        if self._send_thread is not None:
            self._send_thread.join(timeout=timeout)
            if self._send_thread.is_alive():
                _log.warning(
                    "goggles client send thread did not drain within "
                    "%.2fs; %d undelivered events will be dropped",
                    timeout if timeout is not None else -1.0,
                    self._send_queue.qsize(),
                )

        # Anything still in the queue won't ship; unlink its shm blocks.
        self._fail_pending(_SendItem(frame=b""))

        # Final sweep: if emits raced with the send thread dying, we may
        # have pending_shm names whose send-items never made it into the
        # queue we just drained. Reap them now so /dev/shm stays clean.
        with self._pending_shm_lock:
            stragglers = list(self._pending_shm)
            self._pending_shm.clear()
        for name in stragglers:
            _try_unlink_shm(name)

        if self._client_sock is not None:
            try:
                self._client_sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                self._client_sock.close()
            except OSError:
                pass
            self._client_sock = None
