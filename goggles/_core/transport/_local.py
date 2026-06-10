"""``LocalTransport``: cross-platform same-machine transport.

The first process to bind the configured socket path becomes the host
(owns the :class:`EventBus` and dispatches events to attached handlers);
subsequent processes on the same machine connect as clients.

On Unix (Linux, macOS) the transport uses ``AF_UNIX`` streams with the
socket file protected at ``0o600``. On Windows, ``AF_UNIX`` is unreliable
across Python versions, so the transport binds a TCP loopback socket on
``127.0.0.1`` and writes the chosen port to a sidecar discovery file.

Payloads above ``shm_threshold`` bytes take a zero-copy shared-memory
side-channel: the client writes the numpy buffer into a
``multiprocessing.shared_memory.SharedMemory`` block and sends only
metadata over the socket; the host maps the same block, copies the
view out, and unlinks the block.
"""

from __future__ import annotations

import errno
import logging
import os
import pickle
import queue
import socket
import struct
import threading
import time
from collections.abc import Callable
from multiprocessing import shared_memory
from typing import TYPE_CHECKING, Any

import numpy as np

from goggles.types import Event

from ._endpoints import _endpoint
from ._frames import (
    _HEADER_FMT,
    _HEADER_SIZE,
    _IS_WINDOWS,
    _MSG_ATTACH,
    _MSG_BYE,
    _MSG_DETACH,
    _MSG_LARGE,
    _MSG_SMALL,
    _default_shm_threshold,
    _default_socket_path,
    _next_shm_name,
    _pack_large,
    _pack_small_frame,
    _reap_orphan_shm,
    _recvall,
    _try_unlink_shm,
    _unpack_large,
    _unpack_small,
    _untrack_shm,
)

if TYPE_CHECKING:
    from goggles import EventBus

_log = logging.getLogger(__name__)


_SENTINEL = object()  # "stop draining / sending" marker

_DEFAULT_HOST_IDLE_TIMEOUT_S = 5.0


def _host_idle_timeout_s() -> float:
    """Seconds a host with no clients waits before self-reaping.

    A dedicated host stays alive while any client is connected; once the last
    one leaves it winds down after this grace (cancelled if a client
    reconnects first). Tunable via ``GOGGLES_HOST_IDLE_TIMEOUT``; a
    non-positive or unparsable value falls back to the default.

    Returns:
        The idle grace in seconds.
    """
    try:
        value = float(
            os.getenv(
                "GOGGLES_HOST_IDLE_TIMEOUT", str(_DEFAULT_HOST_IDLE_TIMEOUT_S)
            )
        )
    except ValueError:
        return _DEFAULT_HOST_IDLE_TIMEOUT_S
    return value if value > 0 else _DEFAULT_HOST_IDLE_TIMEOUT_S


class _SendItem:
    """Tuple-like wrapper for queued send items.

    Carrying the shm name (if any) alongside the frame bytes lets the
    sender free the shm block if sending fails, so the segment doesn't
    leak when shutdown drops the queue.
    """

    __slots__ = ("frame", "shm_name")

    def __init__(
        self,
        frame: bytes | bytearray,
        shm_name: str | None = None,
    ) -> None:
        """Initialise the send item.

        Args:
            frame: Framed bytes to send on the wire. ``bytes`` for the
                low-frequency control frames built by
                :meth:`LocalTransport._frame` (ATTACH/DETACH/BYE,
                LARGE), ``bytearray`` for the SMALL hot-path frame
                produced by :func:`_pack_small_frame`.
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
        # Per-host secret minted by the endpoint at bind time. Used by
        # endpoints that enforce a wire-level handshake (Windows TCP);
        # ``None`` for endpoints whose isolation is filesystem-based
        # (Unix). Set in ``_connect_or_host``.
        self._endpoint_secret: bytes | None = None
        self._accept_thread: threading.Thread | None = None
        self._reader_threads: list[threading.Thread] = []
        self._client_sockets: list[socket.socket] = []
        self._client_sockets_lock = threading.Lock()
        # Self-reap: when set (by the dedicated host), the host winds down once
        # its last client disconnects and none reconnects within the grace, so
        # its lifetime is tied to "any client connected", not to whichever
        # process happened to spawn it. Guarded by ``_client_sockets_lock``.
        self._idle_callback: Callable[[], None] | None = None
        self._idle_timer: threading.Timer | None = None
        self._idle_timeout_s = _host_idle_timeout_s()
        self._drain_queue: queue.SimpleQueue[Any] = queue.SimpleQueue()
        self._drain_thread: threading.Thread | None = None
        # Set during a bounded shutdown after the drain join times out:
        # the drain loop then discards remaining queued events instead of
        # forwarding them to handlers. Keeps shutdown semantics honest
        # (handlers see "nothing more is coming") and stops new
        # ``handler.handle()`` calls from racing ``handler.close()``.
        self._drain_aborted = threading.Event()

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
            server, secret = self._endpoint.bind(self._socket_path)
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
                server, secret = self._endpoint.bind(self._socket_path)
            except OSError:
                # Another process raced us to the now-empty path.
                if self._try_connect(retries=5, backoff=0.02):
                    self._start_send_worker()
                    return
                raise

        self._server_sock = server
        self._endpoint_secret = secret
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
            server, secret = self._endpoint.bind(self._socket_path)
        except OSError:
            # Race: another process became host between our probe and
            # bind. Try one more time as client.
            if self._try_connect(retries=5, backoff=0.02):
                self._start_send_worker()
                return
            raise
        self._server_sock = server
        self._endpoint_secret = secret
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

    # ----- host self-reap --------------------------------------------------

    def set_idle_callback(self, callback: Callable[[], None]) -> None:
        """Wind the host down when its last client disconnects (host only).

        The dedicated host passes a callback that requests a graceful
        shutdown. It fires once every client has disconnected and none
        reconnects within ``GOGGLES_HOST_IDLE_TIMEOUT``. Arms the grace
        immediately when no client is connected yet, so a host that nobody
        ever connects to also winds down instead of lingering forever.

        Args:
            callback: Invoked (once) to request the host wind down.
        """
        with self._client_sockets_lock:
            self._idle_callback = callback
            if not self._client_sockets:
                self._arm_idle_timer_locked()

    def _arm_idle_timer_locked(self, delay: float | None = None) -> None:
        """Start (or restart) the idle reap timer. Caller holds the lock.

        Args:
            delay: Seconds before reaping; defaults to the idle grace. A clean
                last-client disconnect passes ``0.0`` to reap promptly (so a
                caller's ``finish()`` can finalize handlers), while an unclean
                EOF keeps the grace to avoid churn if a client reconnects.
        """
        if self._idle_timer is not None:
            self._idle_timer.cancel()
        when = self._idle_timeout_s if delay is None else delay
        self._idle_timer = threading.Timer(when, self._reap_if_idle)
        self._idle_timer.daemon = True
        self._idle_timer.start()

    def _cancel_idle_timer_locked(self) -> None:
        """Cancel any pending idle grace timer. Caller holds the lock."""
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None

    def _reap_if_idle(self) -> None:
        """Fire the idle callback iff still no clients (grace elapsed)."""
        with self._client_sockets_lock:
            if self._client_sockets:
                return  # a client reconnected within the grace
            callback = self._idle_callback
        if callback is not None:
            callback()

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
                self._cancel_idle_timer_locked()  # a client is connected
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

        Authorizes the peer before reading any payload so an
        unauthenticated client (Windows TCP loopback) is dropped before
        anything they send is fed to ``pickle.loads``.

        Args:
            conn: Connected client socket.
        """
        clean = False  # set once the client sends BYE (graceful disconnect)
        try:
            if not self._endpoint.authorize(conn, self._endpoint_secret):
                _log.warning(
                    "goggles host: rejected client at %s (handshake failed)",
                    self._endpoint.accept_address_hint(),
                )
                return
            # Loop until the peer goes away (EOF / error), NOT until
            # ``self._running`` flips. A graceful client sends its frames,
            # then ``BYE``, then closes; the reader must drain all buffered
            # frames and hit EOF so they reach the drain queue. Stopping early
            # on ``_running`` would discard the tail still in the socket --
            # shutdown wakes any reader still blocked here via ``SHUT_RDWR``.
            while True:
                header = _recvall(conn, _HEADER_SIZE)
                if header is None:
                    return
                kind, length = struct.unpack(_HEADER_FMT, header)
                body = _recvall(conn, length) if length else b""
                if body is None:
                    return
                if kind == _MSG_BYE:
                    clean = True  # the client is leaving gracefully
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
                if not self._client_sockets and self._idle_callback is not None:
                    # Last client gone: reap promptly on a clean BYE (so its
                    # finish() can finalize handlers), else keep the grace in
                    # case an unclean drop is a reconnecting client.
                    self._arm_idle_timer_locked(0.0 if clean else None)

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
            if self._drain_aborted.is_set():
                # Bounded shutdown timed out: discard the remaining queue
                # rather than dispatching it concurrently with close().
                continue
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
            # goggles unlinks the block from the consumer side (and sweeps
            # crash leftovers at host startup), so detach it from this
            # process's resource tracker; otherwise the tracker reports every
            # block as leaked and warns once per payload at exit.
            _untrack_shm(shm)
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
        return _SendItem(frame=_pack_small_frame(event))

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
        with self._client_sockets_lock:
            self._cancel_idle_timer_locked()
        if self._server_sock is not None:
            try:
                self._server_sock.close()
            except OSError:
                pass
            self._server_sock = None
        self._endpoint.cleanup(self._socket_path)

        # Stop accepting first so no new client races the drain below.
        if self._accept_thread is not None:
            self._accept_thread.join(timeout=timeout)

        # Graceful drain: a client that disconnected cleanly (frames, then
        # BYE, then close) makes its reader hit EOF once the buffered frames
        # are consumed, and the reader then removes itself from
        # ``_client_sockets``. Give readers a brief window to flush every
        # frame into the drain queue that way before forcibly waking any
        # still blocked (idle or crashed peers that never closed) -- without
        # this, ``SHUT_RDWR`` below would discard frames still in the socket.
        drain_grace = 0.5 if timeout is None else min(timeout, 0.5)
        deadline = time.monotonic() + drain_grace
        while time.monotonic() < deadline:
            with self._client_sockets_lock:
                if not self._client_sockets:
                    break
            time.sleep(0.005)

        # Wake up any reader threads still blocked in recv().
        with self._client_sockets_lock:
            conns = list(self._client_sockets)
        for conn in conns:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

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
                    "%.1fs; %d queued events will be discarded",
                    timeout if timeout is not None else -1.0,
                    remaining,
                )
                # Tell the loop to stop dispatching; it will drain the
                # rest of the queue as discards and exit at the SENTINEL.
                # Wait briefly for that — once dispatch is bypassed the
                # loop is fast. If the in-flight ``handler.handle()``
                # call is still running, the thread won't exit until it
                # returns; that's a race we can't safely interrupt, but
                # marking the abort first means no further events are
                # forwarded after this point.
                self._drain_aborted.set()
                self._drain_thread.join(timeout=timeout)

        try:
            self._bus.shutdown(timeout=timeout)
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
