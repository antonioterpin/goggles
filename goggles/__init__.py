"""Goggles: Structured logging and experiment tracking.
===

This package provides a stable public API for logging experiments, metrics,
and media in a consistent and composable way.

>>>    import goggles as gg
>>>
>>>    logger = gg.get_logger(__name__)
>>>    gg.attach(
            gg.ConsoleHandler(name="examples.basic.console", level=gg.INFO),
            scopes=["global"],
        )
>>>    logger.info("Hello, world!")
>>>    gg.attach(
            gg.LocalStorageHandler(
            path=Path("examples/logs"),
            name="examples.jsonl",
        )
       )
>>>    logger.scalar("awesomeness", 42)

See Also:
    - README.md for detailed usage examples.
    - API docs for full reference of public interfaces.
    - Internal implementations live under `goggles/_core/`

"""  # noqa: D205

from __future__ import annotations

import portal
from portal import packlib, client_socket
from portal.buffers import SendBuffer, RecvBuffer
from portal.client import Client, Future
from portal.client_socket import ClientSocket
from portal.server_socket import ServerSocket
from collections import defaultdict
from typing import (
    Any,
    ClassVar,
    Final,
    Protocol,
    runtime_checkable,
    overload,
)
from collections.abc import Callable
from typing_extensions import Self
from typing import Literal, TypeVar, ParamSpec
import logging
import os
import select
import selectors
import time

from .types import Kind, Event, VectorField, Video, Image, Vector, Metrics
from ._core.integrations import ConsoleHandler, LocalStorageHandler
from ._core.decorators import timeit as _timeit, trace_on_error as _trace_on_error
from .shutdown import GracefulShutdown
from .config import load_configuration, save_configuration


P = ParamSpec("P")
T = TypeVar("T")


def timeit(
    severity: int = logging.INFO,
    name: str | None = None,
    scope: str = "global",
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Measure the execution time of a function via decorators.

    Args:
        severity: Log severity level for timing message.
        name: Optional name for the timing entry.
            If None, uses filename:function_name.
        scope: Scope of the logged event (e.g., "global" or "run").

    Returns:
        Decorated function with same signature as input.

    Example:
    >>> @timeit(severity=logging.DEBUG, name="my_function_timing")
    ... def my_function():
    ...     # function logic here
    ...     pass
    >>> my_function()
    DEBUG: my_function_timing took 0.123456s

    """
    # just forward to the real implementation
    return _timeit(severity=severity, name=name, scope=scope)


def trace_on_error(
    scope: str = "global",
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Trace errors and log function parameters via decorators.

    Args:
        scope: Scope of the logged event ("global" or "run").

    Example:
    >>> @trace_on_error()
    ... def my_function(x, y):
    ...     return x / y  # may raise ZeroDivisionError
    >>> my_function(10, 0)
    ERROR: Exception in my_function: division by zero, state:
    {'args': (10, 0), 'kwargs': {}}

    """
    # just forward to the real implementation
    return _trace_on_error(scope=scope)


# Goggles port for bus communication
GOGGLES_PORT: Final[str] = os.getenv("GOGGLES_PORT", "2304")

# ---------------------------------------------------------------------------
# Portal Monkey-Patches (Resilience)
# ---------------------------------------------------------------------------

# We patch portal at import time to handle ConnectionResetError/BrokenPipeError
# which otherwise lead to memory leaks (orphaned futures) or livelocks.
# NOTE: this is a temporary fix until either:
# - portal is fixed.
# - we switch to a different bus implementation.
# - we fork portal and fix it there.


# 1. Patch SendBuffer.send to propagate ConnectionResetError
_original_send = SendBuffer.send


def _safe_send(self, sock):
    try:
        return _original_send(self, sock)
    except (BrokenPipeError, ConnectionResetError) as e:
        raise ConnectionResetError from e


SendBuffer.send = _safe_send


# 2. Patch ServerSocket._loop to explicitly disconnect on write errors
def _patched_server_loop(self):
    writing = False
    try:
        while self.running or self._numsending():
            # Use 0 timeout if we have data to send to avoid artificially slow throughput
            timeout = 0 if writing else 0.2
            for key, mask in self.sel.select(timeout=timeout):
                if key.data == "signal":
                    os.read(self.get_signal, 1)
                    writing = self._numsending() > 0
                elif key.data is None and self.reading:
                    self._accept(key.fileobj)
                elif mask & selectors.EVENT_READ and self.reading:
                    self._recv(key.data)

            # Skip send processing if not actively writing
            if not writing:
                continue

            # Process pending send buffers
            pending = [conn for conn in self.conns.values() if conn.sendbufs]
            for conn in pending:
                try:
                    conn.sendbufs[0].send(conn.sock)
                    if conn.sendbufs[0].done():
                        conn.sendbufs.popleft()
                        if not any(c.sendbufs for c in self.conns.values()):
                            writing = False
                except BlockingIOError:
                    # Non-blocking send would block; try again later
                    pass
                except ConnectionResetError as e:
                    self._disconnect(conn, e)
    except Exception as e:
        self.error = e


ServerSocket._loop = _patched_server_loop

# 3. Silence "Dropping message" log spam
_original_server_log = ServerSocket._log


def _silent_server_log(self, *args):
    if args and isinstance(args[0], str) and "Dropping message" in args[0]:
        return
    return _original_server_log(self, *args)


ServerSocket._log = _silent_server_log


# 5. Patch ClientSocket._loop to fix future leaks and reconnection
def _patched_client_loop(self):
    recvbuf = RecvBuffer(maxsize=self.options.max_msg_size)
    sock = None
    poll = select.poll()
    poll.register(self.get_signal, select.POLLIN)
    isconn = False
    writing = False

    while self.running or (self.sendq and isconn):
        if not isconn:
            if not self.options.autoconn and not self.wantconn.wait(timeout=0.2):
                continue
            sock = self._connect()
            if not sock:
                break
            poll.register(sock, select.POLLIN)
            self.isconn.set()
            isconn = True
            if not self.options.autoconn:
                self.wantconn.clear()
            for callback in self.callbacks_conn:
                callback()

        try:
            # Poll with 0 timeout if writing to avoid delay
            timeout = 0 if writing else 0.2
            fds = [fd for fd, _ in poll.poll(timeout)]
            if self.get_signal in fds:
                os.read(self.get_signal, 1)
                writing = bool(self.sendq)

            try:
                recvbuf.recv(sock)
                if recvbuf.done():
                    if self.recvq.qsize() > self.options.max_recv_queue:
                        raise RuntimeError("Too many incoming messages enqueued")
                    msg = recvbuf.result()
                    self.recvq.put(msg)
                    for callback in self.callbacks_recv:
                        callback(msg)
                    recvbuf = RecvBuffer(maxsize=self.options.max_msg_size)
            except BlockingIOError:
                # Expected with non-blocking sockets; no data yet, retry on next poll
                pass

            if self.sendq:
                try:
                    self.sendq[0].send(sock)
                    if self.sendq[0].done():
                        self.sendq.popleft()
                        if not self.sendq:
                            writing = False
                except BlockingIOError:
                    pass
                except ConnectionResetError:
                    raise

        except OSError as e:
            # Disconnect and trigger high-level recovery
            detail = f"{type(e).__name__}: {e}" if str(e) else f"{type(e).__name__}"
            self._log(f"Connection to server lost ({detail})")
            self.isconn.clear()
            isconn = False
            if sock:
                try:
                    poll.unregister(sock)
                    sock.close()
                except Exception:
                    pass
            self.sendq.clear()  # Clear low-level queue; high-level Client will resend
            recvbuf = RecvBuffer(maxsize=self.options.max_msg_size)
            for callback in self.callbacks_disc:
                callback()
            continue

    if sock:
        try:
            poll.unregister(sock)
            sock.close()
        except Exception:
            pass
    try:
        poll.unregister(self.get_signal)
    except Exception:
        pass
    # Explicitly close poll if method exists (Python 3.4+)
    if hasattr(poll, "close"):
        try:
            poll.close()
        except Exception:
            # Ignore any errors; we're just cleaning up
            pass


ClientSocket._loop = _patched_client_loop

# 6. Patch Client.call to avoid infinite backpressure hang

_original_client_call = Client.call


def _safe_client_call(self, method, *data):
    # We duplicate logic because we need to inject the timeout loop

    reqnum = next(self.reqnum).to_bytes(8, "little", signed=False)
    start = time.time()
    # Hardcoded default 30s as requested ("always on"), overrideable for tests
    timeout_seconds = float(os.getenv("GOGGLES_TRANSPORT_TIMEOUT", "30.0"))

    with self.cond:
        while len(self.futures) >= self.maxinflight:
            # Check for total timeout
            if time.time() - start > timeout_seconds:
                raise TimeoutError(
                    f"Goggles: Timeout after {timeout_seconds}s waiting for in-flight requests to complete. "
                    "The server may be down or unresponsive. Consider checking server connectivity "
                    f"or increasing GOGGLES_TRANSPORT_TIMEOUT. (inflight={len(self.futures)})"
                )

            self.cond.wait(timeout=0.2)
            try:
                self.socket.require_connection(timeout=0)
            except TimeoutError:
                # Connection not established yet; try again later
                pass
            except (BrokenPipeError, ConnectionResetError, ConnectionRefusedError):
                # BrokenPipe/ConnectionReset/ConnectionRefused should behave like not connected
                pass

    with self.lock:
        self.waitmean[1] += time.time() - start
        self.waitmean[0] += 1
        self.sendrate[0] += 1

    if self.errors:  # Raise errors of dropped futures.
        raise self.errors.popleft()

    name = method.encode("utf-8")
    strlen = len(name).to_bytes(8, "little", signed=False)
    sendargs = (reqnum, strlen, name, *packlib.pack(data))
    rai = [False]
    future = Future(rai)
    future.sendargs = sendargs
    self.futures[reqnum] = future
    # Store future before sending request because the response may come fast
    # and the response handler runs in the socket's background thread.
    try:
        self.socket.send(*sendargs)
    except client_socket.Disconnected:
        future = self.futures.pop(reqnum)
        future.rai[0] = True
        raise
    return future


Client.call = _safe_client_call


# Handler registry for custom handlers
_HANDLER_REGISTRY: dict[str, type] = {}
GOGGLES_HOST: Final[str] = os.getenv("GOGGLES_HOST", "localhost")
GOGGLES_ASYNC: Final[bool] = os.getenv("GOGGLES_ASYNC", "1").lower() in (
    "1",
    "true",
    "yes",
)
GOGGLES_SUPPRESS_CONNECTIVITY_LOGS: Final[bool] = os.getenv(
    "GOGGLES_SUPPRESS_CONNECTIVITY_LOGS", "1"
).lower() in (
    "1",
    "true",
    "yes",
)

# Cache the implementation after first use to avoid repeated imports
__impl_get_bus: Callable[[], EventBus] | None = None


def _make_text_logger(
    name: str | None,
    scope: str,
    to_bind: dict[str, Any],
) -> TextLogger:
    from ._core.logger import CoreTextLogger

    return CoreTextLogger(name=name, scope=scope, to_bind=to_bind)


def _make_goggles_logger(
    name: str | None,
    scope: str,
    to_bind: dict[str, Any],
) -> GogglesLogger:
    from ._core.logger import CoreGogglesLogger

    return CoreGogglesLogger(name=name, scope=scope, to_bind=to_bind)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@overload
def get_logger(
    name: str | None = None,
    /,
    *,
    with_metrics: Literal[False] = False,
    scope: str = "global",
    **to_bind: Any,
) -> TextLogger: ...


@overload
def get_logger(
    name: str | None = None,
    /,
    *,
    with_metrics: Literal[True],
    scope: str = "global",
    **to_bind: Any,
) -> GogglesLogger: ...


def get_logger(
    name: str | None = None,
    /,
    *,
    with_metrics: bool = False,
    scope: str = "global",
    **to_bind: Any,
) -> TextLogger | GogglesLogger:
    """Return a structured logger (text-only by default, metrics-enabled on opt-in).

    This is the primary entry point for obtaining Goggles' structured loggers.
    Depending on the active run and configuration, the returned adapter will
    inject structured context (e.g., `RunContext` info) and persistent fields
    into each emitted log record.

    Args:
        name: Logger name. If None, the root logger is used.
        with_metrics: If True, return a logger exposing `.metrics`.
        scope: The logging scope, e.g., "global" or "run".
        **to_bind: Fields persisted and injected into every record.

    Returns:
        A text-only `TextLogger` by default,
        or a `GogglesLogger` when `with_metrics=True`.

    Examples:
        >>> # Text-only
        >>> log = get_logger("eval", dataset="CIFAR10")
        >>> log.info("starting")
        >>>
        >>> # Explicit metrics surface
        >>> tlog = get_logger("train", with_metrics=True, seed=0)
        >>> tlog.scalar("loss", 0.42, step=1)

    """
    if with_metrics:
        return _make_goggles_logger(name, scope, to_bind)
    else:
        return _make_text_logger(name, scope, to_bind)


@runtime_checkable
class TextLogger(Protocol):
    """Protocol for Goggles' structured logger adapters.

    This protocol defines the expected interface for logger adapters returned
    by `goggles.get_logger()`. It extends standard Python logging methods with
    support for persistent bound fields.

    Examples:
        >>> log = get_logger("goggles")
        >>> log.info("Hello, Goggles!", user="alice")
        >>> run_log = log.bind(run_id="exp42")
        >>> run_log.debug("Debugging info", step=1)
        ...    # Both log records include any persistent bound fields.
        ...    # The second record also includes run_id="exp42".

    """

    def bind(self, /, *, scope: str = "global", **fields: Any) -> Self:
        """Create a derived logger with additional persistent fields.

        Args:
            scope: The logging scope, e.g., "global" or "run".
            **fields: Additional fields persisted across all log records.

        Returns:
            New logger instance with persistent fields.

        """
        ...

    def log(
        self,
        severity: int,
        msg: str,
        /,
        *,
        step: int | None = None,
        time: float | None = None,
        async_mode: bool = GOGGLES_ASYNC,
        **extra: Any,
    ) -> None:
        """Log message at the given severity with optional structured extras.

        Args:
            severity: Numeric log level (e.g., logging.INFO).
            msg: The log message.
            step: The step number.
            time: The timestamp.
            async_mode: If True, do not block waiting for delivery.
            **extra:
                Additional structured key-value pairs for this record.

        """
        if severity >= logging.CRITICAL:
            self.critical(msg, step=step, time=time, async_mode=async_mode, **extra)
        elif severity >= logging.ERROR:
            self.error(msg, step=step, time=time, async_mode=async_mode, **extra)
        elif severity >= logging.WARNING:
            self.warning(msg, step=step, time=time, async_mode=async_mode, **extra)
        elif severity >= logging.INFO:
            self.info(msg, step=step, time=time, async_mode=async_mode, **extra)
        elif severity >= logging.DEBUG:
            self.debug(msg, step=step, time=time, async_mode=async_mode, **extra)
        else:
            # Below DEBUG level; no-op by default.
            pass

    def debug(
        self,
        msg: str,
        /,
        *,
        step: int | None = None,
        time: float | None = None,
        async_mode: bool = GOGGLES_ASYNC,
        **extra: Any,
    ) -> None:
        """Log a DEBUG message with optional structured extras.

        Args:
            msg: The log message.
            step: The step number.
            time: The timestamp.
            async_mode: If True, do not block waiting for delivery.
            **extra:
                Additional structured key-value pairs for this record.

        """
        ...

    def info(
        self,
        msg: str,
        /,
        *,
        step: int | None = None,
        time: float | None = None,
        async_mode: bool = GOGGLES_ASYNC,
        **extra: Any,
    ) -> None:
        """Log an INFO message with optional structured extras.

        Args:
            msg: The log message.
            step: The step number.
            time: The timestamp.
            async_mode: If True, do not block waiting for delivery.
            **extra:
                Additional structured key-value pairs for this record.

        """
        ...

    def warning(
        self,
        msg: str,
        /,
        *,
        step: int | None = None,
        time: float | None = None,
        async_mode: bool = GOGGLES_ASYNC,
        **extra: Any,
    ) -> None:
        """Log a WARNING message with optional structured extras.

        Args:
            msg: The log message.
            step: The step number.
            time: The timestamp.
            async_mode: If True, do not block waiting for delivery.
            **extra:
                Additional structured key-value pairs for this record.

        """
        ...

    def error(
        self,
        msg: str,
        /,
        *,
        step: int | None = None,
        time: float | None = None,
        async_mode: bool = GOGGLES_ASYNC,
        **extra: Any,
    ) -> None:
        """Log an ERROR message with optional structured extras.

        Args:
            msg: The log message.
            step: The step number.
            time: The timestamp.
            async_mode: If True, do not block waiting for delivery.
            **extra:
                Additional structured key-value pairs for this record.

        """
        ...

    def critical(
        self,
        msg: str,
        /,
        *,
        step: int | None = None,
        time: float | None = None,
        async_mode: bool = GOGGLES_ASYNC,
        **extra: Any,
    ) -> None:
        """Log a CRITICAL message with current exception info attached.

        Args:
            msg: The log message.
            step: The step number.
            time: The timestamp.
            async_mode: If True, do not block waiting for delivery.
            **extra:
                Additional structured key-value pairs for this record.

        """
        ...


@runtime_checkable
class DataLogger(Protocol):
    """Protocol for logging metrics, media, artifacts, and analytics data."""

    def push(
        self,
        metrics: Metrics,
        step: int,
        *,
        time: float | None = None,
        async_mode: bool = GOGGLES_ASYNC,
        **extra: Any,
    ) -> None:
        """Emit a batch of scalar metrics.

        Args:
            metrics: (Name,value) pairs.
            step: Global step index.
            time: Optional global timestamp.
            async_mode: If True, do not block waiting for delivery.
            **extra:
                Additional routing metadata (e.g., split="train").

        """
        ...

    def scalar(
        self,
        name: str,
        value: float | int,
        step: int,
        *,
        time: float | None = None,
        async_mode: bool = GOGGLES_ASYNC,
        **extra: Any,
    ) -> None:
        """Emit a single scalar metric.

        Args:
            name: Metric name.
            value: Metric value.
            step: Global step index.
            time: Optional global timestamp.
            async_mode: If True, do not block waiting for delivery.
            **extra:
                Additional routing metadata (e.g., split="train").

        """
        ...

    def image(
        self,
        image: Image,
        step: int,
        *,
        name: str | None = None,
        format: str = "png",
        time: float | None = None,
        async_mode: bool = GOGGLES_ASYNC,
        **extra: Any,
    ) -> None:
        """Emit an image artifact (encoded bytes).

        Args:
            image: Image.
            step: Global step index.
            name: Optional artifact name.
            format: Image format, e.g., "png", "jpeg".
            time: Optional global timestamp.
            async_mode: If True, do not block waiting for delivery.
            **extra: Additional routing metadata.

        """
        ...

    def video(
        self,
        video: Video,
        step: int,
        *,
        name: str | None = None,
        fps: int = 30,
        format: str = "gif",
        time: float | None = None,
        async_mode: bool = GOGGLES_ASYNC,
        **extra: Any,
    ) -> None:
        """Emit a video artifact (encoded bytes).

        Args:
            video: Video.
            step: Global step index.
            name: Optional artifact name.
            fps: Frames per second.
            format: Video format, e.g., "gif", "mp4".
            time: Optional global timestamp.
            async_mode: If True, do not block waiting for delivery.
            **extra: Additional routing metadata.

        """
        ...

    def artifact(
        self,
        data: Any,
        step: int,
        *,
        name: str | None = None,
        format: str = "bin",
        time: float | None = None,
        async_mode: bool = GOGGLES_ASYNC,
        **extra: Any,
    ) -> None:
        """Emit a generic artifact (encoded bytes).

        Args:
            data: Artifact data.
            step: Global step index.
            name: Optional artifact name.
            format: Artifact format, e.g., "txt", "bin".
            time: Optional global timestamp.
            async_mode: If True, do not block waiting for delivery.
            **extra: Additional routing metadata.

        """
        ...

    def vector_field(
        self,
        vector_field: VectorField,
        step: int,
        *,
        name: str | None = None,
        time: float | None = None,
        async_mode: bool = GOGGLES_ASYNC,
        **extra: Any,
    ) -> None:
        """Emit a vector field artifact.

        Args:
            vector_field: Vector field data.
            step: Global step index.
            name: Optional artifact name.
            time: Optional global timestamp.
            async_mode: If True, do not block waiting for delivery.
            **extra: Additional routing metadata.

        """
        ...

    def histogram(
        self,
        histogram: Vector,
        step: int,
        *,
        name: str | None = None,
        time: float | None = None,
        static: bool = False,
        async_mode: bool = GOGGLES_ASYNC,
        **extra: Any,
    ) -> None:
        """Emit a histogram artifact.

        Args:
            histogram: Histogram data.
            step: Global step index.
            name: Optional artifact name.
            time: Optional global timestamp.
            static: If True, treat as static histogram.
            async_mode: If True, do not block waiting for delivery.
            **extra: Additional routing metadata.

        """
        ...


@runtime_checkable
class GogglesLogger(TextLogger, DataLogger, Protocol):
    """Protocol for Goggles loggers with metrics support.

    Composite logger combining text logging with a metrics facet.

    Examples:
        >>> # Text-only
        >>> log = get_logger("eval", dataset="CIFAR10")
        >>> log.info("starting")
        >>>
        >>> # Explicit metrics surface
        >>> tlog = get_logger("train", with_metrics=True, seed=0)
        >>> tlog.scalar("loss", 0.42, step=1)
        >>> tlog.info("Training step completed")
        ...   # Both log records include any persistent bound fields.
        ...   # The second record also includes run_id="exp42".

    """


@runtime_checkable
class Handler(Protocol):
    """Protocol for EventBus handlers.

    Attributes:
        name: Stable handler identifier for diagnostics.
        capabilities:
            Supported kinds, e.g. {'logs','metrics','artifacts', ...}.

    """

    name: str
    capabilities: ClassVar[frozenset[Kind]]

    def can_handle(self, kind: Kind) -> bool:
        """Return whether this handler can process events of the given kind.

        Args:
            kind:
                The kind of event ("log", "metric", "image", "artifact").

        Returns:
            True if the handler can process the event kind,
                False otherwise.

        """
        ...

    def handle(self, event: Event) -> None:
        """Handle an emitted event.

        Args:
            event: The event to handle.

        """
        ...

    def open(self) -> None:
        """Initialize the handler (called when entering a scope)."""
        ...

    def close(self) -> None:
        """Flush and release resources (called when leaving a scope).

        Args:
            run: The active run context if any.

        """
        ...

    def to_dict(self) -> dict:
        """Serialize the handler.

        This method is needed during attachment. Will be called before binding.

        Returns:
            A dictionary that allows to instantiate the Handler.
                Must contain:
                    - "cls": The handler class name.
                    - "data": The handler data to be used in from_dict.

        """
        ...

    @classmethod
    def from_dict(cls, serialized: dict) -> Self:
        """De-serialize the handler.

        Args:
            serialized: Serialized handler with handler.to_dict

        Returns:
            The Handler instance.

        """
        ...


# ---------------------------------------------------------------------------
# EventBus and run management
# ---------------------------------------------------------------------------
class EventBus:
    """Protocol for the process-wide event router."""

    handlers: dict[str, Handler]
    scopes: dict[str, set[str]]

    def __init__(self):
        super().__init__()
        self.handlers: dict[str, Handler] = {}
        self.scopes: dict[str, set[str]] = defaultdict(set)

    def shutdown(self) -> None:
        """Shutdown the EventBus and close all handlers."""
        copy_map = {
            scope: handlers_names.copy()
            for scope, handlers_names in self.scopes.items()
        }
        for scope, handlers_names in copy_map.items():
            for handler_name in handlers_names:
                self.detach(handler_name, scope)

    def attach(self, handlers: list[dict], scopes: list[str]) -> None:
        """Attach a handler under the given scope.

        Args:
            handlers:
                The serialized handlers to attach to the scopes.
            scopes: The scopes under which to attach.

        """
        for handler_dict in handlers:
            handler_class = _get_handler_class(handler_dict["cls"])
            handler = handler_class.from_dict(handler_dict["data"])
            if handler.name not in self.handlers:
                # Initialize handler and store it
                handler.open()
                self.handlers[handler.name] = handler

            # Add to requested scopes
            for scope in scopes:
                if scope not in self.scopes:
                    self.scopes[scope] = set()
                self.scopes[scope].add(handler.name)

    def detach(self, handler_name: str, scope: str) -> None:
        """Detach a handler from the given scope.

        Args:
            handler_name: The name of the handler to detach.
            scope: The scope from which to detach.

        Raises:
          ValueError: If the handler was not attached under the requested scope.

        """
        if scope not in self.scopes or handler_name not in self.scopes[scope]:
            raise ValueError(
                f"Handler '{handler_name}' not attached under scope '{scope}'"
            )
        self.scopes[scope].remove(handler_name)
        if not self.scopes[scope]:
            del self.scopes[scope]
        if not any(handler_name in self.scopes[s] for s in self.scopes):
            self.handlers[handler_name].close()
            del self.handlers[handler_name]

    def emit(self, event: dict | Event) -> None:
        """Emit an event to eligible handlers (errors isolated per handler).

        Args:
            event: The event (serialized) to emit, or an Event instance.

        """
        if isinstance(event, dict):
            event = Event.from_dict(event)
        elif not isinstance(event, Event):
            raise TypeError(f"emit expects a dict or Event, got {type(event)!r}")

        # collect all scopes that this event should hit:
        scope = event.scope
        prefix = scope + "."

        # Example:
        # event.scope == "global"
        # matches: "global", "global.local1", "global.local2", but not "another"
        target_scopes = [s for s in self.scopes if s == scope or s.startswith(prefix)]

        if not target_scopes:
            return

        # Ensure we don't call the same handler twice
        # if it's attached to multiple scopes
        seen_handlers: set[str] = set()

        for s in target_scopes:
            for handler_name in self.scopes[s]:
                if handler_name in seen_handlers:
                    continue
                handler = self.handlers.get(handler_name)
                if handler and handler.can_handle(event.kind):
                    handler.handle(event)
                seen_handlers.add(handler_name)


def get_bus() -> portal.Client:
    """Return the process-wide EventBus singleton client.

    The EventBus owns handlers and routes events based on scope and kind.

    Returns:
        The singleton EventBus client.

    """
    global __impl_get_bus
    if __impl_get_bus is None:
        from ._core.routing import get_bus as _impl_get_bus

        __impl_get_bus = _impl_get_bus  # type: ignore
    return __impl_get_bus()  # type: ignore


def attach(handler: Handler, scopes: list[str] = ["global"]) -> None:
    """Attach a handler to the global EventBus under the specified scopes.

    Args:
        handler: The handler to attach.
        scopes: The scopes under which to attach.

    Raises:
        ValueError: If the handler disallows the requested scope.

    """
    bus = get_bus()
    bus.attach([handler.to_dict()], scopes)


def detach(handler_name: str, scope: str) -> None:
    """Detach a handler from the global EventBus under the specified scope.

    Args:
        handler_name: The name of the handler to detach.
        scope: The scope from which to detach.

    Raises:
        ValueError: If the handler was not attached under the requested scope.

    """
    bus = get_bus()
    bus.detach(handler_name, scope)


def finish() -> None:
    """Shutdown the global EventBus and close all handlers."""
    bus = get_bus()
    bus.shutdown().result()


def register_handler(handler_class: type) -> None:
    """Register a custom handler class for serialization/deserialization.

    Args:
        handler_class: The handler class to register. Must have a __name__ attribute.

    Example:
        class CustomHandler(gg.ConsoleHandler):
            pass

        gg.register_handler(CustomHandler)

    """
    _HANDLER_REGISTRY[handler_class.__name__] = handler_class


def _get_handler_class(class_name: str) -> type:
    """Get a handler class by name from registry or globals.

    Args:
        class_name: Name of the handler class.

    Returns:
        The handler class.

    Raises:
        KeyError: If the handler class is not found.

    """
    # First check the registry for custom handlers
    if class_name in _HANDLER_REGISTRY:
        return _HANDLER_REGISTRY[class_name]

    # Fall back to globals for built-in handlers
    if class_name in globals():
        return globals()[class_name]

    raise KeyError(
        f"Handler class '{class_name}' not found. "
        f"Available handlers: {list(_HANDLER_REGISTRY.keys()) + [k for k in globals().keys() if k.endswith('Handler')]}"
    )


# ---------------------------------------------------------------------------
# Logging Levels
# ---------------------------------------------------------------------------

INFO = logging.INFO
DEBUG = logging.DEBUG
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

try:
    from ._core.integrations.wandb import WandBHandler
except Exception:
    WandBHandler = None

__all__ = [
    "TextLogger",
    "GogglesLogger",
    "get_logger",
    "attach",
    "detach",
    "register_handler",
    "load_configuration",
    "save_configuration",
    "timeit",
    "trace_on_error",
    "GracefulShutdown",
    "ConsoleHandler",
    "LocalStorageHandler",
    "WandBHandler",
    "Event",
    "Kind",
    "Metrics",
    "Image",
    "Video",
    "Vector",
    "VectorField",
    "INFO",
    "DEBUG",
    "WARNING",
    "ERROR",
    "CRITICAL",
]

# ---------------------------------------------------------------------------
# Import-time safety
# ---------------------------------------------------------------------------

# Attach a NullHandler so importing goggles never emits logs by default.

_logger = logging.getLogger(__name__)
if not any(isinstance(h, logging.NullHandler) for h in _logger.handlers):
    _logger.addHandler(logging.NullHandler())
