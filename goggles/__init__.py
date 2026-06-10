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

"""

from __future__ import annotations

import gc
import logging
import os
import threading
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Final,
    Literal,
    ParamSpec,
    Protocol,
    TypeVar,
    cast,
    overload,
    runtime_checkable,
)

from typing_extensions import Self

if TYPE_CHECKING:
    from goggles._core.transport import Transport

from . import filters
from ._core.decorators import timeit as _timeit
from ._core.decorators import trace_on_error as _trace_on_error
from ._core.integrations import ConsoleHandler, LocalStorageHandler
from .config import PrettyConfig, load_configuration, save_configuration
from .shutdown import GracefulShutdown
from .types import (
    Event,
    Image,
    Kind,
    Metrics,
    Trajectories,
    Vector,
    VectorField,
    Video,
)

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

    Returns:
        Decorated function with same signature as input.

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


# Handler registry for custom handlers
_HANDLER_REGISTRY: dict[str, type] = {}
GOGGLES_ASYNC: Final[bool] = os.getenv("GOGGLES_ASYNC", "1").lower() in (
    "1",
    "true",
    "yes",
)

# Cache the implementation after first use to avoid repeated imports
__impl_get_bus: Callable[[], Transport] | None = None


def freeze() -> None:
    """Promote currently-live objects out of the GC scan set.

    After attaching handlers and finishing application setup, call
    ``freeze()`` once before entering a hot logging loop. It wraps
    :func:`gc.freeze`: the collector still runs on churn allocated *after*
    this call, but it no longer rescans the thousands of long-lived
    startup objects. This collapses the gen-2 GC pauses that otherwise
    show up as millisecond-scale spikes in high-frequency logging.

    This function is idempotent and safe to call even if the user is not
    concerned about GC jitter; it simply returns without effect if there
    is nothing to freeze.
    """
    gc.freeze()


def _make_text_logger(
    name: str | None,
    scope: str,
    level: int,
    **to_bind: Any,
) -> TextLogger:
    # Importing here to avoid circular imports
    from ._core.logger import CoreTextLogger  # noqa: PLC0415

    return CoreTextLogger(name=name, scope=scope, level=level, **to_bind)


def _make_goggles_logger(
    name: str | None,
    scope: str,
    level: int,
    **to_bind: Any,
) -> GogglesLogger:
    # Importing here to avoid circular imports
    from ._core.logger import CoreGogglesLogger  # noqa: PLC0415

    return CoreGogglesLogger(name=name, scope=scope, level=level, **to_bind)


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
    level: int = logging.NOTSET,
    **to_bind: Any,
) -> TextLogger: ...


@overload
def get_logger(
    name: str | None = None,
    /,
    *,
    with_metrics: Literal[True],
    scope: str = "global",
    level: int = logging.NOTSET,
    **to_bind: Any,
) -> GogglesLogger: ...


def get_logger(
    name: str | None = None,
    /,
    *,
    with_metrics: bool = False,
    scope: str = "global",
    level: int = logging.NOTSET,
    **to_bind: Any,
) -> TextLogger | GogglesLogger:
    """Return a structured logger.

    This is the primary entry point for obtaining Goggles' structured loggers.
    Depending on the active run and configuration, the returned adapter will
    inject structured context (e.g., `RunContext` info) and persistent fields
    into each emitted log record.

    The logger is by default a text-only logger, but it can be configured
    to return a `GogglesLogger` which exposes additional methods for logging
    metrics, media, and artifacts.

    Args:
        name: Logger name. If None, the root logger is used.
        with_metrics: If True, return a logger exposing `.metrics`.
        scope: The logging scope, e.g., "global" or "run".
        level: Minimum severity this logger will emit. Defaults to
            ``logging.NOTSET`` (emit everything); pass
            ``logging.DEBUG`` / ``logging.INFO`` / etc. to drop
            lower-severity records at the source. Also settable later
            via ``logger.set_level(...)``.
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
        >>>
        >>> # Per-file DEBUG without flooding the rest of the app.
        >>> dbg = get_logger(__name__, level=logging.DEBUG)
        >>> dbg.debug("detailed trace")

    """
    if with_metrics:
        return _make_goggles_logger(name, scope, level, **to_bind)
    else:
        return _make_text_logger(name, scope, level, **to_bind)


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

    def set_level(self, level: int) -> None:
        """Set the minimum severity this logger will emit.

        Calls below ``level`` are dropped before reaching the transport.
        Only affects this logger instance; sibling loggers obtained via
        separate ``get_logger(...)`` calls are unaffected.

        Args:
            level: Standard ``logging`` level (e.g. ``logging.DEBUG``).
                ``logging.NOTSET`` (the default) forwards every call.
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
            self.critical(
                msg, step=step, time=time, async_mode=async_mode, **extra
            )
        elif severity >= logging.ERROR:
            self.error(
                msg, step=step, time=time, async_mode=async_mode, **extra
            )
        elif severity >= logging.WARNING:
            self.warning(
                msg, step=step, time=time, async_mode=async_mode, **extra
            )
        elif severity >= logging.INFO:
            self.info(msg, step=step, time=time, async_mode=async_mode, **extra)
        elif severity >= logging.DEBUG:
            self.debug(
                msg, step=step, time=time, async_mode=async_mode, **extra
            )
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

        Notes:
            * For grayscale videos, input shape can be (F, H, W) or (F, H, W, 1)
                or (B, F, 1, H, W).
            With F the number of frames, and B the batch size.

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

    def trajectories(
        self,
        trajectories: Trajectories,
        step: int,
        *,
        name: str | None = None,
        time: float | None = None,
        async_mode: bool = GOGGLES_ASYNC,
        **extra: Any,
    ) -> None:
        """Emit a batch of particle trajectories.

        Args:
            trajectories: Array of shape ``(N, L, dim)`` with ``dim`` in
                ``{2, 3}``.
            step: Global step index.
            name: Optional artifact name.
            time: Optional global timestamp.
            async_mode: If True, do not block waiting for delivery.
            **extra: Additional routing metadata (e.g.
                ``store_visualization=True`` to also save a PNG preview).
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

    def dictionary(
        self,
        name: str,
        data: dict,
        step: int,
        *,
        time: float | None = None,
        async_mode: bool = GOGGLES_ASYNC,
        **extra: Any,
    ) -> None:
        """Emit all key-value pairs in a dictionary as separate metrics.

        Notes:
            * The `name` parameter serves as base name for the emitted metrics.
            * Each key in the `data` dictionary is appended to the base name to
            form the full metric name (e.g., `name/key`).
            * Values in the dictionary are emitted according to their type:
                - Scalars (int, float) are emitted as single metrics.
                - 1D arrays are emitted as multiple metrics with indexed names
                    (e.g., `name/key_0`, `name/key_1`, ...).
                - 2D arrays are emitted as images.
                - 3D arrays are emitted as images if the last dimension has
                    1 or 3 channels, or as vector fields if the last dimension
                    has 2 channels.
             * Unsupported types are logged as errors.

        Args:
            name: Base name for the metrics.
            data: Dictionary data.
            step: Global step index.
            time: Optional global timestamp.
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
        """Flush and release resources (called when leaving a scope)."""
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
    """Process-wide event router.

    Thread-safe: ``attach`` / ``detach`` / ``shutdown`` may be called
    concurrently with ``emit``; handler invocations happen outside the
    internal lock so a slow handler cannot block concurrent registration.
    """

    def __init__(self) -> None:
        super().__init__()
        self.handlers: dict[str, Handler] = {}
        self.scopes: dict[str, set[str]] = defaultdict(set)
        self._lock = threading.RLock()

    def shutdown(self, timeout: float | None = None) -> None:
        """Close every attached handler and clear scope state.

        Every ``handler.close()`` is launched in its own daemon thread
        so slow handlers run concurrently. Each thread is then joined
        with ``timeout`` seconds; clean exits log a debug confirmation
        and timeouts log a warning.

        Args:
            timeout: Per-handler close budget in seconds. ``None`` waits
                indefinitely. On timeout the close thread is abandoned —
                Python cannot safely interrupt blocking I/O — and the
                handler may continue running in the background.
        """
        with self._lock:
            handlers_to_close = list(self.handlers.values())
            self.handlers.clear()
            self.scopes.clear()

        threads: list[tuple[threading.Thread, Handler]] = []
        for handler in handlers_to_close:
            t = threading.Thread(
                target=self._close_handler_safely,
                args=(handler,),
                daemon=True,
                name=f"goggles-close-{handler.name}",
            )
            t.start()
            threads.append((t, handler))

        log = logging.getLogger(__name__)
        for t, handler in threads:
            t.join(timeout=timeout)
            if t.is_alive():
                log.warning(
                    "Handler '%s' did not confirm close within %.1fs; "
                    "abandoning (thread continues as daemon).",
                    handler.name,
                    -1.0 if timeout is None else timeout,
                )
            else:
                log.debug("Handler '%s' close confirmed.", handler.name)

    @staticmethod
    def _close_handler_safely(handler: Handler) -> None:
        """Run ``handler.close()`` and log any exception it raises.

        Args:
            handler: Handler to close.
        """
        try:
            handler.close()
        except Exception:
            logging.getLogger(__name__).exception(
                "Handler '%s' raised in close()", handler.name
            )

    def attach(self, handlers: list[dict], scopes: list[str]) -> None:
        """Attach handler(s) under the given scopes.

        Args:
            handlers:
                The serialized handlers to attach to the scopes.
            scopes: The scopes under which to attach.

        """
        for handler_dict in handlers:
            handler_class = _get_handler_class(handler_dict["cls"])
            handler = handler_class.from_dict(handler_dict["data"])
            with self._lock:
                newly_added = handler.name not in self.handlers
                if newly_added:
                    self.handlers[handler.name] = handler
                for scope in scopes:
                    if scope not in self.scopes:
                        self.scopes[scope] = set()
                    self.scopes[scope].add(handler.name)
            if newly_added:
                # Call handler.open() outside the lock: it may do I/O
                # (e.g. open a wandb run) and must not block readers.
                handler.open()

    def detach(self, handler_name: str, scope: str) -> None:
        """Detach a handler from the given scope.

        Args:
            handler_name: The name of the handler to detach.
            scope: The scope from which to detach.

        Raises:
          ValueError: If the handler was not attached under the requested scope.

        """
        to_close: Handler | None = None
        with self._lock:
            if (
                scope not in self.scopes
                or handler_name not in self.scopes[scope]
            ):
                raise ValueError(
                    f"Handler '{handler_name}' not attached under scope "
                    f"'{scope}'"
                )
            self.scopes[scope].remove(handler_name)
            if not self.scopes[scope]:
                del self.scopes[scope]
            if not any(handler_name in self.scopes[s] for s in self.scopes):
                to_close = self.handlers.pop(handler_name, None)
        # Close outside the lock to avoid blocking emit/attach.
        if to_close is not None:
            try:
                to_close.close()
            except Exception:
                logging.getLogger(__name__).exception(
                    "Handler '%s' raised in close()", handler_name
                )

    def emit(self, event: dict | Event) -> None:
        """Emit an event to eligible handlers (errors isolated per handler).

        Snapshots the routing table under the lock, then invokes handlers
        outside of it so concurrent attach/detach calls are non-blocking.

        Args:
            event: The event (serialized) to emit, or an Event instance.

        Raises:
            TypeError: If `event` is neither a `dict` nor an `Event`.

        """
        if isinstance(event, dict):
            event = Event.from_dict(event)
        elif not isinstance(event, Event):
            raise TypeError(
                f"emit expects a dict or Event, got {type(event)!r}"
            )

        scope = event.scope

        # Snapshot under lock so attach/detach can't race with us.
        with self._lock:
            target_handlers: list[Handler] = []
            seen: set[str] = set()
            for s, names in self.scopes.items():
                if s != scope and not scope.startswith(s + "."):
                    continue
                for handler_name in names:
                    if handler_name in seen:
                        continue
                    seen.add(handler_name)
                    handler = self.handlers.get(handler_name)
                    if handler is not None:
                        target_handlers.append(handler)

        for handler in target_handlers:
            try:
                if handler.can_handle(event.kind):
                    handler.handle(event)
            except Exception:
                logging.getLogger(__name__).exception(
                    "Handler '%s' raised while dispatching event",
                    handler.name,
                )


def get_bus() -> Transport:
    """Return the process-wide transport singleton.

    The transport owns (host mode) or talks to (client mode) a single
    :class:`EventBus` per machine, routing events to attached handlers.

    Returns:
        The singleton transport.

    """
    global __impl_get_bus  # noqa: PLW0603
    if __impl_get_bus is None:
        # Importing here to avoid circular imports
        from ._core.routing import get_bus as _impl_get_bus  # noqa: PLC0415

        __impl_get_bus = cast(Callable[[], Any], _impl_get_bus)
    return __impl_get_bus()


def attach(handler: Handler, scopes: list[str] | None = None) -> None:
    """Attach a handler to the global transport under the specified scopes.

    Args:
        handler: The handler to attach.
        scopes: The scopes under which to attach.

    """
    if scopes is None:
        scopes = ["global"]
    bus = get_bus()
    bus.attach(handlers=[handler.to_dict()], scopes=scopes)


def configure(
    *,
    enable_console: bool = False,
    console_level: int = logging.INFO,
    console_path_style: Literal["absolute", "relative"] = "relative",
    project_root: str | os.PathLike[str] | None = None,
    scopes: list[str] | None = None,
) -> None:
    """One-call shortcut for the common "I just want a console logger" setup.

    Replaces the boilerplate

        gg.attach(gg.ConsoleHandler(level=logging.INFO), scopes=["global"])

    with

        gg.configure(enable_console=True)

    Power users should keep using ``attach`` / handler classes directly --
    this helper only covers the standard cases. Calling it with no arguments
    is a no-op so it is safe to invoke unconditionally during library
    initialization.

    Calling ``configure(enable_console=True, ...)`` while a console handler
    is already attached to the target ``scopes`` re-attaches a fresh handler
    with the new options (the old one is detached first), so the second call
    wins instead of being silently deduped by name.

    Args:
        enable_console: When True, attach a default ``ConsoleHandler``
            under ``scopes``. When False (the default), this argument
            does nothing.
        console_level: Minimum level for the auto-attached console
            handler. Ignored when ``enable_console`` is False.
        console_path_style: Whether the console handler prints absolute
            or project-relative source paths. Ignored when
            ``enable_console`` is False.
        project_root: Root path used to compute relative paths. Defaults
            to the current working directory.
        scopes: Scopes under which to attach the console handler.
            Defaults to ``["global"]``.
    """
    if not enable_console:
        return
    if scopes is None:
        scopes = ["global"]

    # Replace any existing console handler so a second `configure(...)` call
    # with new options actually takes effect (`attach()` dedupes by name and
    # would otherwise silently keep the first instance). Detach from the
    # target scopes unconditionally rather than reading bus state: with a
    # dedicated host the local bus is a client and never holds the handler
    # registry (it lives in the host process), and detach is a no-op when the
    # handler is not attached.
    for s in scopes:
        try:
            detach(ConsoleHandler.name, s)
        except ValueError:
            # Not attached here (in-process bus); nothing to detach.
            pass

    handler = ConsoleHandler(
        level=console_level,
        path_style=console_path_style,
        project_root=Path(project_root) if project_root is not None else None,
    )
    attach(handler, scopes=scopes)


def detach(handler_name: str, scope: str) -> None:
    """Detach a handler from the global transport under the specified scope.

    Args:
        handler_name: The name of the handler to detach.
        scope: The scope from which to detach.

    """
    bus = get_bus()
    bus.detach(handler_name, scope)


def finish(timeout: float | None = None) -> None:
    """Shutdown the global transport and close all handlers.

    Default behavior is to wait indefinitely so no queued events are
    silently dropped. Pass an explicit ``timeout`` to bound the wait —
    on timeout, the rest of the drain queue is discarded and any
    handler still inside ``close()`` is abandoned (the thread continues
    as a daemon). Both events are logged.

    Args:
        timeout: Optional bound in seconds applied to (a) draining queued
            events into handlers and (b) each handler's ``close()`` call.
            If ``None``, falls back to the ``GOGGLES_SHUTDOWN_TIMEOUT``
            env var; an unset env var or a non-positive value means no
            deadline.
    """
    bus = get_bus()
    if timeout is None:
        timeout = float(os.getenv("GOGGLES_SHUTDOWN_TIMEOUT", "0"))
    if timeout <= 0:
        timeout = None
    # Shut down this process's transport: flush queued events and, on a
    # client, disconnect from the dedicated host. The shared host is NOT
    # reaped here -- it owns the handlers (e.g. W&B runs) for every process,
    # so it winds down on its own once its last client disconnects. Reaping
    # it from whichever process calls finish() first is exactly what
    # fragments a multi-process app's runs.
    try:
        bus.shutdown(timeout=timeout)
    except Exception:
        logging.getLogger(__name__).exception(
            "Error while shutting down transport"
        )


def register_handler(handler_class: type) -> None:
    """Register a custom handler class for serialization/deserialization.

    Args:
        handler_class: The handler class to register.
            Must have a __name__ attribute.

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

    available_handlers = list(_HANDLER_REGISTRY.keys()) + [
        k for k in globals().keys() if k.endswith("Handler")
    ]
    raise KeyError(
        f"Handler class '{class_name}' not found. "
        f"Available handlers: {available_handlers}"
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
    "CRITICAL",
    "DEBUG",
    "ERROR",
    "INFO",
    "WARNING",
    "ConsoleHandler",
    "Event",
    "EventBus",
    "GogglesLogger",
    "GracefulShutdown",
    "Image",
    "Kind",
    "LocalStorageHandler",
    "Metrics",
    "PrettyConfig",
    "TextLogger",
    "Trajectories",
    "Vector",
    "VectorField",
    "Video",
    "WandBHandler",
    "attach",
    "detach",
    "filters",
    "finish",
    "freeze",
    "get_logger",
    "load_configuration",
    "register_handler",
    "save_configuration",
    "timeit",
    "trace_on_error",
]

# ---------------------------------------------------------------------------
# Import-time safety
# ---------------------------------------------------------------------------

# Attach a NullHandler so importing goggles never emits logs by default.

_logger = logging.getLogger(__name__)
if not any(isinstance(h, logging.NullHandler) for h in _logger.handlers):
    _logger.addHandler(logging.NullHandler())
