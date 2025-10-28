"""Goggles: Structured Logging and Experiment Tracking.

This package provides a stable public API for logging experiments, metrics,
and media in a consistent and composable way.

>>>    import goggles as gg
>>>
>>>    with gg.run("experiment_42"):
>>>        logger = gg.get_logger("train", seed=0)
>>>        logger.info("Training started.")
>>>        logger.scalar("train/loss", 0.123, step=0)

See Also:
    - README.md for detailed usage examples.
    - API docs for full reference of public interfaces.
    - Internal implementations live under `goggles/_core/`

"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Callable,
    List,
    Literal,
    Mapping,
    Optional,
    Protocol,
    overload,
    runtime_checkable,
)
import logging

# Cache the implementations after first use to avoid repeated imports
__impl_get_logger_text: Optional[
    Callable[[Optional[str], dict[str, Any]], BoundLogger]
] = None
__impl_get_logger_metrics: Optional[
    Callable[[Optional[str], dict[str, Any]], GogglesLogger]
] = None
__impl_get_bus: Optional[Callable[[], EventBus]] = None

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@overload
def get_logger(
    name: Optional[str] = None, /, *, scope: str = "global", **to_bind: Any
) -> BoundLogger: ...


@overload
def get_logger(
    name: Optional[str] = None,
    /,
    *,
    scope: str = "global",
    with_metrics: Literal[True],
    **to_bind: Any,
) -> GogglesLogger: ...


def get_logger(
    name: Optional[str] = None,
    /,
    *,
    scope: str = "global",
    with_metrics: bool = False,
    **to_bind: Any,
) -> BoundLogger | GogglesLogger:
    """Return a structured logger (text-only by default, metrics-enabled on opt-in).

    This is the primary entry point for obtaining Goggles' structured loggers.
    Depending on the active run and configuration, the returned adapter will
    inject structured context (e.g., `RunContext` info) and persistent fields
    into each emitted log record.

    Args:
        name (Optional[str]): Logger name. If None, the root logger is used.
        scope (str): The logging scope, e.g., "global" or "run".
        with_metrics (bool): If True, return a logger exposing `.metrics`.
        **to_bind (Any): Fields persisted and injected into every record.

    Returns:
        Union[BoundLogger, GogglesLogger]: A text-only `BoundLogger` by default,
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
    global __impl_get_logger_text, __impl_get_logger_metrics

    if with_metrics:
        if __impl_get_logger_metrics is None:
            from ._core.logger import get_logger_with_metrics as _get_logger_metrics

            __impl_get_logger_metrics = _get_logger_metrics
        return __impl_get_logger_metrics(name, scope, to_bind)
    else:
        if __impl_get_logger_text is None:
            from ._core.logger import get_logger as _get_logger_text

            __impl_get_logger_text = _get_logger_text
        return __impl_get_logger_text(name, scope, to_bind)


@runtime_checkable
class BoundLogger(Protocol):
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

    def bind(self, **fields: Any) -> "BoundLogger":
        """Return a new adapter with `fields` merged into persistent state.

        Args:
            **fields (Any): Key-value pairs to bind persistently.

        """

    def log(self, severity: int, msg: str, /, **extra: Any) -> None:
        """Log message at the given severity with optional structured extras.

        Args:
            severity (int): Log level (e.g., logging.INFO).
            msg (str): The log message.
            **extra (Any):
                Additional structured key-value pairs for this record.

        """
        if severity >= logging.CRITICAL:
            self.critical(msg, **extra)
        elif severity >= logging.ERROR:
            self.error(msg, **extra)
        elif severity >= logging.WARNING:
            self.warning(msg, **extra)
        elif severity >= logging.INFO:
            self.info(msg, **extra)
        elif severity >= logging.DEBUG:
            self.debug(msg, **extra)
        else:
            # Below DEBUG level; no-op by default.
            pass

    def debug(self, msg: str, /, **extra: Any) -> None:
        """Log a DEBUG message with optional structured extras.

        Args:
            msg (str): The log message.
            **extra (Any):
                Additional structured key-value pairs for this record.

        """

    def info(self, msg: str, /, **extra: Any) -> None:
        """Log an INFO message with optional structured extras.

        Args:
            msg (str): The log message.
            **extra (Any):
                Additional structured key-value pairs for this record.

        """

    def warning(self, msg: str, /, **extra: Any) -> None:
        """Log a WARNING message with optional structured extras.

        Args:
            msg (str): The log message.
            **extra (Any):
                Additional structured key-value pairs for this record.

        """

    def error(self, msg: str, /, **extra: Any) -> None:
        """Log an ERROR message with optional structured extras.

        Args:
            msg (str): The log message.
            **extra (Any):
                Additional structured key-value pairs for this record.

        """

    def critical(self, msg: str, /, **extra: Any) -> None:
        """Log an ERROR with current exception info attached."""


@runtime_checkable
class MetricsEmitter(Protocol):
    """Protocol for metrics and media emission."""

    def push(
        self, metrics: Mapping[str, float], *, step: Optional[int] = None, **meta: Any
    ) -> None:
        """Emit a batch of scalar metrics.

        Args:
            metrics (Mapping[str, float]): Name→value pairs.
            step (Optional[int]): Optional global step index.
            **meta (Any): Additional routing metadata (e.g., split="train").

        """

    def scalar(
        self, name: str, value: float, *, step: Optional[int] = None, **meta: Any
    ) -> None:
        """Emit a single scalar metric.

        Args:
            name (str): Metric name.
            value (float): Metric value.
            step (Optional[int]): Optional global step index.
            **meta (Any): Additional routing metadata (e.g., split="train").

        """

    def image(
        self,
        name: str,
        image: bytes,
        *,
        step: Optional[int] = None,
        format: str = "png",
        **meta: Any,
    ) -> None:
        """Emit an image artifact (encoded bytes).

        Args:
            name (str): Artifact name.
            image (bytes): Encoded image bytes.
            step (Optional[int]): Optional global step index.
            format (str): Image format, e.g., "png", "jpeg".
            **meta (Any): Additional routing metadata.

        """

    def video(
        self,
        name: str,
        data: bytes,
        *,
        step: Optional[int] = None,
        fps: int = 30,
        **meta: Any,
    ) -> None:
        """Emit a video artifact (encoded bytes).

        Args:
            name (str): Artifact name.
            data (bytes): Encoded video bytes.
            step (Optional[int]): Optional global step index.
            fps (int): Frames per second.
            **meta (Any): Additional routing metadata.

        """


@runtime_checkable
class GogglesLogger(BoundLogger, MetricsEmitter, Protocol):
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

    def bind(self, **fields: Any) -> "GogglesLogger":
        """Return a new facade with `fields` merged into persistent state."""


@runtime_checkable
class Handler(Protocol):
    """Protocol for EventBus handlers.

    Attributes:
        name (str): Stable handler identifier for diagnostics.
        capabilities (set[str]):
            Supported kinds, e.g. {'logs','metrics','artifacts'}.

    """

    name: str

    def open(self, run: Optional[RunContext] = None) -> None:
        """Initialize the handler (called when entering a scope).

        Args:
            run (Optional[RunContext]): The active run context if any.

        """

    def close(self) -> None:
        """Flush and release resources (called when leaving a scope).

        Args:
            run (Optional[RunContext]): The active run context if any.

        """


@runtime_checkable
class TextHandler(Handler, Protocol):
    """Protocol for text log handlers."""

    def log(self, message: str, level: str = "info", **meta: Any) -> None:
        """Log a message with the given level and metadata.

        Args:
            message (str): The log message.
            level (str): The log level (e.g., "info", "error").
            **meta (Any): Additional metadata to include in the log.

        """


@runtime_checkable
class MetricsHandler(Handler, Protocol):
    """Protocol for metrics log handlers."""

    def emit(self, metrics: Mapping[str, Any], **meta: Any) -> None:
        """Emit a batch of scalar metrics.

        Args:
            metrics (Mapping[str, float]): Name→value pairs.
            **meta (Any): Additional routing metadata (e.g., split="train").

        """


@runtime_checkable
class ArtifactsHandler(Handler, Protocol):
    """Protocol for artifacts log handlers."""

    def upload(self, name: str, artifact: Any, **meta: Any) -> None:
        """Upload an artifact with the given name and metadata.

        Args:
            name (str): Artifact name.
            artifact (Any): Artifact data.
            **meta (Any): Additional metadata for the artifact.

        """


# ---------------------------------------------------------------------------
# EventBus and run management
# ---------------------------------------------------------------------------
class EventBus(Protocol):
    """Protocol for the process-wide event router."""

    def attach(self, handlers: List[Handler], scopes: List[str]) -> None:
        """Attach a handler under the given scope.

        Args:
            handlers (List[Handler]): The handlers to attach to the scopes.
            scopes (List[str]): The scopes under which to attach.

        Raises:
          ValueError: If the handler disallows the requested scope.

        """

    def detach(self, handler_name: str, scope: str) -> None:
        """Detach a handler from the given scope.

        Args:
            handler_name (str): The name of the handler to detach.
            scope (str): The scope from which to detach.

        Raises:
          ValueError: If the handler was not attached under the requested scope.

        """

    def emit(self, event: Any) -> None:
        """Emit an event to eligible handlers (errors isolated per handler).

        Args:
            event (Any): The event to emit.

        """


def get_bus() -> EventBus:
    """Return the process-wide EventBus singleton.

    The EventBus owns handlers and routes events based on scope and kind.
    """
    global __impl_get_bus
    if __impl_get_bus is None:
        from ._core.eventbus import get_bus as _impl_get_bus

        __impl_get_bus = _impl_get_bus
    return __impl_get_bus()


def attach(handler: Handler, scopes: List[str]) -> None:
    """Attach a handler to the global EventBus under the specified scopes.

    Args:
        handler (Handler): The handler to attach.
        scopes (List[str]): The scopes under which to attach.

    Raises:
        ValueError: If the handler disallows the requested scope.

    """
    bus = get_bus()
    bus.attach([handler], scopes)


def detach(handler_name: str, scope: str) -> None:
    """Detach a handler from the global EventBus under the specified scope.

    Args:
        handler_name (str): The name of the handler to detach.
        scope (str): The scope from which to detach.

    Raises:
        ValueError: If the handler was not attached under the requested scope.

    """
    bus = get_bus()
    bus.detach(handler_name, scope)


__all__ = [
    "BoundLogger",
    "GogglesLogger",
    "get_logger",
    "attach",
    "detach",
]

# ---------------------------------------------------------------------------
# Import-time safety
# ---------------------------------------------------------------------------

# Attach a NullHandler so importing goggles never emits logs by default.

_logger = logging.getLogger(__name__)
if not any(isinstance(h, logging.NullHandler) for h in _logger.handlers):
    _logger.addHandler(logging.NullHandler())
