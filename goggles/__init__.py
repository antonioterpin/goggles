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

from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Callable,
    Dict,
    Literal,
    Mapping,
    Optional,
    Protocol,
    TypedDict,
    Unpack,
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
__impl_current_run: Optional[Callable[[], Optional[RunContext]]] = None
__impl_configure: Optional[Callable[..., None]] = None
__impl_run_start: Optional[Callable[[], None]] = None
__impl_run_stop: Optional[Callable[[], None]] = None
__impl_get_bus: Optional[Callable[[], EventBus]] = None

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@overload
def get_logger(name: Optional[str] = None, /, **bound: Any) -> BoundLogger: ...


@overload
def get_logger(
    name: Optional[str] = None, /, *, with_metrics: Literal[True], **bound: Any
) -> GogglesLogger: ...


def get_logger(
    name: Optional[str] = None, /, *, with_metrics: bool = False, **bound: Any
) -> BoundLogger | GogglesLogger:
    """Return a structured logger (text-only by default, metrics-enabled on opt-in).

    This is the primary entry point for obtaining Goggles' structured loggers.
    Depending on the active run and configuration, the returned adapter will
    inject structured context (e.g., `RunContext` info) and persistent fields
    into each emitted log record.

    Args:
        name (Optional[str]): Logger name. If None, the root logger is used.
        with_metrics (bool): If True, return a logger exposing `.metrics`.
        **bound (Any): Fields persisted and injected into every record.

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
        >>> tlog.metrics.scalar("loss", 0.42, step=1)

    """
    global __impl_get_logger_text, __impl_get_logger_metrics

    if with_metrics:
        if __impl_get_logger_metrics is None:
            from ._core.logger import get_logger_with_metrics as _get_logger_metrics

            __impl_get_logger_metrics = _get_logger_metrics
        return __impl_get_logger_metrics(name, bound)
    else:
        if __impl_get_logger_text is None:
            from ._core.logger import get_logger as _get_logger_text

            __impl_get_logger_text = _get_logger_text
        return __impl_get_logger_text(name, bound)


@dataclass(frozen=True)
class RunContext:
    """Immutable metadata describing a single logging run.

    This object is yielded by the `run(...)` context manager and
    injected into each log record emitted during the run.

    Attributes:
        run_id (str): Unique run identifier (UUID4 as canonical string).
        run_name (Optional[str]): Human-friendly name shown in UIs; may be None.
        log_dir (str): Absolute or relative path to the run directory containing
            `events.log`, optional `events.jsonl`, and `metadata.json`.
        created_at (str): Timestamp of when the run started.
        pid (int): Process ID that opened the run.
        host (str): Hostname of the machine where the run was created.
        python (str): Python version as `major.minor.micro`.
        metadata (Dict[str, Any]): Arbitrary user-provided metadata captured at
            run creation (experiment args, seeds, git SHA, etc.).

    """

    run_id: str
    run_name: Optional[str]
    log_dir: str
    created_at: str
    pid: int
    host: str
    python: str
    metadata: Dict[str, Any] = field(default_factory=dict)


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
class GogglesLogger(BoundLogger, Protocol):
    """Protocol for Goggles loggers with metrics support.

    Composite logger combining text logging with a metrics facet.

    Examples:
        >>> # Text-only
        >>> log = get_logger("eval", dataset="CIFAR10")
        >>> log.info("starting")
        >>>
        >>> # Explicit metrics surface
        >>> tlog = get_logger("train", with_metrics=True, seed=0)
        >>> tlog.metrics.scalar("loss", 0.42, step=1)
        >>> tlog.info("Training step completed")
        ...   # Both log records include any persistent bound fields.
        ...   # The second record also includes run_id="exp42".

    """

    metrics: MetricsEmitter

    def bind(self, **fields: Any) -> "GogglesLogger":
        """Return a new facade with `fields` merged into persistent state."""


@runtime_checkable
class Handler(Protocol):
    """Protocol for EventBus handlers.

    Attributes:
        name (str): Stable handler identifier for diagnostics.
        capabilities (set[str]):
            Supported kinds, e.g. {'logs','metrics','artifacts'}.
        allowed_scopes (set[Literal['global','run']]): Valid scopes for attachment.

    """

    name: str
    capabilities: set[str]
    allowed_scopes: set[Literal["global", "run"]]

    def open(self, run: Optional[RunContext] = None) -> None:
        """Initialize the handler (called when entering a scope).

        Args:
            run (Optional[RunContext]): The active run context if any.

        """

    def handle(self, event: Any) -> None:
        """Process a single event routed by the EventBus.

        Args:
            event (Any): The event to process.

        """

    def close(self) -> None:
        """Flush and release resources (called when leaving a scope).

        Args:
            run (Optional[RunContext]): The active run context if any.

        """


# ---------------------------------------------------------------------------
# EventBus and run management
# ---------------------------------------------------------------------------
class EventBus(Protocol):
    """Protocol for the process-wide event router."""

    def attach(self, handler: Handler, scope: Literal["global", "run"]) -> None:
        """Attach a handler under the given scope.

        Args:
            handler (Handler): The handler to attach.
            scope (Literal["global", "run"]): The scope under which to attach.

        Raises:
          ValueError: If the handler disallows the requested scope.

        """

    def detach(self, handler_name: str, scope: Literal["global", "run"]) -> None:
        """Detach a handler from the given scope.

        Args:
            handler_name (str): The name of the handler to detach.
            scope (Literal["global", "run"]): The scope from which to detach.

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


def current_run() -> Optional[RunContext]:
    """Return the currently active RunContext for this context if any.

    This function allows retrieving the active `RunContext`
    outside of log records. If no run is active, it returns `None`.

    Returns:
        Optional[RunContext]:
            The active run context, or None if no run is active.

    Examples:
        >>> with run("my_experiment") as ctx:
        ...     current = current_run()
        ...     assert current.run_id == ctx.run_id

    """
    global __impl_current_run
    if __impl_current_run is None:
        from ._core.run import get_active_run as _get_active_run

        __impl_current_run = _get_active_run
    return __impl_current_run()


class RunKwargs(TypedDict, total=False):
    name: Optional[str]
    log_dir: Optional[str]
    log_level: Optional[str]
    handlers: list[Handler] | None
    metadata: Mapping[str, Any] | None = None


def configure(**kwargs: Unpack[RunKwargs]) -> None:
    """Override global defaults used by `run(...)`.

    This is an optional convenience to set process-wide defaults *before*
    `run(...)` is called (e.g., enabling JSONL by default). If a subsequent
    `run(...)` call specifies keyword arguments, those take precedence over
    these defaults.

    Args:
        **kwargs: Recognized keys (all optional) are in RunKwargs.
        **metadata (Any): Additional user metadata to include in runs.

    Raises:
        ValueError: If unknown keys are supplied or values have invalid types.

    Examples:
        >>> configure(enable_jsonl=True, log_level="DEBUG")
        >>> with run("test_run") as ctx:
        ...     assert ctx.enable_jsonl is True
        ...     assert ctx.log_level == "DEBUG"

    """
    global __impl_configure
    if __impl_configure is None:
        from ._core.run import _configure as _configure_impl_func

        __impl_configure = _configure_impl_func
    __impl_configure(**kwargs)


def start_run(**kwargs: Unpack[RunKwargs]) -> RunContext:
    """Start a logging run.

    This function configures logging handlers for the current process
    according to the specified configuration.
    The run remains active until `stop_run()` is called.

    Args:
        **kwargs: Recognized keys (all optional) are in RunKwargs.

    Returns:
        RunContext: The context manager for the active run.

    Raises:
        RuntimeError: If a run is already active in this process.
        OSError: If directory creation or file opening fails.
        ValueError: If `log_level` is invalid or incompatible options are set.

    """
    global __impl_run_start
    if __impl_run_start is None:
        from ._core.run import _start_run as _start_run_impl

        __impl_run_start = _start_run_impl

    __impl_run_start(**kwargs)


def stop_run() -> None:
    """Stop the currently active logging run.

    It flushes and closes all logging handlers associated with
    the active run and releases resources.

    Raises:
        RuntimeError: If no run is currently active.

    """
    global __impl_run_stop
    if __impl_run_stop is None:
        from ._core.run import _stop_run as _stop_run_impl

        __impl_run_stop = _stop_run_impl

    __impl_run_stop()


def run(**kwargs: Unpack[RunKwargs]) -> AbstractContextManager[RunContext]:
    """Configure logging and yield a `RunContext`.

    This is the primary entry point to start a logging run. It sets up
    logging handlers according to the specified configuration, creates a
    `RunContext`, and yields it within a context manager. All log records
    emitted while inside the context will include structured metadata from
    the `RunContext`.

    Behavior:
        - Exactly-once configuration: if a run is already active, raise
            `RuntimeError` rather than silently stacking handlers.
        - Creates the specified log directory if it does not exist.
        - Persists `metadata.json` with user-provided metadata.
        - Supports optional integrations (W&B, artifact logging).
        - Defaults passed as keyword arguments override any global defaults
            set via `configure(...)`.

    Args:
        **kwargs: Recognized keys (all optional) are in RunKwargs.

    Returns:
        AbstractContextManager[RunContext]: A context manager yielding `RunContext`.

    Raises:
        RuntimeError: If a run is already active in this process.
        OSError: If directory creation or file opening fails.
        ValueError: If `log_level` is invalid or incompatible options are set.

    Examples:
        >>> # Application entrypoint
        >>> with run("exp42", enable_jsonl=True) as ctx:
        ...     log = get_logger("train", seed=0)
        ...     log.info("start", step=0)

    """
    global __impl_run
    if __impl_run is None:
        from ._core.run import _RunContextManager as _RunContextManagerImpl

        __impl_run = _RunContextManagerImpl

    return __impl_run(**kwargs)


__all__ = [
    "RunContext",
    "BoundLogger",
    "configure",
    "start_run",
    "stop_run",
    "run",
    "get_logger",
    "current_run",
    "get_bus",
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
