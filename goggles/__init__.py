"""Goggles: Structured Logging and Experiment Tracking.

This package provides a stable public API for logging experiments, metrics,
and media in a consistent and composable way.

>>>    import goggles as gg
>>>
>>>    with gg.run("experiment_42"):
>>>        log = gg.get_logger("train", seed=0)
>>>        log.info("Training started.")
>>>        log.scalar("train/loss", 0.123, step=0)

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
    overload,
    runtime_checkable,
)
import logging

# Cache the implementations after first use to avoid repeated imports
__get_logger_text_impl: Optional[
    Callable[[Optional[str], dict[str, Any]], BoundLogger]
] = None
__get_logger_metrics_impl: Optional[
    Callable[[Optional[str], dict[str, Any]], GogglesLogger]
] = None
__current_run_impl: Optional[Callable[[], Optional[RunContext]]] = None
__configure_impl: Optional[Callable[..., None]] = None
__run_impl: Optional[Callable[..., AbstractContextManager[RunContext]]] = None
__impl_get_bus: Optional[Callable[[], EventBus]] = None
__impl_attach: Optional[Callable[[Handler, Literal["global", "run"]], None]] = None
__impl_detach: Optional[Callable[[Handler, Literal["global", "run"]], None]] = None

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
    global __get_logger_text_impl, __get_logger_metrics_impl

    if with_metrics:
        if __get_logger_metrics_impl is None:
            # separate factory that wraps the core bound logger and exposes `.metrics`
            from ._core.logger import get_logger_with_metrics as _get_logger_metrics

            __get_logger_metrics_impl = _get_logger_metrics
        # precise return type on the True branch
        return __get_logger_metrics_impl(name, bound)
    else:
        if __get_logger_text_impl is None:
            from ._core.logger import get_logger as _get_logger_text

            __get_logger_text_impl = _get_logger_text
        # precise return type on the False/default branch
        return __get_logger_text_impl(name, bound)


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
        created_at (str): ISO8601 UTC timestamp of when the run started.
        pid (int): Process ID that opened the run.
        host (str): Hostname of the machine where the run was created.
        python (str): Python version as `major.minor.micro`.
        metadata (Dict[str, Any]): Arbitrary user-provided metadata captured at
            run creation (experiment args, seeds, git SHA, etc.).
        wandb (Optional[Dict[str, Any]]): Optional W&B info (ids, URL, project).
            This field must be `None` if W&B is not enabled.

    """

    run_id: str
    run_name: Optional[str]
    log_dir: str
    created_at: str
    pid: int
    host: str
    python: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    wandb: Optional[Dict[str, Any]] = None


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
        """Return a new adapter with `fields` merged into persistent state."""

    def debug(self, msg: str, /, **extra: Any) -> None:
        """Log a DEBUG message with optional structured extras."""

    def info(self, msg: str, /, **extra: Any) -> None:
        """Log an INFO message with optional structured extras."""

    def warning(self, msg: str, /, **extra: Any) -> None:
        """Log a WARNING message with optional structured extras."""

    def error(self, msg: str, /, **extra: Any) -> None:
        """Log an ERROR message with optional structured extras."""

    def exception(self, msg: str, /, **extra: Any) -> None:
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
        """Emit a single scalar metric."""

    def image(
        self,
        name: str,
        image: bytes,
        *,
        step: Optional[int] = None,
        format: str = "png",
        **meta: Any,
    ) -> None:
        """Emit an image artifact (encoded bytes)."""

    def video(
        self,
        name: str,
        data: bytes,
        *,
        step: Optional[int] = None,
        fps: int = 30,
        **meta: Any,
    ) -> None:
        """Emit a video artifact (encoded bytes)."""


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
        capabilities (set[str]): Supported kinds, e.g. {'logs','metrics','artifacts'}.
        allowed_scopes (set[Literal['global','run']]): Valid scopes for attachment.

    """

    name: str
    capabilities: set[str]
    allowed_scopes: set[Literal["global", "run"]]

    def open(self, run: Optional[RunContext] = None) -> None:
        """Initialize the handler (called when entering a scope)."""

    def handle(self, event: Any) -> None:
        """Process a single event routed by the EventBus."""

    def close(self) -> None:
        """Flush and release resources (called when leaving a scope)."""


# ---------------------------------------------------------------------------
# EventBus and run management
# ---------------------------------------------------------------------------
class EventBus(Protocol):
    """Protocol for the process-wide event router."""

    def attach(self, handler: Handler, scope: Literal["global", "run"]) -> None:
        """Attach a handler under the given scope.

        Raises:
          ValueError: If the handler disallows the requested scope.

        """

    def emit(self, event: Any) -> None:
        """Emit an event to eligible handlers (errors isolated per handler)."""


def get_bus() -> EventBus:
    """Return the process-wide EventBus singleton.

    The EventBus owns handlers and routes events based on scope and kind.
    """
    global __impl_get_bus
    if __impl_get_bus is None:
        from ._core.eventbus import get_bus as _impl_get_bus

        __impl_get_bus = _impl_get_bus
    return __impl_get_bus()


def attach(handler: Handler, scope: Literal["global", "run"]) -> None:
    """Attach a handler to the EventBus under the given scope.

    Args:
      handler (Handler): Configured handler to attach.
      scope (Literal['global','run']): Attachment scope.

    Raises:
      ValueError: If the handler disallows the requested scope.

    """
    global __impl_attach
    if __impl_attach is None:
        from ._core.eventbus import attach as _impl_attach

        __impl_attach = _impl_attach
    __impl_attach(handler, scope)


def detach(handler: Handler, scope: Literal["global", "run"]) -> None:
    """Detach a handler from the EventBus under the given scope.

    Args:
        handler (Handler): Configured handler to detach.
        scope (Literal['global','run']): Detachment scope.

    Raises:
        ValueError: If the handler disallows the requested scope.
        ValueError: If the handler is not attached under the given scope.

    """
    global __impl_detach
    if __impl_detach is None:
        from ._core.eventbus import detach as _impl_detach

        __impl_detach = _impl_detach
    __impl_detach(handler, scope)


def current_run() -> Optional[RunContext]:
    """Return the currently active RunContext for this context if any.

    This function allows retrieving the active `RunContext` outside of log records.
    If no run is active, it returns `None`.

    Returns:
        Optional[RunContext]: The active run context, or None if no run is active.

    Examples:
        >>> with run("my_experiment") as ctx:
        ...     current = current_run()
        ...     assert current.run_id == ctx.run_id

    """
    global __current_run_impl
    if __current_run_impl is None:
        # Lazy import to avoid import-time side effects / cycles
        from ._core.run import get_active_run as _get_active_run

        __current_run_impl = _get_active_run

    return __current_run_impl()


def configure(**defaults: Any) -> None:
    """Override global defaults used by `run(...)`.

    This is an optional convenience to set process-wide defaults *before*
    `run(...)` is called (e.g., enabling JSONL by default). If a subsequent
    `run(...)` call specifies keyword arguments, those take precedence over
    these defaults.

    Recognized keys (all optional):
      - enable_console (bool): Enable console handler. Default: True.
      - enable_file (bool): Enable text file `events.log`. Default: True.
      - enable_jsonl (bool): Enable `events.jsonl`. Default: False.
      - enable_wandb (bool): Enable W&B integration. Default: False.
      - log_level (str): e.g., "INFO", "DEBUG". Default: "INFO".
      - propagate (bool): Set logger propagation. Default: False.
      - reset_root (bool): Remove existing root handlers at run start.
      - capture_warnings (bool): Route `warnings` to logging. Default: True.

    Raises:
        ValueError: If unknown keys are supplied or values have invalid types.

    Examples:
        >>> configure(enable_jsonl=True, log_level="DEBUG")
        >>> with run("test_run") as ctx:
        ...     assert ctx.enable_jsonl is True
        ...     assert ctx.log_level == "DEBUG"

    """
    global __configure_impl
    if __configure_impl is None:
        # Lazy import to avoid import-time side effects / cycles
        from ._core.run import _configure as _configure_impl_func

        __configure_impl = _configure_impl_func
    __configure_impl(**defaults)


def run(
    name: Optional[str] = None,
    log_dir: Optional[str] = None,
    *,
    enable_console: Optional[bool] = None,
    enable_file: Optional[bool] = None,
    enable_jsonl: Optional[bool] = None,
    enable_wandb: Optional[bool] = None,
    log_level: Optional[str] = None,
    propagate: Optional[bool] = None,
    reset_root: Optional[bool] = None,
    capture_warnings: Optional[bool] = None,
    enable_artifacts: Optional[bool] = None,
    artifact_name: Optional[str] = None,
    artifact_type: Optional[str] = None,
    **metadata: Any,
) -> AbstractContextManager[RunContext]:
    """Configure logging handlers for the current process and yield a `RunContext`.

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
        name (Optional[str]): Human-readable name; may be None.
        log_dir (Optional[str]): Target directory; default `./runs/<run_id>`.
        enable_console (Optional[bool]): Console handler toggle.
        enable_file (Optional[bool]): File handler toggle.
        enable_jsonl (Optional[bool]): JSONL handler toggle.
        enable_wandb (Optional[bool]): W&B integration toggle.
        log_level (Optional[str]): Log level ("INFO", "DEBUG", ...).
        propagate (Optional[bool]): Root logger propagation.
        reset_root (Optional[bool]): Remove existing root handlers first.
        capture_warnings (Optional[bool]): Route `warnings` to logging.
        enable_artifacts (Optional[bool]): Enable artifact logging.
        artifact_name (Optional[str]): Default artifact name.
        artifact_type (Optional[str]): Default artifact type.
        **metadata (Any): User-defined metadata persisted in `metadata.json`.

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
    global __run_impl
    if __run_impl is None:
        # Lazy import to avoid import-time side effects / cycles
        from ._core.run import _RunContextManager as _RunContextManagerImpl

        __run_impl = _RunContextManagerImpl

    return __run_impl(
        name=name,
        log_dir=log_dir,
        user_metadata=metadata,
        enable_console=enable_console,
        enable_file=enable_file,
        enable_jsonl=enable_jsonl,
        enable_wandb=enable_wandb,
        log_level=log_level,
        propagate=propagate,
        reset_root=reset_root,
        capture_warnings=capture_warnings,
        enable_artifacts=enable_artifacts,
        artifact_name=artifact_name,
        artifact_type=artifact_type,
    )


from .legacy import (
    scalar as _scalar_impl,
    image as _image_impl,
    video as _video_impl,
)

__all__ = [
    "RunContext",
    "BoundLogger",
    "configure",
    "run",
    "get_logger",
    "current_run",
    "get_bus",
    "attach",
    "detach",
    "scalar",
    "image",
    "video",
]

# ---------------------------------------------------------------------------
# Import-time safety
# ---------------------------------------------------------------------------

# Attach a NullHandler so importing goggles never emits logs by default.

_logger = logging.getLogger(__name__)
if not any(isinstance(h, logging.NullHandler) for h in _logger.handlers):
    _logger.addHandler(logging.NullHandler())

# ---------------------------------------------------------------------------
# Legacy metrics / media logging API
# ---------------------------------------------------------------------------


def scalar(tag, value, *, step=None, **kw):
    """Log a scalar metric."""
    _scalar_impl(tag, value, step=step, **kw)


def image(tag, data, *, step=None, **kw):
    """Log an image artifact."""
    _image_impl(tag, data, step=step, **kw)


def video(tag, data, *, step=None, **kw):
    """Log a video artifact."""
    _video_impl(tag, data, step=step, **kw)
