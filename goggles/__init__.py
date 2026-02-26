"""A simple Goggles implementation to test integrations without the full machinery."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
import time
from typing import (
    Any,
    Final,
    Literal,
    ParamSpec,
    Protocol,
    TypeVar,
    cast,
    overload,
    runtime_checkable,
)

try:
    from ._core.integrations.wandb import WandBHandler
except Exception:
    WandBHandler = None


from typing_extensions import Self


from . import filters
from ._core.decorators import timeit as _timeit
from ._core.decorators import trace_on_error as _trace_on_error
from ._core.integrations import ConsoleHandler
from .config import PrettyConfig, load_configuration, save_configuration
from .shutdown import GracefulShutdown
from .types import Event, Image, Kind, Metrics, Vector, VectorField, Video
from ._core.config_simple import CONSOLE, WANDB


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


def _make_text_logger(
    name: str | None,
    scope: str,
    **to_bind: Any,
) -> TextLogger:
    # Importing here to avoid circular imports
    from ._core.logger import CoreTextLogger  # noqa: PLC0415

    return CoreTextLogger(
        name=name, scope=scope, console_config=CONSOLE, **to_bind
    )


def _make_goggles_logger(
    name: str | None,
    scope: str,
    **to_bind: Any,
) -> GogglesLogger:
    # Importing here to avoid circular imports
    from ._core.logger import CoreGogglesLogger  # noqa: PLC0415

    return CoreGogglesLogger(
        name=name,
        scope=scope,
        console_config=CONSOLE,
        wandb_config=WANDB,
        **to_bind,
    )


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
        return _make_goggles_logger(name, scope, **to_bind)
    else:
        return _make_text_logger(name, scope, **to_bind)


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
    """Protocol for Handlers"""

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


def attach(handler: Handler, scopes: list[str] | None = None) -> None:
    """Change handler variables in simple mode to simulate attaching to the bus.

    scopes are ignored in simple mode, we consider all loggers to be in the same global scope.

    Args:
        handler: The handler to attach. Supported handlers are ConsoleHandler and WandBHandler.
        scopes: The scopes to attach to. Ignored in simple mode.

    Raises:
        NotImplementedError: If the handler type is not supported in simple mode.
    """
    # scopes ignored in simple mode; keep for API compatibility
    if isinstance(handler, ConsoleHandler):
        CONSOLE.name = handler.name
        CONSOLE.level = handler.level
        CONSOLE.path_style = cast(
            Literal["relative", "absolute"], handler.path_style
        )
        CONSOLE.project_root = handler.project_root
        CONSOLE.enabled = True
        return

    if WandBHandler and isinstance(handler, WandBHandler):
        WANDB.project = handler.project
        WANDB.entity = handler.entity
        WANDB.run_name = handler.run_name
        WANDB.group = handler.group
        WANDB.reinit = handler.reinit
        WANDB.config = dict(handler.config or {})
        WANDB.enabled = True
        return

    raise NotImplementedError(
        f"Attaching handler of type {type(handler).__name__} is not supported in simple mode."
    )


def detach(handler_name: str, scope: str | None = None) -> None:
    """Change handler variables in simple mode to simulate detaching from the bus.

    Args:
        handler_name: The name of the handler to detach. Supported values are:
            - "ConsoleHandler", "goggles.console", "console" for the console handler
            - "WandBHandler", "goggles.wandb", "wandb" for the WandB handler
        scope: The scope to detach from.

    Raises:
        NotImplementedError: If the handler type is not supported in simple mode.
    """
    # In simple mode, we only support detaching the ConsoleHandler and WandBHandler
    if handler_name in {"ConsoleHandler", "goggles.console", CONSOLE.name}:
        CONSOLE.enabled = False
        return

    if WandBHandler and handler_name in {
        "WandBHandler",
        "goggles.wandb",
        "wandb",
    }:
        WANDB.enabled = False
        return

    raise NotImplementedError(
        f"Detaching handler '{handler_name}' is not supported in simple mode."
    )


def finish(timeout: float | None = None) -> None:
    """No-op in simple mode since we don't have an actual bus or background threads.

    Args:
        timeout: Optional timeout for graceful shutdown. Ignored in simple mode.
    """
    from ._core.logger import ACTIVE

    start_time = time.monotonic()
    for logger in list(ACTIVE):
        try:
            logger.close(
                timeout=timeout - (time.monotonic() - start_time)
                if timeout is not None
                else None
            )
        except Exception as e:
            logging.getLogger(__name__).error(
                f"Error closing logger {logger}: {e}"
            )

    pass


# ---------------------------------------------------------------------------
# Logging Levels
# ---------------------------------------------------------------------------

INFO = logging.INFO
DEBUG = logging.DEBUG
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

__all__ = [
    "CRITICAL",
    "DEBUG",
    "ERROR",
    "INFO",
    "WARNING",
    "ConsoleHandler",
    "Event",
    "GogglesLogger",
    "GracefulShutdown",
    "Image",
    "Kind",
    "Metrics",
    "PrettyConfig",
    "TextLogger",
    "Vector",
    "VectorField",
    "Video",
    "WandBHandler",
    "attach",
    "detach",
    "filters",
    "get_logger",
    "load_configuration",
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
