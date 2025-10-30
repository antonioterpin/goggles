"""Goggles: Structured logging and experiment tracking.

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

import atexit
from collections import defaultdict
from typing import (
    Any,
    Callable,
    Callable,
    ClassVar,
    FrozenSet,
    List,
    Literal,
    Optional,
    Protocol,
    Dict,
    Set,
    Union,
    overload,
    runtime_checkable,
)
from typing_extensions import Self
import logging
import os

from .types import Kind, Event, Video, Image, Vector, Metrics
from ._core.integrations import *
from .decorators import timeit, trace_on_error
from .shutdown import GracefulShutdown
from .config import load_configuration, save_configuration

# Goggles port for bus communication
GOGGLES_PORT = os.getenv("GOGGLES_PORT", "2401")
GOGGLES_HOST = os.getenv("GOGGLES_HOST", "localhost")

# Cache the implementations after first use to avoid repeated imports
__impl_get_logger_text: Optional[
    Callable[[Optional[str], dict[str, Any]], TextLogger]
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
    name: Optional[str] = None,
    /,
    *,
    scope: str = "global",
    **to_bind: Any,
) -> TextLogger: ...


@overload
def get_logger(
    name: Optional[str] = None,
    /,
    *,
    with_metrics: Literal[True],
    scope: str = "global",
    **to_bind: Any,
) -> GogglesLogger: ...


@overload
def get_logger(name: Optional[str] = None, /, **to_bind: Any) -> TextLogger: ...


def get_logger(
    name: Optional[str] = None,
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
        name (Optional[str]): Logger name. If None, the root logger is used.
        with_metrics (bool): If True, return a logger exposing `.metrics`.
        scope (str): The logging scope, e.g., "global" or "run".
        **to_bind (Any): Fields persisted and injected into every record.

    Returns:
        Union[TextLogger, GogglesLogger]: A text-only `TextLogger` by default,
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
            from ._core.logger import CoreGogglesLogger

            __impl_get_logger_metrics = lambda n, s, tb: CoreGogglesLogger(
                name=n, scope=s, to_bind=tb
            )
        return __impl_get_logger_metrics(name, scope, to_bind)
    else:
        if __impl_get_logger_text is None:
            from ._core.logger import CoreTextLogger

            __impl_get_logger_text = lambda n, s, tb: CoreTextLogger(
                name=n, scope=s, to_bind=tb
            )
        return __impl_get_logger_text(name, scope, to_bind)


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
        """Return a new adapter with `fields` merged into persistent state.

        Args:
            scope (str): The binding scope, e.g., "global" or "run".
            **fields (Any): Key-value pairs to bind persistently.


        Returns:
            Self: A new `TextLogger` instance
                with updated bound fields and scope.

        """
        ...

    def log(
        self,
        severity: int,
        msg: str,
        /,
        *,
        step: Optional[int] = None,
        time: Optional[float] = None,
        **extra: Any,
    ) -> None:
        """Log message at the given severity with optional structured extras.

        Args:
            severity (int): Numeric log level (e.g., logging.INFO).
            msg (str): The log message.
            step (Optional[int]): The step number.
            time (Optional[float]): The timestamp.
            **extra (Any):
                Additional structured key-value pairs for this record.

        """
        if severity >= logging.CRITICAL:
            self.critical(msg, step=step, time=time, **extra)
        elif severity >= logging.ERROR:
            self.error(msg, step=step, time=time, **extra)
        elif severity >= logging.WARNING:
            self.warning(msg, step=step, time=time, **extra)
        elif severity >= logging.INFO:
            self.info(msg, step=step, time=time, **extra)
        elif severity >= logging.DEBUG:
            self.debug(msg, step=step, time=time, **extra)
        else:
            # Below DEBUG level; no-op by default.
            pass

    def debug(
        self,
        msg: str,
        /,
        *,
        step: Optional[int] = None,
        time: Optional[float] = None,
        **extra: Any,
    ) -> None:
        """Log a DEBUG message with optional structured extras.

        Args:
            msg (str): The log message.
            step (Optional[int]): The step number.
            time (Optional[float]): The timestamp.
            **extra (Any):
                Additional structured key-value pairs for this record.

        """

    def info(
        self,
        msg: str,
        /,
        *,
        step: Optional[int] = None,
        time: Optional[float] = None,
        **extra: Any,
    ) -> None:
        """Log an INFO message with optional structured extras.

        Args:
            msg (str): The log message.
            step (Optional[int]): The step number.
            time (Optional[float]): The timestamp.
            **extra (Any):
                Additional structured key-value pairs for this record.

        """

    def warning(
        self,
        msg: str,
        /,
        *,
        step: Optional[int] = None,
        time: Optional[float] = None,
        **extra: Any,
    ) -> None:
        """Log a WARNING message with optional structured extras.

        Args:
            msg (str): The log message.
            step (Optional[int]): The step number.
            time (Optional[float]): The timestamp.
            **extra (Any):
                Additional structured key-value pairs for this record.

        """

    def error(
        self,
        msg: str,
        /,
        *,
        step: Optional[int] = None,
        time: Optional[float] = None,
        **extra: Any,
    ) -> None:
        """Log an ERROR message with optional structured extras.

        Args:
            msg (str): The log message.
            step (Optional[int]): The step number.
            time (Optional[float]): The timestamp.
            **extra (Any):
                Additional structured key-value pairs for this record.

        """

    def critical(
        self,
        msg: str,
        /,
        *,
        step: Optional[int] = None,
        time: Optional[float] = None,
        **extra: Any,
    ) -> None:
        """Log a CRITICAL message with current exception info attached.

        Args:
            msg (str): The log message.
            step (Optional[int]): The step number.
            time (Optional[float]): The timestamp.
            **extra (Any):
                Additional structured key-value pairs for this record.

        """


@runtime_checkable
class DataLogger(Protocol):
    """Protocol for logging metrics, media, artifacts, and analytics data."""

    def push(
        self,
        metrics: Metrics,
        *,
        step: Optional[int] = None,
        time: Optional[float] = None,
        **extra: Dict[str, Any],
    ) -> None:
        """Emit a batch of scalar metrics.

        Args:
            metrics (Metrics): (Name,value) pairs.
            step (Optional[int]): Optional global step index.
            time (Optional[float]): Optional global timestamp.
            **extra (Dict[str, Any]):
                Additional routing metadata (e.g., split="train").

        """

    def scalar(
        self,
        name: str,
        value: float | int,
        *,
        step: Optional[int] = None,
        time: Optional[float] = None,
        **extra: Dict[str, Any],
    ) -> None:
        """Emit a single scalar metric.

        Args:
            name (str): Metric name.
            value (float|int): Metric value.
            step (Optional[int]): Optional global step index.
            time (Optional[float]): Optional global timestamp.
            **extra (Dict[str, Any]):
                Additional routing metadata (e.g., split="train").

        """

    def image(
        self,
        image: Image,
        *,
        name: Optional[str] = None,
        format: str = "png",
        step: Optional[int] = None,
        time: Optional[float] = None,
        **extra: Dict[str, Any],
    ) -> None:
        """Emit an image artifact (encoded bytes).

        Args:
            name (str): Artifact name.
            image (Image): Image.
            format (str): Image format, e.g., "png", "jpeg".
            step (Optional[int]): Optional global step index.
            time (Optional[float]): Optional global timestamp.
            **extra: Dict[str, Any]: Additional routing metadata.

        """

    def video(
        self,
        video: Video,
        *,
        name: Optional[str] = None,
        fps: int = 30,
        format: str = "gif",
        step: Optional[int] = None,
        time: Optional[float] = None,
        **extra: Dict[str, Any],
    ) -> None:
        """Emit a video artifact (encoded bytes).

        Args:
            video (Video): Video.
            name (Optional[str]): Artifact name.
            fps (int): Frames per second.
            format (str): Video format, e.g., "gif", "mp4".
            step (Optional[int]): Optional global step index.
            time (Optional[float]): Optional global timestamp.
            **extra (Dict[str, Any]): Additional routing metadata.

        """

    def artifact(
        self,
        name: str,
        data: bytes,
        *,
        format: str = "bin",
        step: Optional[int] = None,
        time: Optional[float] = None,
        **extra: Dict[str, Any],
    ) -> None:
        """Emit a generic artifact (encoded bytes).

        Args:
            name (str): Artifact name.
            data (bytes): Artifact data.
            format (str): Artifact format, e.g., "txt", "bin".
            step (Optional[int]): Optional global step index.
            time (Optional[float]): Optional global timestamp.
            **extra (Dict[str, Any]): Additional routing metadata.

        """

    def vector_field(
        self,
        name: str,
        vector_field: VectorField,
        *,
        step: Optional[int] = None,
        time: Optional[float] = None,
        **extra: Dict[str, Any],
    ) -> None:
        """Emit a vector field artifact.

        Args:
            name (str): Artifact name.
            vector_field (VectorField): Vector field data.
            step (Optional[int]): Optional global step index.
            time (Optional[float]): Optional global timestamp.
            **extra (Dict[str, Any]): Additional routing metadata.

        """

    def histogram(
        self,
        name: str,
        histogram: Vector,
        *,
        step: Optional[int] = None,
        time: Optional[float] = None,
        **extra: Dict[str, Any],
    ) -> None:
        """Emit a histogram artifact.

        Args:
            name (str): Artifact name.
            histogram (Vector): Histogram data.
            step (Optional[int]): Optional global step index.
            time (Optional[float]): Optional global timestamp.
            **extra (Dict[str, Any]): Additional routing metadata.

        """

    def trajectory(
        self,
        name: str,
        trajectory: List[Union[Vector, Image, VectorField]],
        *,
        step: Optional[int] = None,
        time: Optional[float] = None,
        **extra: Dict[str, Any],
    ) -> None:
        """Emit a trajectory artifact.

        Args:
            name (str): Artifact name.
            trajectory (Vector): Trajectory data.
            step (Optional[int]): Optional global step index.
            time (Optional[float]): Optional global timestamp.
            **extra (Dict[str, Any]): Additional routing metadata.

        """


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
        name (str): Stable handler identifier for diagnostics.
        capabilities (FrozenSet[Kind]):
            Supported kinds, e.g. {'logs','metrics','artifacts', ...}.

    """

    name: str
    capabilities: ClassVar[FrozenSet[Kind]]

    def can_handle(self, kind: Kind) -> bool:
        """Return whether this handler can process events of the given kind.

        Args:
            kind (Kind):
                The kind of event ("log", "metric", "image", "artifact").

        Returns:
            bool: True if the handler can process the event kind,
                False otherwise.

        """
        ...

    def handle(self, event: Event) -> None:
        """Handle an emitted event.

        Args:
            event (Event): The event to handle.

        """

    def open(self) -> None:
        """Initialize the handler (called when entering a scope)."""

    def close(self) -> None:
        """Flush and release resources (called when leaving a scope).

        Args:
            run (Optional[RunContext]): The active run context if any.

        """

    def to_dict(self) -> Dict:
        """Serialize the handler.

        This method is needed during attachment. Will be called before binding.

        Returns:
            (dict) A dictionary that allows to instantiate the Handler.
                Must contain:
                    - "cls": The handler class name.
                    - "data": The handler data to be used in from_dict.

        """
        ...

    @classmethod
    def from_dict(cls, serialized: Dict) -> Self:
        """De-serialize the handler.

        Args:
            serialized (Dict): Serialized handler with handler.to_dict

        Returns:
            Self: The Handler instance.

        """
        ...


# ---------------------------------------------------------------------------
# EventBus and run management
# ---------------------------------------------------------------------------
class EventBus:
    """Protocol for the process-wide event router."""

    handlers: Dict[str, Handler]
    scopes: Dict[str, Set[str]]

    def __init__(self):
        super().__init__()
        self.handlers: Dict[str, Handler] = {}
        self.scopes: Dict[str, Set[str]] = defaultdict(set)

        atexit.register(self._shutdown)

    def _shutdown(self) -> None:
        """Shutdown the EventBus and close all handlers."""
        copy_map = {
            scope: handlers_names.copy()
            for scope, handlers_names in self.scopes.items()
        }
        for scope, handlers_names in copy_map.items():
            for handler_name in handlers_names:
                self.detach(handler_name, scope)

    def attach(self, handlers: List[dict], scopes: List[str]) -> None:
        """Attach a handler under the given scope.

        Args:
            handlers (List[dict]):
                The serialized handlers to attach to the scopes.
            scopes (List[str]): The scopes under which to attach.

        """
        for handler_dict in handlers:
            handler = globals()[handler_dict["cls"]].from_dict(handler_dict["data"])
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
            handler_name (str): The name of the handler to detach.
            scope (str): The scope from which to detach.

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

    def emit(self, event: Dict | Event) -> None:
        """Emit an event to eligible handlers (errors isolated per handler).

        Args:
            event (dict | Event): The event (serialized) to emit, or an Event instance.

        """
        if isinstance(event, dict):
            event = Event.from_dict(event)
        elif not isinstance(event, Event):
            raise TypeError(f"emit expects a dict or Event, got {type(event)!r}")

        if event.scope not in self.scopes:
            return

        for handler_name in self.scopes[event.scope]:
            handler = self.handlers.get(handler_name)
            if handler and handler.can_handle(event.kind):
                handler.handle(event)


def get_bus() -> EventBus:
    """Return the process-wide EventBus singleton.

    The EventBus owns handlers and routes events based on scope and kind.
    """
    global __impl_get_bus
    if __impl_get_bus is None:
        from ._core.routing import get_bus as _impl_get_bus

        __impl_get_bus = _impl_get_bus
    return __impl_get_bus()


def attach(handler: Handler, scopes: List[str] = ["global"]) -> None:
    """Attach a handler to the global EventBus under the specified scopes.

    Args:
        handler (Handler): The handler to attach.
        scopes (List[str]): The scopes under which to attach.

    Raises:
        ValueError: If the handler disallows the requested scope.

    """
    bus = get_bus()
    bus.attach([handler.to_dict()], scopes)


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
    "TextLogger",
    "GogglesLogger",
    "get_logger",
    "attach",
    "detach",
    "load_configuration",
    "save_configuration",
    "timeit",
    "trace_on_error",
    "GracefulShutdown",
    "ConsoleHandler",
    "WandbHandler",
    "JSONLHandler",
    "LocalStorageHandler",
]

# ---------------------------------------------------------------------------
# Import-time safety
# ---------------------------------------------------------------------------

# Attach a NullHandler so importing goggles never emits logs by default.

_logger = logging.getLogger(__name__)
if not any(isinstance(h, logging.NullHandler) for h in _logger.handlers):
    _logger.addHandler(logging.NullHandler())
