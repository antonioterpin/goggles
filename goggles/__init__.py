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
    overload,
    runtime_checkable,
)
from typing_extensions import Self
import logging
import os

from .types import Kind, Event, VectorField, Video, Image, Vector, Metrics
from ._core.integrations import ConsoleHandler, LocalStorageHandler
from .decorators import timeit, trace_on_error
from .shutdown import GracefulShutdown
from .config import load_configuration, save_configuration

# Goggles port for bus communication
GOGGLES_PORT = os.getenv("GOGGLES_PORT", "2304")

# Handler registry for custom handlers
_HANDLER_REGISTRY: Dict[str, type] = {}
GOGGLES_HOST = os.getenv("GOGGLES_HOST", "localhost")
GOGGLES_ASYNC = os.getenv("GOGGLES_ASYNC", "1").lower() in ("1", "true", "yes")

# Cache the implementation after first use to avoid repeated imports
__impl_get_bus: Optional[Callable[[], EventBus]] = None


def _make_text_logger(
    name: Optional[str],
    scope: str,
    to_bind: dict[str, Any],
) -> TextLogger:
    from ._core.logger import CoreTextLogger

    return CoreTextLogger(name=name, scope=scope, to_bind=to_bind)


def _make_goggles_logger(
    name: Optional[str],
    scope: str,
    to_bind: dict[str, Any],
) -> GogglesLogger:
    from ._core.logger import CoreGogglesLogger

    return CoreGogglesLogger(name=name, scope=scope, to_bind=to_bind)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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
        data: Any,
        *,
        name: Optional[str] = None,
        format: str = "bin",
        step: Optional[int] = None,
        time: Optional[float] = None,
        **extra: Dict[str, Any],
    ) -> None:
        """Emit a generic artifact (encoded bytes).

        Args:
            data (bytes): Artifact data.
            name (Optional[str]): Artifact name.
            format (str): Artifact format, e.g., "txt", "bin".
            step (Optional[int]): Optional global step index.
            time (Optional[float]): Optional global timestamp.
            **extra (Dict[str, Any]): Additional routing metadata.

        """

    def vector_field(
        self,
        vector_field: VectorField,
        *,
        name: Optional[str] = None,
        step: Optional[int] = None,
        time: Optional[float] = None,
        **extra: Dict[str, Any],
    ) -> None:
        """Emit a vector field artifact.

        Args:
            vector_field (VectorField): Vector field data.
            name (Optional[str]): Artifact name.
            step (Optional[int]): Optional global step index.
            time (Optional[float]): Optional global timestamp.
            **extra (Dict[str, Any]): Additional routing metadata.

        """

    def histogram(
        self,
        histogram: Vector,
        *,
        name: Optional[str] = None,
        step: Optional[int] = None,
        time: Optional[float] = None,
        **extra: Dict[str, Any],
    ) -> None:
        """Emit a histogram artifact.

        Args:
            histogram (Vector): Histogram data.
            name (Optional[str]): Artifact name.
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

    def shutdown(self) -> None:
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


def get_bus() -> portal.Client:
    """Return the process-wide EventBus singleton client.

    The EventBus owns handlers and routes events based on scope and kind.

    Returns:
        portal.Client: The singleton EventBus client.

    """
    global __impl_get_bus
    if __impl_get_bus is None:
        from ._core.routing import get_bus as _impl_get_bus

        __impl_get_bus = _impl_get_bus  # type: ignore
    return __impl_get_bus()  # type: ignore


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
