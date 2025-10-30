"""Internal logger implementation.

WARNING: This module is an internal implementation detail of Goggles'
logging system. It is not part of the public API.

External code should not import from this module. Instead, depend on:
  - `goggles.TextLogger`, `goggles.GogglesLogger` (protocol / interface), and
  - `goggles.get_logger()` (factory returning a TextLogger/GogglesLogger).
"""

import logging
import inspect
from typing import Any, Dict, Mapping, Optional, Any
from typing_extensions import Self

from goggles import TextLogger, GogglesLogger, Event, GOGGLES_ASYNC
from goggles.types import Metrics, Image, Video, VectorField, Vector


class CoreTextLogger(TextLogger):
    """Internal concrete implementation of the TextLogger protocol.

    This adapter wraps a `logging.Logger` and maintains a dictionary of
    persistent, structured fields ("bound" context). Each log call merges
    the bound context with per-call extras before delegating to the underlying
    logger.

    Notes:
        * This class is **internal** to Goggles. Do not rely on its presence,
          constructor, or attributes from external code.
        * External users should obtain a `TextLogger` via
          `goggles.get_logger()` and program against the protocol.

    Attributes:
        _logger: Underlying `logging.Logger` instance. Internal use only.
        _bound: Persistent structured fields merged into each record.
            Internal use only.
        _client: EventBus client for emitting structured events.

    """

    def __init__(
        self,
        scope: str,
        name: Optional[str] = None,
        to_bind: Optional[Mapping[str, Any]] = None,
    ):
        """Initialize the CoreTextLogger.

        Args:
            scope (str): Scope to bind the logger to (e.g., "global", "run", ecc.).
            name (str): Name of the logger.
            to_bind (Optional[Mapping[str, Any]]):
                Optional initial persistent context to bind.

        """
        from goggles._core.routing import get_bus

        self.name = name
        self._scope = scope
        self._bound: Dict[str, Any] = dict(to_bind or {})
        self._client = get_bus()

    def bind(self, /, *, scope: str = "global", **fields: Any) -> Self:
        """Return a new logger with `fields` merged into persistent context.

        This method does not mutate the current instance. It returns a new
        adapter whose bound context is the shallow merge of the existing bound
        dictionary and `fields`. Keys in `fields` overwrite existing keys.

        Args:
            scope: Scope to bind the new logger under (e.g., "global" or "run").
            **fields: Key-value pairs to bind into the new logger's context.

        Returns:
            Self: A new adapter with the merged persistent context.

        Raises:
            TypeError: If provided keys are not strings (may occur in stricter
                configurations; current implementation assumes string keys).

        Examples:
            >>> log = get_logger("goggles")  # via public API
            >>> run_log = log.bind(scope="exp42", module="train")
            >>> run_log.info("Initialized")

        """
        self._bound = {**self._bound, **fields}
        self._scope = scope

        return self

    def get_bound(self) -> Dict[str, Any]:
        """Get a copy of the current persistent bound context.

        Returns:
            Dict[str, Any]: A shallow copy of the bound context dictionary.

        """
        return dict(self._bound)

    def debug(
        self,
        msg: str,
        /,
        *,
        step: Optional[int] = None,
        time: Optional[float] = None,
        **extra: Any,
    ) -> None:
        """Log a DEBUG message with optional per-call structured fields.

        Args:
            msg: Human-readable message.
            step: Step number associated with the event.
            time: Timestamp of the event in seconds since epoch.
            **extra: Per-call structured fields merged with the bound context.

        """
        filepath, lineno = _caller_id()
        future = self._client.emit(
            Event(
                kind="log",
                scope=self._scope,
                payload=msg,
                filepath=filepath,
                lineno=lineno,
                level=logging.DEBUG,
                step=step,
                time=time,
                extra={**self._bound, **extra},
            ).to_dict()
        )
        if not GOGGLES_ASYNC:
            future.result()

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
        filepath, lineno = _caller_id()
        future = self._client.emit(
            Event(
                kind="log",
                scope=self._scope,
                payload=msg,
                filepath=filepath,
                lineno=lineno,
                level=logging.INFO,
                step=step,
                time=time,
                extra={**self._bound, **extra},
            ).to_dict()
        )

        if not GOGGLES_ASYNC:
            future.result()

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
            msg: Human-readable message.
            step (Optional[int]): The step number.
            time (Optional[float]): The timestamp.
            **extra: Per-call structured fields merged with the bound context.

        """
        filepath, lineno = _caller_id()
        future = self._client.emit(
            Event(
                kind="log",
                scope=self._scope,
                payload=msg,
                filepath=filepath,
                lineno=lineno,
                level=logging.WARNING,
                step=step,
                time=time,
                extra={**self._bound, **extra},
            ).to_dict()
        )

        if not GOGGLES_ASYNC:
            future.result()

    def error(
        self,
        msg: str,
        /,
        *,
        step: Optional[int] = None,
        time: Optional[float] = None,
        **extra: Any,
    ) -> None:
        """Log an ERROR message with optional per-call structured fields.

        Args:
            msg: Human-readable message.
            step (Optional[int]): The step number.
            time (Optional[float]): The timestamp.
            **extra: Per-call structured fields merged with the bound context.

        """
        filepath, lineno = _caller_id()
        future = self._client.emit(
            Event(
                kind="log",
                scope=self._scope,
                payload=msg,
                level=logging.ERROR,
                filepath=filepath,
                lineno=lineno,
                step=step,
                time=time,
                extra={**self._bound, **extra},
            ).to_dict()
        )

        if not GOGGLES_ASYNC:
            future.result()

    def critical(
        self,
        msg: str,
        /,
        *,
        step: Optional[int] = None,
        time: Optional[float] = None,
        **extra: Dict[str, Any],
    ) -> None:
        """Log a CRITICAL message with optional per-call structured fields.

        Args:
            msg: Human-readable message.
            step (Optional[int]): The step number.
            time (Optional[float]): The timestamp.
            **extra: Per-call structured fields merged with the bound context.

        """
        filepath, lineno = _caller_id()
        future = self._client.emit(
            Event(
                kind="log",
                scope=self._scope,
                payload=msg,
                level=logging.CRITICAL,
                filepath=filepath,
                lineno=lineno,
                step=step,
                time=time,
                extra={**self._bound, **extra},
            ).to_dict()
        )

        if not GOGGLES_ASYNC:
            future.result()

    def __repr__(self) -> str:
        """Return a developer-friendly string representation.

        Returns:
            str: String representation showing the underlying
                logger and bound context.

        """
        return (
            f"{self.__class__.__name__}(name={self.name!r}, " f"bound={self._bound!r})"
        )


class CoreGogglesLogger(GogglesLogger, CoreTextLogger):
    """A GogglesLogger that is also a CoreTextLogger."""

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
        filepath, lineno = _caller_id()
        future = self._client.emit(
            Event(
                kind="metric",
                scope=self._scope,
                payload=metrics,
                level=None,
                filepath=filepath,
                lineno=lineno,
                step=step,
                time=time,
                extra={**self._bound, **extra},
            ).to_dict()
        )

        if not GOGGLES_ASYNC:
            future.result()

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
        filepath, lineno = _caller_id()
        future = self._client.emit(
            Event(
                kind="metric",
                scope=self._scope,
                payload={name: value},
                level=None,
                filepath=filepath,
                lineno=lineno,
                step=step,
                time=time,
                extra={**self._bound, **extra},
            ).to_dict()
        )

        if not GOGGLES_ASYNC:
            future.result()

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
        filepath, lineno = _caller_id()
        extra = {**self._bound, **extra}
        if name is not None:
            extra["name"] = name
        extra["format"] = format
        future = self._client.emit(
            Event(
                kind="image",
                scope=self._scope,
                payload=image,
                level=None,
                filepath=filepath,
                lineno=lineno,
                step=step,
                time=time,
                extra=extra,
            ).to_dict()
        )

        if not GOGGLES_ASYNC:
            future.result()

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
        filepath, lineno = _caller_id()
        extra = {**self._bound, **extra}
        if name is not None:
            extra["name"] = name
        extra["fps"] = fps
        extra["format"] = format

        future = self._client.emit(
            Event(
                kind="video",
                scope=self._scope,
                payload=video,
                level=None,
                filepath=filepath,
                lineno=lineno,
                step=step,
                time=time,
                extra=extra,
            ).to_dict()
        )

        if not GOGGLES_ASYNC:
            future.result()

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
            name (str): Artifact name.
            data (bytes): Artifact data.
            format (str): Artifact format, e.g., "txt", "bin".
            step (Optional[int]): Optional global step index.
            time (Optional[float]): Optional global timestamp.
            **extra (Dict[str, Any]): Additional routing metadata.

        """
        filepath, lineno = _caller_id()
        extra = {**self._bound, **extra}
        if name is not None:
            extra["name"] = name
        extra["format"] = format

        future = self._client.emit(
            Event(
                kind="artifact",
                scope=self._scope,
                payload=data,
                level=None,
                filepath=filepath,
                lineno=lineno,
                step=step,
                time=time,
                extra=extra,
            ).to_dict()
        )

        if not GOGGLES_ASYNC:
            future.result()

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
        filepath, lineno = _caller_id()
        extra = {**self._bound, **extra}
        if name is not None:
            extra["name"] = name

        future = self._client.emit(
            Event(
                kind="vector_field",
                scope=self._scope,
                payload=vector_field,
                level=None,
                filepath=filepath,
                lineno=lineno,
                step=step,
                time=time,
                extra=extra,
            ).to_dict()
        )

        if not GOGGLES_ASYNC:
            future.result()

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
            name (str): Artifact name.
            histogram (Vector): Histogram data.
            step (Optional[int]): Optional global step index.
            time (Optional[float]): Optional global timestamp.
            **extra (Dict[str, Any]): Additional routing metadata.

        """
        filepath, lineno = _caller_id()
        extra = {**self._bound, **extra}
        if name is not None:
            extra["name"] = name

        future = self._client.emit(
            Event(
                kind="histogram",
                scope=self._scope,
                payload=histogram,
                level=None,
                filepath=filepath,
                lineno=lineno,
                step=step,
                time=time,
                extra=extra,
            ).to_dict()
        )

        if not GOGGLES_ASYNC:
            future.result()


def _caller_id() -> tuple[str, int]:
    """Get the caller's filepath and line number for logging purposes.

    Returns:
        tuple[str, int]: A tuple of (file path, line number).

    """
    frame = inspect.currentframe()
    if frame is None or frame.f_back is None or frame.f_back.f_back is None:
        return ("<unknown>", 0)
    caller_frame = frame.f_back.f_back
    filename = caller_frame.f_code.co_filename
    line_number = caller_frame.f_lineno
    return (filename, line_number)
