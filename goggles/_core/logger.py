"""Internal logger implementation.

WARNING: This module is an internal implementation detail of Goggles'
logging system. It is not part of the public API.

External code should not import from this module. Instead, depend on:
  - `goggles.BoundLogger`, `goggles.GogglesLogger` (protocol / interface), and
  - `goggles.get_logger()` (factory returning a BoundLogger/GogglesLogger).
"""

import logging
from typing import Any, Dict, Mapping, Optional, Self

from goggles import BoundLogger, GogglesLogger, Event
from goggles.types import Metrics, Image, Video


class CoreBoundLogger(BoundLogger):
    """Internal concrete implementation of the BoundLogger protocol.

    This adapter wraps a `logging.Logger` and maintains a dictionary of
    persistent, structured fields ("bound" context). Each log call merges
    the bound context with per-call extras before delegating to the underlying
    logger.

    Notes:
        * This class is **internal** to Goggles. Do not rely on its presence,
          constructor, or attributes from external code.
        * External users should obtain a `BoundLogger` via
          `goggles.get_logger()` and program against the protocol.

    Attributes:
        _logger: Underlying `logging.Logger` instance. Internal use only.
        _bound: Persistent structured fields merged into each record.
            Internal use only.
        _client: EventBus client for emitting structured events.

    """

    def __init__(
        self,
        logger: logging.Logger,
        scope: str,
        to_bind: Optional[Mapping[str, Any]] = None,
    ):
        """Initialize the CoreBoundLogger.

        Args:
            logger (logging.Logger): Underlying Python logger.
            scope (str): Scope to bind the logger to (e.g., "global" or "run").
            to_bind (Optional[Mapping[str, Any]]):
                Optional initial persistent context to bind.

        """
        from goggles._core.routing import get_bus

        self._logger = logger
        self._scope = scope
        self._bound: Dict[str, Any] = dict(to_bind or {})
        self._client = get_bus()

    def bind(self, *, scope: str, **fields: Any) -> Self:
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
        self._client.emit(
            Event(
                kind="log",
                scope=self._scope,
                payload=msg,
                level=logging.DEBUG,
                step=step,
                time=time,
                extra={**self._bound, **extra},
            )
        )

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
        self._client.emit(
            Event(
                kind="log",
                scope=self._scope,
                payload=msg,
                level=logging.INFO,
                step=step,
                time=time,
                extra={**self._bound, **extra},
            )
        )

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
        self._client.emit(
            Event(
                kind="log",
                scope=self._scope,
                payload=msg,
                level=logging.WARNING,
                step=step,
                time=time,
                extra={**self._bound, **extra},
            )
        )

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
        self._client.emit(
            Event(
                kind="log",
                scope=self._scope,
                payload=msg,
                level=logging.ERROR,
                step=step,
                time=time,
                extra={**self._bound, **extra},
            )
        )

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
        self._client.emit(
            Event(
                kind="log",
                scope=self._scope,
                payload=msg,
                level=logging.CRITICAL,
                step=step,
                time=time,
                extra={**self._bound, **extra},
            )
        )

    def __repr__(self) -> str:
        """Return a developer-friendly string representation.

        Returns:
            str: String representation showing the underlying
                logger and bound context.

        """
        return (
            f"{self.__class__.__name__}(logger={self._logger!r}, "
            f"bound={self._bound!r})"
        )


class CoreGogglesLogger(GogglesLogger, CoreBoundLogger):
    """A GogglesLogger that is also a CoreBoundLogger."""

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
        self._client.emit(
            Event(
                kind="metric",
                scope=self._scope,
                payload=metrics,
                level=None,
                step=step,
                time=time,
                extra={**self._bound, **extra},
            )
        )

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
        self.push({name: value}, step=step, time=time, **extra)

    def image(
        self,
        name: str,
        image: Image,
        *,
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
        self._client.emit(
            Event(
                kind="image",
                scope=self._scope,
                payload={"name": name, "data": image, "format": format},
                level=None,
                step=step,
                time=time,
                extra={**self._bound, **extra},
            )
        )

    def video(
        self,
        name: str,
        video: Video,
        *,
        fps: int = 30,
        step: Optional[int] = None,
        time: Optional[float] = None,
        **extra: Dict[str, Any],
    ) -> None:
        """Emit a video artifact (encoded bytes).

        Args:
            name (str): Artifact name.
            video (Video): Video.
            fps (int): Frames per second.
            step (Optional[int]): Optional global step index.
            time (Optional[float]): Optional global timestamp.
            **extra (Dict[str, Any]): Additional routing metadata.

        """
        self._client.emit(
            Event(
                kind="video",
                scope=self._scope,
                payload={"name": name, "video": video, "fps": fps},
                level=None,
                step=step,
                time=time,
                extra={**self._bound, **extra},
            )
        )
