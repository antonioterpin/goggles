"""Internal logger implementation.

WARNING: This module is an internal implementation detail of Goggles'
logging system. It is not part of the public API.

External code should not import from this module. Instead, depend on:
  - `goggles.TextLogger`, `goggles.GogglesLogger` (protocol / interface), and
  - `goggles.get_logger()` (factory returning a TextLogger/GogglesLogger).
"""

import inspect
import logging
import os
from typing import Any

import numpy as np
from typing_extensions import Self

from goggles import GOGGLES_ASYNC, Event, GogglesLogger, TextLogger
from goggles.types import (
    Image,
    Metrics,
    Trajectories,
    Vector,
    VectorField,
    Video,
)

# Walking the call stack via `inspect.currentframe` is ~5-15 μs and
# allocates. At 10 kHz that's measurable on the producer hot path. Set
# GOGGLES_CAPTURE_CALLER=0 to skip it — every event will carry
# ("<unknown>", 0) for filepath/lineno, which matters only for the console
# formatter (wandb and file handlers don't use it).
_CAPTURE_CALLER: bool = os.getenv("GOGGLES_CAPTURE_CALLER", "1").lower() in (
    "1",
    "true",
    "yes",
)
_UNKNOWN_CALLER: tuple[str, int] = ("<unknown>", 0)


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
    """

    def __init__(
        self,
        scope: str,
        name: str | None = None,
        **to_bind: Any,
    ):
        """Initialize the CoreTextLogger.

        Args:
            scope: Scope to bind the logger to (e.g., "global", "run", etc.).
            name: Optional name of the logger.
            **to_bind:
                Optional initial persistent context to bind.

        """
        # Importing here to avoid circular imports
        from goggles._core.routing import get_bus  # noqa: PLC0415

        self.name = name
        self._scope = scope
        self._bound: dict[str, Any] = dict(**to_bind or {})
        self._client = get_bus()

    def _dispatch(self, event: Event, *, async_mode: bool) -> None:
        """Route ``event`` through the transport.

        Args:
            event: Event to dispatch.
            async_mode: If True, fire-and-forget. Otherwise perform a
                synchronous handoff to the configured transport before
                returning. This does not imply cross-process delivery,
                remote routing completion, or acknowledgement for
                transports that do not provide those guarantees (for
                example, ``LocalTransport`` in client mode, where
                "sync" is a best-effort local-queue flush).
        """
        if async_mode:
            self._client.emit(event)
        else:
            self._client.emit_sync(event)

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

        Examples:
            >>> log = get_logger("goggles")  # via public API
            >>> run_log = log.bind(scope="exp42", module="train")
            >>> run_log.info("Initialized")

        """
        return self.__class__(
            scope=scope,
            name=self.name,
            **{**self._bound, **fields},
        )

    def get_bound(self) -> dict[str, Any]:
        """Get a copy of the current persistent bound context.

        Returns:
            A shallow copy of the bound context dictionary.

        """
        return dict(self._bound)

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
        """Log a DEBUG message with optional per-call structured fields.

        Args:
            msg: Human-readable message.
            step: Step number associated with the event.
            time: Timestamp of the event in seconds since epoch.
            async_mode: If True, do not block waiting for delivery.
            **extra: Per-call structured fields merged with the bound context.

        """
        filepath, lineno = _caller_id()
        self._dispatch(
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
            ),
            async_mode=async_mode,
        )

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
            **extra: Additional structured key-value pairs for this record.

        """
        filepath, lineno = _caller_id()
        self._dispatch(
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
            ),
            async_mode=async_mode,
        )

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
            msg: Human-readable message.
            step: (int | None): The step number.
            time: (float | None): The timestamp.
            async_mode: If True, do not block waiting for delivery.
            **extra: Per-call structured fields merged with the bound context.

        """
        filepath, lineno = _caller_id()
        self._dispatch(
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
            ),
            async_mode=async_mode,
        )

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
        """Log an ERROR message with optional per-call structured fields.

        Args:
            msg: Human-readable message.
            step: The step number.
            time: The timestamp.
            async_mode: If True, do not block waiting for delivery.
            **extra: Per-call structured fields merged with the bound context.

        """
        filepath, lineno = _caller_id()
        self._dispatch(
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
            ),
            async_mode=async_mode,
        )

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
        """Log a CRITICAL message with optional per-call structured fields.

        Args:
            msg: Human-readable message.
            step: (int | None): The step number.
            time: (float | None): The timestamp.
            async_mode: If True, do not block waiting for delivery.
            **extra: Per-call structured fields merged with the bound context.

        """
        filepath, lineno = _caller_id()
        self._dispatch(
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
            ),
            async_mode=async_mode,
        )

    def __repr__(self) -> str:
        """Return a developer-friendly string representation.

        Returns:
            str: String representation showing the underlying
                logger and bound context.

        """
        return (
            f"{self.__class__.__name__}(name={self.name!r}, "
            f"bound={self._bound!r})"
        )


class CoreGogglesLogger(GogglesLogger, CoreTextLogger):
    """A GogglesLogger that is also a CoreTextLogger."""

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
            **extra: Additional routing metadata (e.g., split="train").

        """
        filepath, lineno = _caller_id()
        self._dispatch(
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
            ),
            async_mode=async_mode,
        )

    def scalar(
        self,
        name: str,
        value: float | int,
        step: int | None = None,
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
            **extra: Additional routing metadata (e.g., split="train").

        """
        filepath, lineno = _caller_id()
        self._dispatch(
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
            ),
            async_mode=async_mode,
        )

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
            name: Artifact name.
            format: Image format, e.g., "png", "jpeg".
            time: Optional global timestamp.
            async_mode: If True, do not block waiting for delivery.
            **extra: Additional routing metadata.

        """
        filepath, lineno = _caller_id()
        extra = {**self._bound, **extra}
        if name is not None:
            extra["name"] = name
        extra["format"] = format
        self._dispatch(
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
            ),
            async_mode=async_mode,
        )

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
            name: Artifact name.
            fps: Frames per second.
            format: Video format, e.g., "gif", "mp4".
            time: Optional global timestamp.
            async_mode: If True, do not block waiting for delivery.
            **extra: Additional routing metadata.

        """
        filepath, lineno = _caller_id()
        extra = {**self._bound, **extra}
        if name is not None:
            extra["name"] = name
        extra["fps"] = fps
        extra["format"] = format

        self._dispatch(
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
            ),
            async_mode=async_mode,
        )

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
            name: Artifact name.
            format: Artifact format, e.g., "txt", "bin".
            time: Optional global timestamp.
            async_mode: If True, do not block waiting for delivery.
            **extra: Additional routing metadata.

        """
        filepath, lineno = _caller_id()
        extra = {**self._bound, **extra}
        if name is not None:
            extra["name"] = name
        extra["format"] = format

        self._dispatch(
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
            ),
            async_mode=async_mode,
        )

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
        filepath, lineno = _caller_id()
        extra = {**self._bound, **extra}
        if name is not None:
            extra["name"] = name

        self._dispatch(
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
            ),
            async_mode=async_mode,
        )

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
            trajectories: Array of shape ``(N, L, dim)`` where ``N`` is the
                number of trajectories, ``L`` their length, and ``dim`` the
                spatial dimension (2 or 3).
            step: Global step index.
            name: Optional artifact name.
            time: Optional global timestamp.
            async_mode: If True, do not block waiting for delivery.
            **extra: Additional routing metadata (e.g.
                ``store_visualization=True`` to also save a PNG preview).
        """
        filepath, lineno = _caller_id()
        extra = {**self._bound, **extra}
        if name is not None:
            extra["name"] = name

        self._dispatch(
            Event(
                kind="trajectories",
                scope=self._scope,
                payload=trajectories,
                level=None,
                filepath=filepath,
                lineno=lineno,
                step=step,
                time=time,
                extra=extra,
            ),
            async_mode=async_mode,
        )

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
        filepath, lineno = _caller_id()
        extra = {**self._bound, **extra}
        extra["static"] = static
        if name is not None:
            extra["name"] = name

        self._dispatch(
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
            ),
            async_mode=async_mode,
        )

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
            * The `name` parameter serves as a base for the emitted metrics.
            * Each key in the `data` dictionary is appended to the base name
                to form the full metric name (e.g., `name/key`).
            * Values in the dictionary are emitted according to their type:
                - Scalars (int, float) are emitted as single metrics.
                - 1D arrays are emitted as multiple metrics with indexed names
                    (e.g., `name/key_0`, `name/key_1`, ...).
                - 2D arrays are emitted as images.
                - 3D arrays are emitted as images if the last dimension has
                    1, 3, or 4 channels;
                    if the last dimension has 2 channels, they are emitted
                    as vector fields.
            * Unsupported types are logged as errors.

        Args:
            name: Base name for the metrics.
            data: Dictionary data.
            step: Global step index.
            time: Optional global timestamp.
            async_mode: If True, do not block waiting for delivery.
            **extra: Additional routing metadata.

        """
        for topic, value in data.items():
            topic_str = str(topic)  # Ensure str
            name_log = (
                f"{name}{topic_str}"
                if topic_str.startswith("/")
                else f"{name}/{topic_str}"
            )

            if isinstance(value, (int, float, np.number)):
                self.scalar(
                    name_log,
                    float(value),
                    step=step,
                    time=time,
                    async_mode=async_mode,
                    **extra,
                )
                continue

            if isinstance(value, np.ndarray):
                if value.size == 1:
                    self.scalar(
                        name_log,
                        float(value.item()),
                        step=step,
                        time=time,
                        async_mode=async_mode,
                        **extra,
                    )
                    continue

                elif value.ndim == 1:
                    for i, v in enumerate(value):
                        self.scalar(
                            f"{name_log}_{i}",
                            float(v),
                            step=step,
                            time=time,
                            async_mode=async_mode,
                            **extra,
                        )
                    continue

                elif value.ndim == 2:
                    self.image(
                        value,
                        step=step,
                        name=name_log,
                        time=time,
                        async_mode=async_mode,
                        **extra,
                    )
                    continue

                elif value.ndim == 3:
                    if value.shape[2] in (1, 3, 4):
                        self.image(
                            value,
                            step=step,
                            name=name_log,
                            time=time,
                            async_mode=async_mode,
                            **extra,
                        )
                        continue
                    elif value.shape[2] == 2:
                        self.vector_field(
                            value,
                            step=step,
                            name=name_log,
                            time=time,
                            async_mode=async_mode,
                            **extra,
                        )
                        continue

                elif value.ndim == 4:
                    if value.shape[2] in (1, 3, 4):
                        self.image(
                            value,
                            step=step,
                            name=name_log,
                            time=time,
                            async_mode=async_mode,
                            **extra,
                        )
                        continue
                    elif value.shape[2] == 2:
                        self.vector_field(
                            value,
                            step=step,
                            name=name_log,
                            time=time,
                            async_mode=async_mode,
                            **extra,
                        )
                        continue

            self.error(
                f"Unsupported type for dictionary logging: topic={topic}, "
                f"type={type(value)}",
                time=time,
                step=step,
                async_mode=async_mode,
                **extra,
            )


def _caller_id() -> tuple[str, int]:
    """Get the caller's filepath and line number for logging purposes.

    Honours the ``GOGGLES_CAPTURE_CALLER`` env var: when disabled, returns
    a constant tuple and skips the frame walk entirely — relevant for
    producers logging at 10 kHz+ where the 5-15 μs stack walk is visible
    in p99 latency.

    Returns:
        A tuple of (file path, line number).

    """
    if not _CAPTURE_CALLER:
        return _UNKNOWN_CALLER
    frame = inspect.currentframe()
    if frame is None or frame.f_back is None or frame.f_back.f_back is None:
        return _UNKNOWN_CALLER
    caller_frame = frame.f_back.f_back
    return (caller_frame.f_code.co_filename, caller_frame.f_lineno)
