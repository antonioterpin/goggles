"""Internal logger implementation."""

from __future__ import annotations
import numpy as np
import inspect
import logging
from typing import Any, Mapping, Sequence, cast, Literal
from pathlib import Path
from queue import Empty, Queue
from threading import Event as ThreadEvent, Thread

from goggles._core.config_simple import (
    ConsoleConfig,
    WandBConfig,
    CONSOLE,
    WANDB,
)

from goggles import GOGGLES_ASYNC, Event, GogglesLogger, TextLogger
from goggles.types import Image, Metrics, Vector, VectorField, Video
import weakref

ACTIVE: weakref.WeakSet[CoreGogglesLogger] = weakref.WeakSet()
Run = Any  # wandb.sdk.wandb_run.Run


class CoreTextLogger(TextLogger):
    """Internal concrete implementation of the TextLogger protocol."""

    def __init__(
        self,
        scope: str,
        name: str | None = None,
        console_config: ConsoleConfig = CONSOLE,
        **to_bind: Any,
    ):
        """Initialize the CoreTextLogger.

        Args:
            scope: Scope to bind the logger to (e.g., "global", "run", etc.).
            name: Optional name of the logger.
            console_config: ConsoleConfig instance for console logging.
            **to_bind:
                Optional initial persistent context to bind.

        """
        self.name = name
        self._scope = scope
        self._bound: dict[str, Any] = dict(**to_bind or {})
        self.console_config = console_config
        self.color_codes = {
            "DEBUG": "\033[36m",  # Cyan
            "INFO": "\033[34m",  # Blue
            "WARNING": "\033[33m",  # Yellow
            "ERROR": "\033[31m",  # Red
            "CRITICAL": "\033[91m",  # Bright Red
        }
        self.reset_code = "\033[0m"  # Reset color
        self._logger: logging.Logger | None = None

    def open(self) -> None:
        """Initialize the logger (create logger and formatter)."""
        if not self.console_config.enabled:
            return  # Don't initialize if console logging is disabled
        if self._logger is not None:
            return

        logger = logging.getLogger(self.console_config.name)
        logger.propagate = False
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            logger.addHandler(handler)
        logger.setLevel(self.console_config.level or logging.INFO)
        self._logger = logger

    def _log(self, level: int, msg: str, filepath: str, lineno: int) -> None:
        """Forward a log event to Python's logging system.

        Args:
            level: The logging level (e.g., logging.DEBUG).
            msg: The message to log.
            filepath: The file path where the log event originated.
            lineno: The line number in the file where the log event originated.
        """
        if self._logger is None:
            self.open()

        if self._logger is None:
            return

        # Derive display path
        path = Path(filepath)
        if self.console_config.path_style == "relative":
            try:
                path = path.relative_to(self.console_config.project_root)
            except ValueError:
                pass  # fallback to absolute if outside root
        path_str = f"{path}:{lineno}"

        # ANSI color codes for different log levels
        level_name = logging.getLevelName(level)

        color = self.color_codes.get(level_name, "")
        colored_message = f"{color}{path_str} - {msg}{self.reset_code}"

        # We manually construct prefix since stacklevel=3 may mislead
        self._logger.log(level, colored_message, stacklevel=2)

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
        self._log(logging.DEBUG, msg, filepath, lineno)

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
        self._log(logging.INFO, msg, filepath, lineno)

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
        self._log(logging.WARNING, msg, filepath, lineno)

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
        self._log(logging.ERROR, msg, filepath, lineno)

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
        self._log(logging.CRITICAL, msg, filepath, lineno)

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

    def __init__(
        self,
        scope: str,
        name: str | None = None,
        console_config: ConsoleConfig = CONSOLE,
        wandb_config: WandBConfig = WANDB,
        **to_bind: Any,
    ) -> None:
        """Initialize logger and start a placeholder background worker.

        Args:
            scope: The scope of the logger.
            name: The name of the logger.
            console_config: The console configuration for this logger.
            wandb_config: The W&B configuration for this logger.
            **to_bind:
                Additional key-value pairs to bind to the logger's context.
        """
        CoreTextLogger.__init__(
            self,
            scope=scope,
            name=name,
            console_config=console_config,
            **to_bind,
        )
        self.wandb_config = wandb_config
        self._event_queue: Queue[Event | None] = Queue()
        self._worker_stop = ThreadEvent()
        self._worker = Thread(
            target=self._worker_loop,
            name="goggles-event-worker",
            daemon=True,
        )
        self._worker.start()
        ACTIVE.add(self)

    def _wlog(self, level: int, msg: str, *args) -> None:
        """Worker-safe logging method for internal use.

        Args:
            level: The logging level.
            msg: The message to log.
            *args: Arguments to format the message with.
        """
        try:
            text = msg % args if args else msg
        except Exception:
            text = f"{msg} {args}"
        self._log(level, text, "<wandb-worker>", 0)

    def _worker_loop(self) -> None:
        """W&B worker: mirrors the old WandBHandler.handle() switch.

        Raises:
            ValueError: If the event payload is invalid for its kind.
        """
        import wandb
        from goggles.media import create_numpy_vector_field_visualization

        self._wandb_run = None

        while True:
            try:
                event = self._event_queue.get(timeout=0.1)
            except Empty:
                if self._worker_stop.is_set():
                    break
                continue

            if event is None:
                self._event_queue.task_done()
                break

            try:
                if not self.wandb_config.enabled:
                    # if wandb is disabled, just drop metric/media events
                    continue

                kind = getattr(event, "kind", None) or "metric"
                step = getattr(event, "step", None)
                payload = getattr(event, "payload", None)
                extra = dict(getattr(event, "extra", {}) or {})
                extra_config = extra.pop("config_wandb", {}) or {}

                run = self._ensure_wandb_run(extra_config)
                if run is None:
                    continue

                if kind == "metric":
                    if not isinstance(payload, Mapping):
                        raise ValueError(
                            "Metric event payload must be a mapping of name→value."
                        )
                    payload = {
                        k: v for k, v in payload.items() if v is not None
                    }
                    if not payload:
                        self._wlog(
                            logging.WARNING,
                            "Skipping metric log with empty payload (scope=%s).",
                            self._scope,
                        )
                        continue
                    for k, v in extra.items():
                        payload[k] = v
                    run.log(payload, step=step)
                    continue

                if kind in {"image", "video"}:
                    key_name = extra.pop("name", kind)
                    items = (
                        payload.items()
                        if isinstance(payload, Mapping)
                        else [(key_name, payload)]
                    )

                    logs = {}
                    for name, value in items:
                        if value is None:
                            self._wlog(
                                logging.WARNING,
                                "Skipping %s '%s' with None payload (scope=%s).",
                                kind,
                                name,
                                self._scope,
                            )
                            continue
                        if kind == "image":
                            logs[name] = wandb.Image(value)
                        else:
                            fps = int(extra.get("fps", 20))
                            fmt = str(extra.get("format", "mp4"))
                            if fmt not in {"mp4", "gif"}:
                                self._wlog(
                                    logging.WARNING,
                                    "Unsupported video format '%s' for '%s'; "
                                    "defaulting to 'mp4'.",
                                    fmt,
                                    name,
                                )

                                fmt = "mp4"
                            new_value = self._prepare_video_for_wandb(value)
                            logs[name] = wandb.Video(
                                new_value, fps=fps, format=fmt
                            )  # pyright: ignore[reportArgumentType]

                    for k, v in extra.items():
                        logs[k] = v
                    if logs:
                        run.log(logs, step=step)
                    continue

                if kind == "artifact":
                    if not isinstance(payload, Mapping):
                        self._wlog(
                            logging.WARNING,
                            "Artifact payload must be mapping; got %r",
                            type(payload),
                        )
                        continue
                    path = payload.get("path")
                    name = payload.get("name", "artifact")
                    art_type = payload.get("type", "misc")
                    if not isinstance(path, str):
                        self._wlog(
                            logging.WARNING,
                            "Artifact payload missing 'path' string; got %r",
                            type(path),
                        )
                        continue
                    artifact = wandb.Artifact(
                        name=name, type=art_type, metadata=extra
                    )
                    artifact.add_file(path)
                    run.log_artifact(artifact)
                    continue

                if kind == "histogram":
                    name = extra.pop("name", "histogram")
                    static = bool(extra.pop("static", False))
                    num_bins = int(extra.pop("num_bins", extra.pop("bins", 64)))

                    logs = {}
                    try:
                        if not isinstance(payload, (Sequence, np.ndarray)):
                            self._wlog(
                                logging.WARNING,
                                "Invalid histogram payload for '%s' (scope=%s): "
                                "must be a sequence or tuple.",
                                name,
                                self._scope,
                            )
                            continue

                        if static:
                            payload_list = list(payload)
                            data = [[v] for v in payload_list]
                            table = wandb.Table(data=data, columns=["values"])
                            logs[name] = wandb.plot.histogram(
                                table,
                                "values",
                                title="Histogram of Random Values",
                            )
                        else:
                            logs[name] = wandb.Histogram(
                                np_histogram=np.histogram(
                                    payload, bins=num_bins
                                )
                            )
                    except Exception as exc:
                        self._wlog(
                            logging.WARNING,
                            f"Invalid histogram payload for '{name}' "
                            f"(scope={self._scope}): {exc}",
                        )
                        continue

                    for k, v in extra.items():
                        logs[k] = v
                    if logs:
                        run.log(logs, step=step)
                    continue

                if kind == "vector_field":
                    name = extra.pop("name", "vector_field")
                    mode = str(extra.pop("mode", "magnitude"))
                    add_colorbar = bool(extra.pop("add_colorbar", False))

                    if mode not in {"vorticity", "magnitude"}:
                        self._wlog(
                            logging.WARNING,
                            f"Unknown vector field visualization mode '{mode}'. "
                            "Supported modes are: 'vorticity', 'magnitude'. "
                            "The vector field visualization will not be sent to W&B.",
                        )
                        continue
                    mode_literal = cast(Literal["vorticity", "magnitude"], mode)

                    logs = {}
                    items = (
                        payload.items()
                        if isinstance(payload, Mapping)
                        else [(name, payload)]
                    )
                    for field_name, value in items:
                        if value is None:
                            self._wlog(
                                logging.WARNING,
                                f"Skipping vector field '{field_name}' with None "
                                f"payload (scope={self._scope}).",
                            )
                            continue
                        try:
                            image = create_numpy_vector_field_visualization(
                                value,
                                mode=mode_literal,
                                add_colorbar=add_colorbar,
                            )
                            logs[field_name] = wandb.Image(image)
                        except Exception as exc:
                            self._wlog(
                                logging.WARNING,
                                f"Invalid vector field payload for '{field_name}' "
                                f"(scope={self._scope}): {exc}",
                            )
                            continue

                    for k, v in extra.items():
                        logs[k] = v
                    if logs:
                        run.log(logs, step=step)
                    continue

                # unsupported kinds: drop
                self._wlog(
                    logging.WARNING,
                    "Dropping unsupported event kind '%s' (scope=%s).",
                    kind,
                    self._scope,
                )
                continue

            except Exception as exc:
                self._wlog(
                    logging.ERROR,
                    f"Error processing event in W&B worker (scope={self._scope}): {exc}",
                )
                continue
            finally:
                self._event_queue.task_done()

        # shutdown: finish run if created
        run = getattr(self, "_wandb_run", None)
        if run is not None:
            try:
                run.finish()
            except Exception:
                pass
        self._wandb_run = None

    def _enqueue_event(self, event: Event) -> None:
        """Enqueue an event for background processing.

        Args:
            event: The event to enqueue.
        """
        self._event_queue.put(event)

    def close(self, timeout: float | None = None) -> None:
        """Stop the placeholder worker thread.

        Args:
            timeout: Maximum time to wait for the worker thread to finish. If None, wait indefinitely.
        """
        if self._worker_stop.is_set():
            return
        self._event_queue.put(None)  # Sentinel to unblock the worker if waiting
        self._worker_stop.set()
        self._worker.join(timeout=timeout)

    def _ensure_wandb_run(self, extra_config: dict) -> Any:
        """Create the per-logger W&B run if needed.

        Args:
            extra_config:
                Additional configuration to merge with the logger's WandBConfig when initializing

        Returns:
            The active W&B run, or None if W&B logging is disabled.
        """
        import wandb

        if getattr(self, "_wandb_run", None) is not None:
            return self._wandb_run

        if not self.wandb_config.enabled:
            return None

        # Run name: base run_name if provided, else derive from logger name.
        base = self.wandb_config.run_name
        if self._scope == "global" and base:
            run_name = base
        else:
            run_name = f"{base or 'run'}-{self._scope}"

        cfg = dict(self.wandb_config.config or {})
        cfg.update({"scope": self._scope})
        cfg.update(extra_config or {})

        self._wandb_run = wandb.init(
            project=self.wandb_config.project,
            entity=self.wandb_config.entity,
            name=run_name,
            group=self.wandb_config.group,
            reinit=cast(
                Literal[
                    "default",
                    "return_previous",
                    "finish_previous",
                    "create_new",
                ],
                self.wandb_config.reinit,
            ),
            config=cfg,
        )
        return self._wandb_run

    def _prepare_video_for_wandb(self, value: np.ndarray) -> np.ndarray:
        """Normalize video array to shape (B, F, 3, H, W) for W&B logging.

        Args:
            value: The input video array to prepare for W&B logging.

        Returns:
            A numpy array with shape (B, F, 3, H, W)
        """
        if value.ndim == 3:
            value = value[:, None, :, :]
        elif value.ndim not in (4, 5):
            # keep behavior: log error and continue
            self._wlog(logging.ERROR, f"Video has invalid shape {value.shape}")
            return value

        if value.ndim == 4 and value.shape[1] == 1:
            value = np.repeat(value, 3, axis=1)
        if value.ndim == 5 and value.shape[2] == 1:
            value = np.repeat(value, 3, axis=2)
        return value

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
        self._enqueue_event(
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
            )
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
        self._enqueue_event(
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
            )
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
        self._enqueue_event(
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
            )
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

        self._enqueue_event(
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
            )
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

        self._enqueue_event(
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
            )
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

        self._enqueue_event(
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
            )
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

        self._enqueue_event(
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
            )
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

    Returns:
        A tuple of (file path, line number).

    """
    frame = inspect.currentframe()
    if frame is None or frame.f_back is None or frame.f_back.f_back is None:
        return ("<unknown>", 0)
    caller_frame = frame.f_back.f_back
    filename = caller_frame.f_code.co_filename
    line_number = caller_frame.f_lineno
    return (filename, line_number)
