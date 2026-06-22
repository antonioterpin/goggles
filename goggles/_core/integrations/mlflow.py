"""MLflow integration handler for the Goggles logging framework.

Forwards Goggles events to `MLflow <https://mlflow.org>`_ via the
low-level :class:`mlflow.tracking.MlflowClient`. The client API (rather
than the global ``mlflow.log_*`` fluent API) is used deliberately: the
fluent API tracks a single *active run* on a thread-local stack, which
cannot represent the multiple concurrently-open runs this handler needs
(one run per scope, mirroring the W&B handler).

Event-kind mapping:

- ``metric`` -> ``MlflowClient.log_batch`` of ``Metric`` points.
- ``image`` -> ``MlflowClient.log_image`` (stepped image viewer).
- ``video`` -> encoded to a temp ``.gif``/``.mp4`` and logged as an
  artifact (MLflow has no native video type).
- ``vector_field`` / ``trajectories`` -> rendered to an RGB image via
  the base ``goggles.media`` helpers, then ``log_image``.
- ``histogram`` -> rendered to a Matplotlib figure, then ``log_figure``.
- ``artifact`` -> ``log_artifact`` (file) or ``log_artifacts`` (dir).

Remote use: pass ``tracking_uri="http://host:5000"`` to log to a remote
MLflow tracking server (the handler is then just an HTTP client and the
runs are viewable live at that URL); leave it unset to log to a local
``./mlruns`` store that ``mlflow ui`` can serve. The ``mlflow`` SDK
manages all transport, so this works across machines without Goggles'
own transport needing to be cross-machine.
"""

from __future__ import annotations

import logging
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

import numpy as np
from matplotlib.figure import Figure
from mlflow.entities import Metric, Param
from mlflow.tracking import MlflowClient
from typing_extensions import Self

from goggles.media import (
    create_numpy_trajectories_visualization,
    create_numpy_vector_field_visualization,
    save_numpy_gif,
    save_numpy_mp4,
)
from goggles.types import Kind

from ._step_guard import StepGuard

# Default number of bins for a dynamic histogram render.
_DEFAULT_HISTOGRAM_BINS = 64
# MLflow's always-present fallback experiment when none is configured.
_DEFAULT_EXPERIMENT = "Default"


def _to_metric_value(value: Any) -> float | None:
    """Coerce a metric value to a finite float, or None if not numeric.

    Args:
        value: A scalar or numpy value.

    Returns:
        The value as a Python ``float``, or None when it cannot be
        represented as one (MLflow metrics must be numeric).
    """
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _video_to_channels_last(array: np.ndarray) -> np.ndarray:
    """Normalize a video tensor to channels-last frames for ``imageio``.

    Accepted shapes:
    - ``(F, H, W)`` -- grayscale frames (returned unchanged).
    - ``(F, H, W, C)`` -- channels-last (``C`` in ``{1, 3, 4}``).
    - ``(F, C, H, W)`` -- channels-first (``C`` in ``{1, 3, 4}``).
    - ``(B, F, C, H, W)`` / ``(F, T, C, H, W)`` -- the two leading axes
      are collapsed into a single frame axis, then re-normalized.

    Args:
        array: The video tensor.

    Returns:
        A ``(F, H, W)`` or ``(F, H, W, C)`` channels-last array.

    Raises:
        ValueError: If the rank or channel layout is unsupported.
    """
    array = np.asarray(array)
    if array.ndim == 3:
        return array
    if array.ndim == 4:
        if array.shape[-1] in (1, 3, 4):
            return array
        if array.shape[1] in (1, 3, 4):
            return np.moveaxis(array, 1, -1)
        raise ValueError(
            f"4D video has shape {array.shape}; expected a channel dim "
            "(size 1, 3, or 4) at axis 1 or axis -1."
        )
    if array.ndim == 5:
        return _video_to_channels_last(array.reshape(-1, *array.shape[2:]))
    raise ValueError(
        f"Video has shape {array.shape}; expected (F, H, W), (F, H, W, C), "
        "(F, C, H, W), or a 5D batched variant."
    )


def _build_histogram_figure(values: np.ndarray, bins: int, title: str) -> Any:
    """Build a Matplotlib histogram figure for the given values.

    Args:
        values: The raw values to bin.
        bins: Number of histogram bins.
        title: Figure title (typically the histogram name).

    Returns:
        A ``matplotlib.figure.Figure``. Built without ``pyplot`` so it is
        not tracked in pyplot's global registry (no ``close`` needed) and
        no process-global backend state is touched -- safe to call from
        concurrent emit threads.
    """
    flat = np.asarray(values, dtype=np.float64).ravel()
    fig = Figure()
    ax = fig.subplots()
    ax.hist(flat, bins=bins)
    ax.set_title(title)
    ax.set_xlabel("value")
    ax.set_ylabel("count")
    return fig


class MLflowHandler:
    """Forward Goggles events to MLflow runs (one run per scope).

    Compatible with the ``Handler`` protocol used by the EventBus. Each
    scope maps to a distinct MLflow run that stays open until
    :meth:`close`, so concurrent scopes do not clobber one another's
    metrics.

    Out-of-order steps (``event.step`` strictly less than the highest
    step previously seen on the same scope) are dropped with a warning,
    so each run's metric history is non-decreasing per scope. Events with
    ``step is None`` are forwarded unchanged; ``artifact`` events are
    step-less and bypass the check.

    Unlike :class:`~goggles._core.integrations.wandb.WandBHandler`, this
    handler does not coalesce same-step events: MLflow records each
    ``(key, step, timestamp)`` point independently, so the W&B same-step
    workaround (issue #177) does not apply here.

    Attributes:
        name: Stable handler identifier.
        capabilities: Supported event kinds.
        GLOBAL_SCOPE: Default scope name for events without an explicit
            scope.
    """

    name: str = "mlflow"
    capabilities: ClassVar[frozenset[Kind]] = frozenset(
        {
            "metric",
            "image",
            "video",
            "artifact",
            "histogram",
            "vector_field",
            "trajectories",
        }
    )
    GLOBAL_SCOPE: ClassVar[str] = "global"

    def __init__(
        self,
        experiment: str | None = None,
        *,
        tracking_uri: str | None = None,
        artifact_location: str | None = None,
        run_name: str | None = None,
        params: Mapping[str, Any] | None = None,
        tags: Mapping[str, Any] | None = None,
        name: str = "mlflow",
    ) -> None:
        """Initialize the MLflow handler.

        Args:
            experiment: MLflow experiment name (the rough analog of a W&B
                project). Created on first use if it does not exist.
                Defaults to MLflow's ``"Default"`` experiment.
            tracking_uri: Where runs are logged. Pass an ``http(s)://``
                URL to stream to a remote MLflow tracking server, a
                ``sqlite:///...`` (or other database) URI for a local
                store, or None to use MLflow's ambient configuration (the
                ``MLFLOW_TRACKING_URI`` env var). Note: as of MLflow 3 the
                bare ``./mlruns`` file store is in maintenance mode and
                rejected unless ``MLFLOW_ALLOW_FILE_STORE=true`` is set, so
                a database backend is the recommended local option.
            artifact_location: Root URI for run artifacts (images, videos,
                checkpoints), applied only when this handler creates the
                experiment. Useful with a database backend, where artifacts
                otherwise default to ``./mlartifacts`` under the working
                directory. Ignored if the experiment already exists.
            run_name: Base name for the per-scope runs.
            params: Parameters logged once per run (stringified, as MLflow
                params are immutable strings). The rough analog of a W&B
                ``config``.
            tags: MLflow run tags applied to every run this handler
                creates. Pass a mapping of ``str -> value``.
            name: Stable handler identifier (for diagnostics and routing).
        """
        self._logger = logging.getLogger(self.name)
        # Keep Goggles' own diagnostics off the root logger to avoid
        # duplicate lines, mirroring the other handlers.
        self._logger.propagate = False

        self._experiment = experiment
        self._tracking_uri = tracking_uri
        self._artifact_location = artifact_location
        self._base_run_name = run_name
        self._params: dict[str, Any] = dict(params) if params else {}
        self._tags: dict[str, Any] = dict(tags) if tags else {}
        self.name = name

        self._client: MlflowClient | None = None
        self._experiment_id: str | None = None
        self._runs: dict[str, str] = {}
        self._step_guard = StepGuard()

    # -- protocol ---------------------------------------------------------

    def can_handle(self, kind: str) -> bool:
        """Return True if this handler supports the given event kind.

        Args:
            kind: The event kind to check.

        Returns:
            True if the kind is supported, False otherwise.
        """
        return kind in self.capabilities

    def open(self) -> None:
        """Create the MLflow client and resolve the target experiment."""
        self._ensure_client()

    def close(self) -> None:
        """Mark every open run as finished."""
        if self._client is not None:
            for run_id in self._runs.values():
                try:
                    self._client.set_terminated(run_id)
                except Exception as exc:
                    self._logger.warning(
                        "Failed to finish MLflow run %s: %s", run_id, exc
                    )
        self._runs.clear()

    def handle(self, event: Any) -> None:
        """Process a Goggles event and forward it to MLflow.

        A ``metric`` event whose payload is not a mapping raises a
        ``ValueError`` (from :meth:`_handle_metric`); the EventBus
        isolates that per handler.

        Args:
            event: The Goggles event to process.
        """
        scope = getattr(event, "scope", None) or self.GLOBAL_SCOPE
        kind = getattr(event, "kind", None) or "metric"
        step = getattr(event, "step", None)
        payload = getattr(event, "payload", None)

        # Artifacts are step-less and bypass the monotonic-step check.
        if kind != "artifact" and self._step_guard.check(scope, step):
            self._logger.warning(
                "Dropping out-of-order event (scope=%s, step=%s) -- "
                "step regressed below previously seen max",
                scope,
                step,
            )
            return

        # Copy so our reads/pops never mutate the shared event.extra that
        # sibling handlers on the same bus will read afterwards.
        extra = dict(getattr(event, "extra", None) or {})
        timestamp_ms = self._timestamp_ms(event)
        run_id = self._get_or_create_run(scope)

        if kind == "metric":
            self._handle_metric(
                run_id, payload, step, extra, scope, timestamp_ms
            )
        elif kind == "image":
            self._handle_image(run_id, payload, step, extra, scope)
        elif kind == "video":
            self._handle_video(run_id, payload, step, extra, scope)
        elif kind == "artifact":
            self._handle_artifact(run_id, payload)
        elif kind == "histogram":
            self._handle_histogram(run_id, payload, step, extra, scope)
        elif kind == "vector_field":
            self._handle_vector_field(run_id, payload, step, extra, scope)
        elif kind == "trajectories":
            self._handle_trajectories(run_id, payload, step, extra, scope)
        else:
            self._logger.warning("Unsupported event kind: %s", kind)

    # -- handlers ---------------------------------------------------------

    def _handle_metric(
        self,
        run_id: str,
        payload: Any,
        step: int | None,
        extra: dict[str, Any],
        scope: str,
        timestamp_ms: int,
    ) -> None:
        if not isinstance(payload, Mapping):
            raise ValueError(
                "Metric event payload must be a mapping of name->value."
            )
        # Numeric extras (e.g. a custom step axis) are logged as metrics
        # too; non-numeric routing metadata is ignored.
        merged = {**payload, **extra}
        metrics: list[Metric] = []
        for key, value in merged.items():
            numeric = _to_metric_value(value)
            if numeric is None:
                continue
            metrics.append(
                Metric(
                    key=key,
                    value=numeric,
                    timestamp=timestamp_ms,
                    step=step or 0,
                )
            )
        if not metrics:
            self._logger.warning(
                "Skipping metric log with no numeric values (scope=%s).",
                scope,
            )
            return
        self._client_or_raise().log_batch(run_id, metrics=metrics)

    def _handle_image(
        self,
        run_id: str,
        payload: Any,
        step: int | None,
        extra: dict[str, Any],
        scope: str,
    ) -> None:
        key_name = extra.get("name", "image")
        items = (
            payload.items()
            if isinstance(payload, Mapping)
            else [(key_name, payload)]
        )
        for leaf, value in items:
            if value is None:
                self._logger.warning(
                    "Skipping image '%s' with None payload (scope=%s).",
                    leaf,
                    scope,
                )
                continue
            self._log_image(run_id, np.asarray(value), leaf, step)

    def _handle_video(
        self,
        run_id: str,
        payload: Any,
        step: int | None,
        extra: dict[str, Any],
        scope: str,
    ) -> None:
        if payload is None:
            self._logger.warning(
                "Skipping video with None payload (scope=%s).", scope
            )
            return
        name = extra.get("name", "video")
        fps = int(extra.get("fps", 20))
        fmt = str(extra.get("format", "gif"))
        if fmt not in {"gif", "mp4"}:
            self._logger.warning(
                "Unsupported video format '%s' for '%s'; using 'gif'.",
                fmt,
                name,
            )
            fmt = "gif"
        frames = _video_to_channels_last(payload)
        suffix = "" if step is None else f"_step_{step}"
        with tempfile.TemporaryDirectory(prefix="goggles-mlflow-") as tmp:
            out = Path(tmp) / f"{name}{suffix}.{fmt}"
            if fmt == "gif":
                save_numpy_gif(frames, str(out), fps=fps)
            else:
                save_numpy_mp4(frames, out, fps=fps)
            self._client_or_raise().log_artifact(
                run_id, str(out), artifact_path=f"videos/{name}"
            )

    def _handle_artifact(self, run_id: str, payload: Any) -> None:
        """Upload an artifact event's file or directory to the run.

        Payload schema (mirrors the W&B handler, minus W&B-only fields):

        - ``path`` (str, required): file or directory to upload. A
          directory is uploaded recursively via ``log_artifacts``
          (useful for multi-shard checkpoints); a file via
          ``log_artifact``. Non-existent paths are skipped with a warning.
        - ``name`` (str, optional): destination sub-path inside the run's
          artifact tree. Defaults to the artifact root.

        Args:
            run_id: Active MLflow run for the artifact's scope.
            payload: Mapping describing the artifact (see schema above).
        """
        if not isinstance(payload, Mapping):
            self._logger.warning(
                "Artifact payload must be a mapping; got %r", type(payload)
            )
            return
        raw_path = payload.get("path")
        if not isinstance(raw_path, str):
            self._logger.warning(
                "Artifact missing valid 'path' field; skipping."
            )
            return
        path = Path(raw_path)
        if not path.exists():
            self._logger.warning(
                "Artifact path does not exist: %s; skipping.", raw_path
            )
            return
        artifact_path = payload.get("name")
        client = self._client_or_raise()
        if path.is_dir():
            client.log_artifacts(run_id, str(path), artifact_path=artifact_path)
        else:
            client.log_artifact(run_id, str(path), artifact_path=artifact_path)

    def _handle_histogram(
        self,
        run_id: str,
        payload: Any,
        step: int | None,
        extra: dict[str, Any],
        scope: str,
    ) -> None:
        name = extra.get("name", "histogram")
        bins = int(
            extra.get("num_bins", extra.get("bins", _DEFAULT_HISTOGRAM_BINS))
        )
        if not isinstance(payload, (Sequence, np.ndarray)):
            self._logger.warning(
                "Invalid histogram payload for '%s' (scope=%s): "
                "must be a sequence or array.",
                name,
                scope,
            )
            return
        # Mirror the W&B handler: a bad single payload warns and returns
        # rather than propagating (the bus would otherwise log a full
        # traceback). Logging errors are left to propagate, like metrics.
        try:
            fig = _build_histogram_figure(np.asarray(payload), bins, name)
        except Exception as exc:
            self._logger.warning(
                "Invalid histogram payload for '%s' (scope=%s): %s",
                name,
                scope,
                exc,
            )
            return
        leaf = name if step is None else f"{name}/step_{step}"
        self._client_or_raise().log_figure(
            run_id, fig, artifact_file=f"histograms/{leaf}.png"
        )

    def _handle_vector_field(
        self,
        run_id: str,
        payload: Any,
        step: int | None,
        extra: dict[str, Any],
        scope: str,
    ) -> None:
        name = extra.get("name", "vector_field")
        mode = str(extra.get("mode", "magnitude"))
        add_colorbar = bool(extra.get("add_colorbar", False))
        if mode not in {"vorticity", "magnitude"}:
            self._logger.warning(
                "Unknown vector field visualization mode '%s'. Supported "
                "modes are: 'vorticity', 'magnitude'. Skipping.",
                mode,
            )
            return
        mode_literal = cast(Literal["vorticity", "magnitude"], mode)
        items = (
            payload.items()
            if isinstance(payload, Mapping)
            else [(name, payload)]
        )
        for leaf, value in items:
            if value is None:
                self._logger.warning(
                    "Skipping vector field '%s' with None payload (scope=%s).",
                    leaf,
                    scope,
                )
                continue
            try:
                image = create_numpy_vector_field_visualization(
                    value, mode=mode_literal, add_colorbar=add_colorbar
                )
            except Exception as exc:
                self._logger.warning(
                    "Invalid vector field payload for '%s' (scope=%s): %s",
                    leaf,
                    scope,
                    exc,
                )
                continue
            self._log_image(run_id, image, leaf, step)

    def _handle_trajectories(
        self,
        run_id: str,
        payload: Any,
        step: int | None,
        extra: dict[str, Any],
        scope: str,
    ) -> None:
        name = extra.get("name", "trajectories")
        items = (
            payload.items()
            if isinstance(payload, Mapping)
            else [(name, payload)]
        )
        for leaf, value in items:
            if value is None:
                self._logger.warning(
                    "Skipping trajectories '%s' with None payload (scope=%s).",
                    leaf,
                    scope,
                )
                continue
            try:
                image = create_numpy_trajectories_visualization(value)
            except Exception as exc:
                self._logger.warning(
                    "Invalid trajectories payload for '%s' (scope=%s): %s",
                    leaf,
                    scope,
                    exc,
                )
                continue
            self._log_image(run_id, image, leaf, step)

    # -- helpers ----------------------------------------------------------

    def _log_image(
        self, run_id: str, image: np.ndarray, key: str, step: int | None
    ) -> None:
        """Log a single RGB(A) image to the run's stepped image viewer.

        Args:
            run_id: Active MLflow run id.
            image: Image array (H, W) / (H, W, C).
            key: Image key (groups frames across steps in the UI).
            step: Step index, or None.
        """
        self._client_or_raise().log_image(
            run_id, image, key=key, step=step or 0
        )

    @staticmethod
    def _timestamp_ms(event: Any) -> int:
        """Return the event's log time in integer milliseconds.

        Uses ``event.time`` (POSIX seconds) when present, otherwise the
        current wall-clock time.

        Args:
            event: The Goggles event.

        Returns:
            Log time in integer milliseconds since the Unix epoch.
        """
        t = getattr(event, "time", None)
        if t is None:
            t = time.time()
        return int(t * 1000)

    def _ensure_client(self) -> None:
        """Create the client and resolve the experiment id, once."""
        if self._client is None:
            self._client = MlflowClient(tracking_uri=self._tracking_uri)
        if self._experiment_id is None:
            self._experiment_id = self._resolve_experiment_id()

    def _client_or_raise(self) -> MlflowClient:
        """Return the live client, creating it if needed.

        Returns:
            The MLflow client for this handler.
        """
        self._ensure_client()
        assert self._client is not None
        return self._client

    def _resolve_experiment_id(self) -> str:
        """Resolve (creating if needed) the configured experiment id.

        Returns:
            The MLflow experiment id to create runs under.
        """
        assert self._client is not None
        name = self._experiment or _DEFAULT_EXPERIMENT
        experiment = self._client.get_experiment_by_name(name)
        if experiment is not None:
            return experiment.experiment_id
        return self._client.create_experiment(
            name, artifact_location=self._artifact_location
        )

    def _get_or_create_run(self, scope: str) -> str:
        """Get or create the MLflow run id for the given scope.

        Args:
            scope: The scope to get or create a run for.

        Returns:
            The MLflow run id associated with the scope.
        """
        run_id = self._runs.get(scope)
        if run_id is not None:
            return run_id
        client = self._client_or_raise()
        assert self._experiment_id is not None  # set by _ensure_client
        run_name = (
            self._base_run_name
            if scope == self.GLOBAL_SCOPE and self._base_run_name
            else f"{self._base_run_name or 'run'}-{scope}"
        )
        tags = {**self._tags, "goggles.scope": scope}
        run = client.create_run(
            experiment_id=self._experiment_id,
            run_name=run_name,
            tags=tags,
        )
        run_id = run.info.run_id
        if self._params:
            client.log_batch(
                run_id,
                params=[
                    Param(key, str(value))
                    for key, value in self._params.items()
                ],
            )
        self._runs[scope] = run_id
        return run_id

    # -- serialization ----------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize the handler for attachment across the bus.

        Returns:
            The dictionary representation of the handler.
        """
        return {
            "cls": self.__class__.__name__,
            "data": {
                "experiment": self._experiment,
                "tracking_uri": self._tracking_uri,
                "artifact_location": self._artifact_location,
                "run_name": self._base_run_name,
                "params": self._params,
                "tags": self._tags,
                "name": self.name,
            },
        }

    @classmethod
    def from_dict(cls, serialized: dict) -> Self:
        """Reconstruct a handler from its serialized representation.

        Args:
            serialized: The dictionary representation of the handler.

        Returns:
            The reconstructed MLflowHandler instance.
        """
        data = serialized.get("data", serialized)
        return cls(
            experiment=data.get("experiment"),
            tracking_uri=data.get("tracking_uri"),
            artifact_location=data.get("artifact_location"),
            run_name=data.get("run_name"),
            params=data.get("params"),
            tags=data.get("tags"),
            name=data.get("name", "mlflow"),
        )
