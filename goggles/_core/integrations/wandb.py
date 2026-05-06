"""WandB integration handler for Goggles logging framework."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Literal, TypeAlias, cast

import numpy as np
import plotly.graph_objects as go
from typing_extensions import Self

import wandb
from goggles.media import create_numpy_vector_field_visualization
from goggles.types import Kind

from ._step_guard import StepGuard


def create_plotly_trajectories_figure(trajectories: np.ndarray) -> Any:
    """Render trajectories as an interactive Plotly figure.

    Each path is drawn as line+marker segments whose color encodes the
    per-step speed ``||x[t+1] - x[t]||``. All trajectories share a
    single trace (separated by NaNs) so the figure has one colorbar.
    Lives in the W&B integration so plotly stays in the ``wandb`` extra.

    Args:
        trajectories: Array of shape ``(N, L, dim)`` with ``dim`` in
            ``{2, 3}`` and ``L >= 2``.

    Returns:
        A ``plotly.graph_objects.Figure``.

    Raises:
        ValueError: If the input shape or dimension is invalid.
    """
    if trajectories.ndim != 3:
        raise ValueError(
            "Trajectories must have shape (N, L, dim); "
            f"got {trajectories.shape}."
        )
    N, L, dim = trajectories.shape
    if dim not in (2, 3):
        raise ValueError(f"Trajectories dim must be 2 or 3; got {dim}.")
    if L < 2:
        raise ValueError(f"Trajectories must have length L >= 2; got {L}.")

    # Per-step speed; pad to L by repeating the last value so each
    # vertex (not just each segment) has a color.
    deltas = np.diff(trajectories, axis=1)
    speed = np.linalg.norm(deltas, axis=-1)
    speed_per_pt = np.concatenate([speed, speed[:, -1:]], axis=1)

    # NaN-separate trajectories so a single Plotly trace renders them
    # as disjoint paths with a shared colorbar.
    nan_pt = np.full((1, dim), np.nan, dtype=trajectories.dtype)
    nan_speed = np.array([np.nan], dtype=speed_per_pt.dtype)
    pts_chunks: list[np.ndarray] = []
    speed_chunks: list[np.ndarray] = []
    for n in range(N):
        pts_chunks.append(trajectories[n])
        speed_chunks.append(speed_per_pt[n])
        if n < N - 1:
            pts_chunks.append(nan_pt)
            speed_chunks.append(nan_speed)
    pts = np.concatenate(pts_chunks, axis=0)
    color = np.concatenate(speed_chunks)

    marker = {
        "color": color,
        "colorscale": "Viridis",
        "cmin": float(np.nanmin(color)),
        "cmax": float(np.nanmax(color)),
        "size": 2,
        "showscale": True,
        "colorbar": {"title": "speed"},
    }
    line = {"color": "rgba(110,110,110,0.45)", "width": 2}

    starts = trajectories[:, 0]
    ends = trajectories[:, -1]
    start_marker = {
        "symbol": "circle-open",
        "size": 6 if dim == 2 else 4,
        "color": "black",
        "line": {"width": 1.2, "color": "black"},
    }
    end_marker = {
        "symbol": "circle",
        "size": 6 if dim == 2 else 4,
        "color": "black",
    }

    if dim == 3:
        trace = go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode="lines+markers",
            line=line,
            marker=marker,
            connectgaps=False,
            showlegend=False,
        )
        starts_trace = go.Scatter3d(
            x=starts[:, 0],
            y=starts[:, 1],
            z=starts[:, 2],
            mode="markers",
            marker=start_marker,
            name="start",
        )
        ends_trace = go.Scatter3d(
            x=ends[:, 0],
            y=ends[:, 1],
            z=ends[:, 2],
            mode="markers",
            marker=end_marker,
            name="end",
        )
        layout = {
            "scene": {
                "xaxis_title": "x",
                "yaxis_title": "y",
                "zaxis_title": "z",
                "aspectmode": "data",
            },
            "margin": {"l": 0, "r": 0, "t": 30, "b": 0},
            "showlegend": True,
            "legend": {"x": 0.0, "y": 1.0},
        }
    else:
        trace = go.Scatter(
            x=pts[:, 0],
            y=pts[:, 1],
            mode="lines+markers",
            line=line,
            marker=marker,
            connectgaps=False,
            showlegend=False,
        )
        starts_trace = go.Scatter(
            x=starts[:, 0],
            y=starts[:, 1],
            mode="markers",
            marker=start_marker,
            name="start",
        )
        ends_trace = go.Scatter(
            x=ends[:, 0],
            y=ends[:, 1],
            mode="markers",
            marker=end_marker,
            name="end",
        )
        layout = {
            "xaxis": {"scaleanchor": "y", "scaleratio": 1, "title": "x"},
            "yaxis": {"title": "y"},
            "margin": {"l": 40, "r": 10, "t": 30, "b": 40},
            "showlegend": True,
            "legend": {"x": 0.0, "y": 1.0},
        }

    return go.Figure(data=[trace, starts_trace, ends_trace], layout=layout)


Run = Any  # wandb.sdk.wandb_run.Run
Reinit: TypeAlias = Literal[
    "default", "return_previous", "finish_previous", "create_new"
]


class WandBHandler:
    """Forward Goggles events to W&B runs (supports concurrent scopes).

    Each scope corresponds to a distinct W&B run that remains active until
    explicitly closed. Compatible with the `Handler` protocol used by the
    EventBus.

    Out-of-order steps (``event.step`` strictly less than the highest step
    previously seen on the same scope) are dropped with a warning, so the
    W&B run timeline is non-decreasing per scope. Events with
    ``step is None`` are forwarded unchanged. Artifact events are
    step-less and bypass the check.

    Attributes:
        name: Stable handler identifier.
        capabilities: Supported event kinds
            ({"metric", "image", "video",
            "artifact", "histogram", "vector_field"}).
        GLOBAL_SCOPE: The default scope name for events w/o an explicit scope.

    """

    name: str = "wandb"
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
    _WANDB_INIT_RESERVED_KEYS: ClassVar[frozenset[str]] = frozenset(
        {"project", "entity", "name", "config", "group", "tags", "reinit"}
    )

    def __init__(
        self,
        project: str | None = None,
        entity: str | None = None,
        run_name: str | None = None,
        config: Mapping[str, Any] | None = None,
        group: str | None = None,
        tags: Sequence[str] | None = None,
        reinit: Reinit = "create_new",
        wandb_init_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize the W&B handler.

        Args:
            project: W&B project name.
            entity: W&B entity (user or team) name.
            run_name: Base name for W&B runs.
            config: Configuration dictionary to log with the run(s).
            group: W&B group name. Use it to keep related runs together
                in the W&B UI; per-scope runs created by this handler
                share the same group.
            tags: W&B tags applied to every run created by this handler.
                Pass an iterable of strings (e.g. ``["baseline", "v2"]``).
                A bare string is rejected because W&B would silently
                iterate it character by character.
            reinit: W&B reinitialization strategy when opening runs.
                One of:
                {"finish_previous", "return_previous", "create_new", "default"}.
            wandb_init_kwargs: Additional keyword arguments forwarded to
                ``wandb.init``.

        Raises:
            ValueError: If `reinit` is not a valid option, or if
                `wandb_init_kwargs` contains invalid or handler-owned keys.
            TypeError: If `tags` is a `str` or contains non-string items.
        """
        self._logger = logging.getLogger(self.name)
        # Ensure that Goggles logs are not propagated to the root logger
        # to avoid duplicates
        self._logger.propagate = False
        valid_reinit: set[str] = {
            "finish_previous",
            "return_previous",
            "create_new",
            "default",
        }
        if reinit not in valid_reinit:
            raise ValueError(
                f"Invalid reinit value '{reinit}'. Must be one of: "
                f"{', '.join(valid_reinit)}."
            )

        if isinstance(tags, str):
            raise TypeError(
                "tags must be a sequence of strings, not a single str. "
                'Wrap it in a list: tags=["my-tag"].'
            )
        normalized_tags: list[str] | None = None
        if tags is not None:
            normalized_tags = list(tags)
            for item in normalized_tags:
                if not isinstance(item, str):
                    raise TypeError(
                        "tags must contain only strings; "
                        f"got {type(item).__name__}."
                    )

        self._project = project
        self._entity = entity
        self._group = group
        self._tags: list[str] | None = normalized_tags
        self._base_run_name = run_name
        self._config: dict[str, Any] = (
            dict(config) if config is not None else {}
        )
        self._reinit: Reinit = reinit or "finish_previous"
        self._wandb_init_kwargs = self._validate_wandb_init_kwargs(
            wandb_init_kwargs
        )
        self._runs: dict[str, Run] = {}
        self._wandb_run: Run | None = None
        self._current_scope: str | None = None
        self._step_guard = StepGuard()
        # Pending writes batched per scope, keyed by step. The W&B cloud
        # backend treats multiple ``run.log({...}, step=N)`` calls at the
        # same N as best-effort merges (a panel can register one key but
        # show "No data available" for sibling keys logged at the same
        # step — see issue #177). Coalesce same-step events into a single
        # commit; a step change or ``close()`` flushes the buffer.
        # ``step is None`` events bypass the buffer to preserve W&B's
        # auto-increment semantics.
        self._pending: dict[str, dict[str, Any]] = {}

    @classmethod
    def _validate_wandb_init_kwargs(
        cls, wandb_init_kwargs: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        """Validate extra ``wandb.init`` kwargs and return a dict copy.

        Args:
            wandb_init_kwargs: Additional keyword arguments to forward to
                ``wandb.init``.

        Returns:
            A copied dictionary of validated keyword arguments.

        Raises:
            ValueError: If a key is unknown to ``wandb.init`` or conflicts
                with parameters managed directly by this handler.
        """
        if wandb_init_kwargs is None:
            return {}

        init_kwargs = dict(wandb_init_kwargs)
        reserved = sorted(
            set(init_kwargs).intersection(cls._WANDB_INIT_RESERVED_KEYS)
        )
        if reserved:
            raise ValueError(
                "wandb_init_kwargs cannot override handler-owned "
                f"wandb.init parameters: {', '.join(reserved)}."
            )

        try:
            signature = inspect.signature(wandb.init)
        except (TypeError, ValueError):
            return init_kwargs

        parameters = signature.parameters
        accepts_var_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        unknown = sorted(
            key
            for key in init_kwargs
            if key not in parameters and not accepts_var_kwargs
        )
        if unknown:
            raise ValueError(
                "Unknown wandb.init keyword argument(s) in "
                f"wandb_init_kwargs: {', '.join(unknown)}."
            )

        return init_kwargs

    def can_handle(self, kind: str) -> bool:
        """Return True if the handler supports this event kind.

        Args:
            kind: Kind of event ("log", "metric", "image", "artifact").

        Returns:
            True if the kind is supported, False otherwise.

        """
        return kind in self.capabilities

    def open(self) -> None:
        """Do nothing. Run is initialized on handle."""

    def handle(self, event: Any) -> None:
        """Process a Goggles event and forward it to W&B.

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

        # Copy so our .pop(...) calls don't mutate the shared event.extra
        # that other handlers on the same bus will read afterwards.
        extra = dict(getattr(event, "extra", {}) or {})
        extra_config = extra.pop("config_wandb", {})

        run = self._get_or_create_run(scope, extra_config)

        if kind == "metric":
            self._handle_metric(run, payload, step, extra, scope)
        elif kind in {"image", "video"}:
            self._handle_media(run, kind, payload, step, extra, scope)
        elif kind == "artifact":
            self._handle_artifact(run, payload, extra)
        elif kind == "histogram":
            self._handle_histogram(run, payload, step, extra, scope)
        elif kind == "vector_field":
            self._handle_vector_field(run, payload, step, extra, scope)
        elif kind == "trajectories":
            self._handle_trajectories(run, payload, step, extra, scope)
        else:
            self._logger.warning("Unsupported event kind: %s", kind)

    def _handle_metric(
        self,
        run: Run,
        payload: Any,
        step: int | None,
        extra: dict[str, Any],
        scope: str,
    ) -> None:
        if not isinstance(payload, Mapping):
            raise ValueError(
                "Metric event payload must be a mapping of name->value."
            )
        payload = {k: v for k, v in payload.items() if v is not None}
        if not payload:
            self._logger.warning(
                "Skipping metric log with empty payload (scope=%s).", scope
            )
            return
        for k, v in extra.items():
            payload[k] = v
        self._stage(run, scope, step, payload)

    def _handle_media(
        self,
        run: Run,
        kind: str,
        payload: Any,
        step: int | None,
        extra: dict[str, Any],
        scope: str,
    ) -> None:
        key_name = extra.pop("name", kind)
        items = (
            payload.items()
            if isinstance(payload, Mapping)
            else [(key_name, payload)]
        )
        logs: dict[str, Any] = {}
        for name, value in items:
            if value is None:
                self._logger.warning(
                    "Skipping %s '%s' with None payload (scope=%s).",
                    kind,
                    name,
                    scope,
                )
                continue
            logs[name] = (
                wandb.Image(value)
                if kind == "image"
                else self._build_wandb_video(name, value, extra)
            )
        for k, v in extra.items():
            logs[k] = v
        if logs:
            self._stage(run, scope, step, logs)

    def _build_wandb_video(
        self, name: str, value: Any, extra: Mapping[str, Any]
    ) -> Any:
        fps = int(extra.get("fps", 20))
        fmt = str(extra.get("format", "mp4"))
        if fmt not in {"mp4", "gif"}:
            self._logger.warning(
                "Unsupported video format '%s' for '%s'; defaulting to 'mp4'.",
                fmt,
                name,
            )
            fmt = "mp4"
        prepared = self._prepare_video_for_wandb(value)
        return wandb.Video(prepared, fps=fps, format=fmt)  # pyright: ignore[reportArgumentType]

    def _handle_artifact(
        self, run: Run, payload: Any, extra: dict[str, Any]
    ) -> None:
        if not isinstance(payload, Mapping):
            self._logger.warning(
                "Artifact payload must be a mapping; got %r", type(payload)
            )
            return
        path = payload.get("path")
        name = payload.get("name", "artifact")
        art_type = payload.get("type", "misc")
        if not isinstance(path, str):
            self._logger.warning(
                "Artifact missing valid 'path' field; skipping."
            )
            return
        artifact = wandb.Artifact(name=name, type=art_type, metadata=extra)
        artifact.add_file(path)
        run.log_artifact(artifact)

    def _handle_histogram(
        self,
        run: Run,
        payload: Any,
        step: int | None,
        extra: dict[str, Any],
        scope: str,
    ) -> None:
        name = extra.pop("name", "histogram")
        static = extra.pop("static", False)
        num_bins = int(extra.pop("num_bins", extra.pop("bins", 64)))

        if not isinstance(payload, (Sequence, np.ndarray)):
            self._logger.warning(
                "Invalid histogram payload for '%s' (scope=%s): "
                "must be a sequence or tuple.",
                name,
                scope,
            )
            return

        logs: dict[str, Any] = {}
        try:
            if static:
                payload_list = list(payload)
                data = [[v] for v in payload_list]
                table = wandb.Table(data=data, columns=["values"])
                logs[name] = wandb.plot.histogram(
                    table, "values", title="Histogram of Random Values"
                )
            else:
                logs[name] = wandb.Histogram(
                    np_histogram=np.histogram(payload, bins=num_bins)
                )
        except Exception as exc:
            self._logger.warning(
                f"Invalid histogram payload for '{name}' "
                f"(scope={scope}): {exc}",
            )
            return

        for k, v in extra.items():
            logs[k] = v
        if logs:
            self._stage(run, scope, step, logs)

    def _handle_vector_field(
        self,
        run: Run,
        payload: Any,
        step: int | None,
        extra: dict[str, Any],
        scope: str,
    ) -> None:
        name = extra.pop("name", "vector_field")
        mode = str(extra.pop("mode", "magnitude"))
        add_colorbar = bool(extra.pop("add_colorbar", False))

        if mode not in {"vorticity", "magnitude"}:
            self._logger.warning(
                f"Unknown vector field visualization mode '{mode}'. "
                "Supported modes are: 'vorticity', 'magnitude'. "
                "The vector field visualization will not be sent to W&B.",
            )
            return
        mode_literal = cast(Literal["vorticity", "magnitude"], mode)

        logs: dict[str, Any] = {}
        items = (
            payload.items()
            if isinstance(payload, Mapping)
            else [(name, payload)]
        )
        for field_name, value in items:
            if value is None:
                self._logger.warning(
                    f"Skipping vector field '{field_name}' with None "
                    f"payload (scope={scope}).",
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
                self._logger.warning(
                    f"Invalid vector field payload for '{field_name}' "
                    f"(scope={scope}): {exc}",
                )

        for k, v in extra.items():
            logs[k] = v
        if logs:
            self._stage(run, scope, step, logs)

    def _handle_trajectories(
        self,
        run: Run,
        payload: Any,
        step: int | None,
        extra: dict[str, Any],
        scope: str,
    ) -> None:
        name = extra.pop("name", "trajectories")

        logs: dict[str, Any] = {}
        items = (
            payload.items()
            if isinstance(payload, Mapping)
            else [(name, payload)]
        )
        for field_name, value in items:
            if value is None:
                self._logger.warning(
                    f"Skipping trajectories '{field_name}' with None "
                    f"payload (scope={scope}).",
                )
                continue
            try:
                fig = create_plotly_trajectories_figure(value)
                logs[field_name] = wandb.Plotly(fig)
            except Exception as exc:
                self._logger.warning(
                    f"Invalid trajectories payload for '{field_name}' "
                    f"(scope={scope}): {exc}",
                )

        for k, v in extra.items():
            logs[k] = v
        if logs:
            self._stage(run, scope, step, logs)

    def close(self) -> None:
        """Finish all active W&B runs."""
        for scope in list(self._pending.keys()):
            self._flush_pending(scope)
        for run in list(self._runs.values()):
            if run is not None:
                try:
                    run.finish()
                except Exception as exc:
                    self._logger.warning("Failed to finish W&B run: %s", exc)
        self._runs.clear()
        self._pending.clear()
        self._wandb_run = None
        self._current_scope = None

    def to_dict(self) -> dict:
        """Serialize the handler for attachment.

        Returns:
            The dictionary representation of the handler.
        """
        return {
            "cls": self.__class__.__name__,
            "data": {
                "project": self._project,
                "entity": self._entity,
                "run_name": self._base_run_name,
                "config": self._config,
                "reinit": self._reinit,
                "group": self._group,
                "tags": list(self._tags) if self._tags is not None else None,
                "wandb_init_kwargs": self._wandb_init_kwargs,
            },
        }

    @classmethod
    def from_dict(cls, serialized: dict) -> Self:
        """De-serialize the handler from its dictionary representation.

        Args:
            serialized: The dictionary representation of the handler.

        Returns:
            The reconstructed WandBHandler instance.
        """
        data = serialized.get("data", serialized)
        return cls(
            project=data.get("project"),
            entity=data.get("entity"),
            run_name=data.get("run_name"),
            config=data.get("config"),
            reinit=data.get("reinit", "create_new"),
            group=data.get("group"),
            tags=data.get("tags"),
            wandb_init_kwargs=data.get("wandb_init_kwargs"),
        )

    def _stage(
        self,
        run: Run,
        scope: str,
        step: int | None,
        logs: Mapping[str, Any],
    ) -> None:
        """Buffer a log payload for ``scope`` at ``step``.

        Same-step events are merged into a single ``run.log`` commit; a
        step change flushes the previous buffer first. ``step is None``
        bypasses the buffer entirely so W&B's auto-increment semantics
        are preserved for callers that don't pin a step.

        Args:
            run: The W&B run for ``scope`` (already created).
            scope: The scope being logged under.
            step: User-supplied step, or None for auto-increment.
            logs: Mapping of keys to values to merge into the commit.
        """
        if step is None:
            self._flush_pending(scope)
            run.log(dict(logs))
            return
        pending = self._pending.get(scope)
        if pending is None:
            pending = {"step": step, "logs": {}}
            self._pending[scope] = pending
        elif pending["step"] != step:
            self._flush_pending(scope)
            pending = self._pending.setdefault(
                scope, {"step": step, "logs": {}}
            )
            pending["step"] = step
        pending["logs"].update(logs)

    def _flush_pending(self, scope: str) -> None:
        """Commit any pending logs for ``scope`` in one ``run.log`` call.

        Args:
            scope: The scope whose pending buffer should be flushed.
        """
        pending = self._pending.get(scope)
        if not pending or not pending["logs"]:
            return
        run = self._runs.get(scope)
        if run is not None:
            run.log(pending["logs"], step=pending["step"])
        pending["logs"] = {}
        pending["step"] = None

    def _get_or_create_run(self, scope: str, extra_config: dict) -> Run:
        """Get or create a W&B run for the given scope.

        Args:
            scope: The scope for which to get or create the W&B run.
            extra_config: Additional config to pass when creating a new run.

        Returns:
            The W&B run associated with the given scope.

        """
        run = self._runs.get(scope)
        if run is not None:
            return run
        name = (
            self._base_run_name
            if scope == self.GLOBAL_SCOPE and self._base_run_name
            else f"{self._base_run_name or 'run'}-{scope}"
        )
        run = wandb.init(
            project=self._project,
            entity=self._entity,
            name=name,
            config={**self._config, "scope": scope, **extra_config},
            group=self._group,
            tags=list(self._tags) if self._tags is not None else None,
            reinit=self._reinit,
            **self._wandb_init_kwargs,
        )
        self._runs[scope] = run
        return run

    def _prepare_video_for_wandb(self, value: np.ndarray) -> np.ndarray:
        """Normalize video tensors to channels-first layout for W&B.

        Accepted shapes:
        - (F, H, W) — implicit grayscale
        - (F, C, H, W) — channels-first
        - (F, H, W, C) — channels-last (C in {1, 3, 4})
        - (F, T, C, H, W) — batched, channels-first

        Args:
            value: The input video tensor.

        Returns:
            Channels-first video, with 3 or 4 channels preserved: shape
            (F, 3|4, H, W) or (F, T, 3|4, H, W). Grayscale inputs are
            repeated to 3 channels.

        Raises:
            ValueError: If the input has an unsupported dimensionality.
        """
        if value.ndim == 3:
            # (F, H, W) → (F, 1, H, W)
            value = value[:, None, :, :]
        elif value.ndim == 4:
            channels_first_like = value.shape[1] in (1, 3, 4)
            channels_last_like = value.shape[-1] in (1, 3, 4)
            if channels_last_like and not channels_first_like:
                # (F, H, W, C) → (F, C, H, W)
                value = np.moveaxis(value, -1, 1)
            elif channels_first_like and channels_last_like:
                # Ambiguous (e.g. W in {1,3,4}); prefer channels-first,
                # which is the documented canonical layout.
                self._logger.warning(
                    "Ambiguous 4D video shape %s: both axis 1 and axis -1 "
                    "look like channel dimensions; preserving channels-first.",
                    value.shape,
                )
            elif not channels_first_like:
                # Neither axis 1 nor axis -1 has a channel-shaped extent
                # (1/3/4). We can't recover the intended layout — refuse
                # rather than silently mislabel the data.
                raise ValueError(
                    f"4D video has shape {value.shape}; expected channel "
                    "dim (size 1, 3, or 4) at axis 1 (channels-first) "
                    "or axis -1 (channels-last)."
                )
        elif value.ndim == 5:
            # (F, T, C, H, W): the channel dim is axis 2.
            if value.shape[2] not in (1, 3, 4):
                raise ValueError(
                    f"5D video has shape {value.shape}; expected channel "
                    "dim (size 1, 3, or 4) at axis 2 for the documented "
                    "(F, T, C, H, W) layout."
                )
        else:
            raise ValueError(
                f"Video has invalid shape {value.shape}; "
                "expected (F,H,W), (F,C,H,W), (F,H,W,C), or (F,T,C,H,W)."
            )

        if value.shape[1] == 1 and value.ndim == 4:
            # Grayscale → RGB
            value = np.repeat(value, 3, axis=1)

        if value.shape[2] == 1 and value.ndim == 5:
            # Grayscale → RGB
            value = np.repeat(value, 3, axis=2)

        return value
