"""WandB integration handler for Goggles logging framework."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Literal, TypeAlias, cast

import numpy as np
import plotly.graph_objects as go
from typing_extensions import Self

import wandb
from goggles.media import create_numpy_vector_field_visualization
from goggles.types import Kind


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

    def __init__(
        self,
        project: str | None = None,
        entity: str | None = None,
        run_name: str | None = None,
        config: Mapping[str, Any] | None = None,
        group: str | None = None,
        reinit: Reinit = "create_new",
    ) -> None:
        """Initialize the W&B handler.

        Args:
            project: W&B project name.
            entity: W&B entity (user or team) name.
            run_name: Base name for W&B runs.
            config: Configuration dictionary to log with the run(s).
            group: W&B group name for runs.
            reinit: W&B reinitialization strategy when opening runs.
                One of:
                {"finish_previous", "return_previous", "create_new", "default"}.

        Raises:
            ValueError: If `reinit` is not a valid option.
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

        self._project = project
        self._entity = entity
        self._group = group
        self._base_run_name = run_name
        self._config: dict[str, Any] = (
            dict(config) if config is not None else {}
        )
        self._reinit: Reinit = reinit or "finish_previous"
        self._runs: dict[str, Run] = {}
        self._wandb_run: Run | None = None
        self._current_scope: str | None = None

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

        Raises:
            ValueError: If the event kind is unsupported or if the payload is
                malformed for the given kind.
        """
        scope = getattr(event, "scope", None) or self.GLOBAL_SCOPE
        kind = getattr(event, "kind", None) or "metric"
        step = getattr(event, "step", None)
        payload = getattr(event, "payload", None)
        # Copy so our .pop(...) calls don't mutate the shared event.extra
        # that other handlers on the same bus will read afterwards.
        extra = dict(getattr(event, "extra", {}) or {})
        extra_config = extra.pop("config_wandb", {})

        # Get or create the W&B run for the given scope
        run = self._get_or_create_run(scope, extra_config)

        if kind == "metric":
            if not isinstance(payload, Mapping):
                raise ValueError(
                    "Metric event payload must be a mapping of name→value."
                )
            payload = {k: v for k, v in payload.items() if v is not None}
            if not payload:
                self._logger.warning(
                    "Skipping metric log with empty payload (scope=%s).", scope
                )
                return
            for k, v in extra.items():
                payload[k] = v
            run.log(payload, step=step)
            return

        if kind in {"image", "video"}:
            # Preferred key name comes from event.extra["name"],
            # else "image"/"video"
            key_name = extra.pop("name", kind)

            # Allow payload to be either a mapping {name: data}
            # or a single datum
            items = (
                payload.items()
                if isinstance(payload, Mapping)
                else [(key_name, payload)]
            )
            logs = {}
            for name, value in items:
                if value is None:
                    self._logger.warning(
                        "Skipping %s '%s' with None payload (scope=%s).",
                        kind,
                        name,
                        scope,
                    )
                    continue
                if kind == "image":
                    logs[name] = wandb.Image(value)
                else:
                    fps = int(extra.get("fps", 20))
                    fmt = str(extra.get("format", "mp4"))
                    if fmt not in {"mp4", "gif"}:
                        self._logger.warning(
                            "Unsupported video format '%s' for '%s'; "
                            "defaulting to 'mp4'.",
                            fmt,
                            name,
                        )
                        fmt = "mp4"
                    new_value = self._prepare_video_for_wandb(value)
                    logs[name] = wandb.Video(new_value, fps=fps, format=fmt)  # pyright: ignore[reportArgumentType]
            # Add the extra fields to the logged object
            for k, v in extra.items():
                logs[k] = v

            if logs:
                # Use a single API across kinds for consistency
                run.log(logs, step=step)
            return

        if kind == "artifact":
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
            return

        if kind == "histogram":
            # Support a vector of samples or an (hist, bin_edges) tuple
            # via `np_hist`.
            # Name defaults to "histogram" unless provided in extra.
            name = extra.pop("name", "histogram")
            static = extra.pop("static", False)
            # `num_bins` can be overridden through extra; default to 64 like W&B
            num_bins = int(
                extra.pop("num_bins", extra.pop("bins", 64))
            )  # TODO: check if bins is needed

            logs: dict[str, Any] = {}

            try:
                if not isinstance(payload, (Sequence, np.ndarray)):
                    self._logger.warning(
                        "Invalid histogram payload for '%s' (scope=%s): "
                        "must be a sequence or tuple.",
                        name,
                        scope,
                    )
                    return

                if static:
                    payload_list = list(payload)
                    data = [[v] for v in payload_list]
                    table = wandb.Table(data=data, columns=["values"])
                    # Treat payload as a 1D sequence of samples.
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
                run.log(logs, step=step)
            return

        if kind == "vector_field":
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

            logs = {}
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
                run.log(logs, step=step)
            return

        if kind == "trajectories":
            name = extra.pop("name", "trajectories")

            logs = {}
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
                run.log(logs, step=step)
            return

        self._logger.warning("Unsupported event kind: %s", kind)

    def close(self) -> None:
        """Finish all active W&B runs."""
        for run in list(self._runs.values()):
            if run is not None:
                try:
                    run.finish()
                except Exception as exc:
                    self._logger.warning("Failed to finish W&B run: %s", exc)
        self._runs.clear()
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
        return cls(
            project=serialized.get("project"),
            entity=serialized.get("entity"),
            run_name=serialized.get("run_name"),
            config=serialized.get("config"),
            reinit=serialized.get("reinit", "create_new"),
            group=serialized.get("group"),
        )

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
            reinit=self._reinit,
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
