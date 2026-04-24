"""WandB integration handler for Goggles logging framework."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Literal, TypeAlias, cast

import numpy as np
from typing_extensions import Self

import wandb
from goggles.media import (
    create_numpy_trajectories_visualization,
    create_numpy_vector_field_visualization,
)
from goggles.types import Kind

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
        """
        scope = getattr(event, "scope", None) or self.GLOBAL_SCOPE
        kind = getattr(event, "kind", None) or "metric"
        step = getattr(event, "step", None)
        payload = getattr(event, "payload", None)
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
            run.log(logs, step=step)

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
            run.log(logs, step=step)

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
            run.log(logs, step=step)

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
                image = create_numpy_trajectories_visualization(value)
                logs[field_name] = wandb.Image(image)
            except Exception as exc:
                self._logger.warning(
                    f"Invalid trajectories payload for '{field_name}' "
                    f"(scope={scope}): {exc}",
                )

        for k, v in extra.items():
            logs[k] = v
        if logs:
            run.log(logs, step=step)

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
