"""WandB integration handler for Goggles logging framework."""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Literal
from collections.abc import Mapping, Sequence
import numpy as np
from typing_extensions import Self

import wandb

from goggles.types import Kind

Run = Any  # wandb.sdk.wandb_run.Run
Reinit = Literal["default", "return_previous", "finish_previous", "create_new"]


class WandBHandler:
    """Forward Goggles events to W&B runs (supports concurrent scopes).

    Each scope corresponds to a distinct W&B run that remains active until
    explicitly closed. Compatible with the `Handler` protocol used by the
    EventBus.

    Attributes:
        name: Stable handler identifier.
        capabilities: Supported event kinds
            ({"metric", "image", "video", "artifact"}).

    """

    name: str = "wandb"
    capabilities: ClassVar[frozenset[Kind]] = frozenset(
        {"metric", "image", "video", "artifact", "histogram"}
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
                One of {"finish_previous", "return_previous", "create_new", "default"}.

        """
        self._logger = logging.getLogger(self.name)
        self._logger.propagate = True
        valid_reinit = {"finish_previous", "return_previous", "create_new", "default"}
        if reinit not in valid_reinit:
            raise ValueError(
                f"Invalid reinit value '{reinit}'. Must be one of: "
                f"{', '.join(valid_reinit)}."
            )

        self._project = project
        self._entity = entity
        self._group = group
        self._base_run_name = run_name
        self._config: dict[str, Any] = dict(config) if config is not None else {}
        self._reinit = reinit or "finish_previous"
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
        """Initialize the global W&B run."""
        if self._wandb_run is not None:
            return
        self._wandb_run = wandb.init(
            project=self._project,
            entity=self._entity,
            name=self._base_run_name,
            config=self._config,
            reinit=self._reinit,  # type: ignore
            group=self._group,
        )
        self._runs[self.GLOBAL_SCOPE] = self._wandb_run
        self._current_scope = self.GLOBAL_SCOPE

    def handle(self, event: Any) -> None:
        """Process a Goggles event and forward it to W&B.

        Args:
            event: The Goggles event to process.

        """
        scope = getattr(event, "scope", None) or self.GLOBAL_SCOPE
        kind = getattr(event, "kind", None) or "metric"
        step = getattr(event, "step", None)
        payload = getattr(event, "payload", None)
        extra = getattr(event, "extra", {}) or {}

        run = self._get_or_create_run(scope)

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
            # Preferred key name comes from event.extra["name"], else "image"/"video"
            default_key = "image" if kind == "image" else "video"
            key_name = extra.pop("name", default_key)

            # Allow payload to be either a mapping {name: data} or a single datum
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
                            "Unsupported video format '%s' for '%s'; defaulting to 'mp4'.",
                            fmt,
                            name,
                        )
                        fmt = "mp4"
                    logs[name] = wandb.Video(value, fps=fps, format=fmt)  # type: ignore
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
                self._logger.warning("Artifact missing valid 'path' field; skipping.")
                return
            artifact = wandb.Artifact(name=name, type=art_type, metadata=extra)
            artifact.add_file(path)
            run.log_artifact(artifact)
            return

        if kind == "histogram":
            # Support a vector of samples or an (hist, bin_edges) tuple via `np_hist`.
            # Name defaults to "histogram" unless provided in extra.
            name = extra.pop("name", "histogram")
            static = extra.pop("static", False)
            # `num_bins` can be overridden through extra; default to 64 like W&B.
            num_bins = int(
                extra.pop("num_bins", extra.pop("bins", 64))
            )  # TODO: check if bins is needed

            logs: dict[str, Any] = {}

            try:
                if not isinstance(payload, (Sequence, np.ndarray)):
                    self._logger.warning(
                        "Invalid histogram payload for '%s' (scope=%s): must be a sequence or tuple.",
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

            except Exception as exc:  # noqa: BLE001
                self._logger.warning(
                    "Invalid histogram payload for '%s' (scope=%s): %s",
                    name,
                    scope,
                    exc,
                )
                return
            for k, v in extra.items():
                logs[k] = v

            if logs:
                run.log(logs, step=step)
            return

        self._logger.warning("Unsupported event kind: %s", kind)

    def close(self) -> None:
        """Finish all active W&B runs."""
        for scope, run in list(self._runs.items()):
            if run is not None:
                try:
                    run.finish()
                except:
                    pass
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

    def _get_or_create_run(self, scope: str) -> Run:
        """Get or create a W&B run for the given scope.

        Args:
            scope: The scope for which to get or create the W&B run.

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
            config={**self._config, "scope": scope},
            group=self._group,
            reinit=self._reinit,  # type: ignore
        )
        self._runs[scope] = run
        return run
