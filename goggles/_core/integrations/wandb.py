"""WandB integration handler for Goggles logging framework."""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional, Set

import wandb


class WandBHandler:
    """Forward metric, image, video, and artifact events to Weights & Biases.

    Event compatibility with CoreGogglesLogger:
      - Metrics: event.kind == "metric" (or missing), payload = {name: value, ...}
      - Images/Videos: event.kind in {"image", "video"}, payload can be an array-like
        or a mapping {"name": array_like}.
      - Artifacts: event.kind == "artifact", payload = {"path": str, "name": str, "type": str}

    Attributes:
        name (str): Stable handler identifier.
        capabilities (set[str]): Supported kinds: {"metric", "image", "video", "artifact"}.

    """

    name: str = "wandb"
    capabilities: Set[str] = frozenset({"metric", "image", "video", "artifact"})

    def __init__(
        self,
        project: Optional[str] = None,
        entity: Optional[str] = None,
        run_name: Optional[str] = None,
        config: Optional[Mapping[str, Any]] = None,
        reinit: Optional[str] = None,
    ) -> None:
        """Initialize the WandBHandler.

        Args:
            project (Optional[str]): W&B project name.
            entity (Optional[str]): W&B entity name.
            run_name (Optional[str]): Optional run display name.
            config (Optional[Mapping[str, Any]]): Optional run configuration.
            reinit (Optional[str]): W&B reinit mode: "finish_previous", "return_previous",
                "create_new", or None.

        Raises:
            ValueError: If `reinit` is invalid.

        """
        valid_reinit = {"finish_previous", "return_previous", "create_new", None}
        if reinit not in valid_reinit:
            raise ValueError(
                f"Invalid reinit value '{reinit}'. Must be one of: "
                f"{', '.join([r for r in valid_reinit if r is not None])} or None."
            )
        self._logger = logging.getLogger(self.name)
        self._wandb_run: Optional[wandb.sdk.wandb_run.Run] = None
        self._project = project
        self._entity = entity
        self._run_name = run_name
        self._config = dict(config) if config is not None else {}
        self._reinit = reinit

    def open(self) -> None:
        """Initialize the W&B run."""
        if self._wandb_run is not None:
            return
        self._logger.debug("Initializing W&B run.")
        kwargs: dict[str, Any] = {
            "project": self._project,
            "entity": self._entity,
            "name": self._run_name,
            "config": self._config,
        }
        if self._reinit is not None:
            kwargs["reinit"] = self._reinit
        self._wandb_run = wandb.init(**kwargs)

    def close(self) -> None:
        """Finish the W&B run and clean up resources."""
        if self._wandb_run is None:
            return
        self._logger.debug("Closing W&B run.")
        wandb.finish()
        self._wandb_run = None

    def can_handle(self, kind: str) -> bool:
        """Return whether this handler can process the given kind.

        Args:
            kind (str): Kind of event ("metric", "image", "video", "artifact", etc).

        Returns:
            bool: True if kind is supported, False otherwise.

        """
        return kind in self.capabilities

    def handle(self, event: Any) -> None:
        """Handle an event (metric/image/video/artifact) and log it to W&B.

        Args:
            event (Event): Event emitted by CoreGogglesLogger.

        Raises:
            ValueError: If a metric payload is not a mapping.

        """
        if not self._wandb_run:
            self._logger.warning("W&B handler not opened; ignoring event.")
            return

        raw_kind = getattr(event, "kind", None)
        kind = raw_kind if isinstance(raw_kind, str) and raw_kind else "metric"
        raw_step = getattr(event, "step", None)
        step = raw_step if (isinstance(raw_step, int) or raw_step is None) else None
        payload = getattr(event, "payload", None)

        if kind == "metric":
            if not isinstance(payload, Mapping):
                raise ValueError(
                    "Metric event payload must be a mapping of name→value."
                )
            wandb.log(dict(payload), step=step)
            self._logger.debug("Logged metrics to W&B: %s", list(payload.keys()))
            return

        if kind in {"image", "video"}:
            if isinstance(payload, Mapping):
                items = payload.items()
            else:
                items = [("data", payload)]

            logs: dict[str, Any] = {}
            for name, value in items:
                logs[name] = (
                    wandb.Image(value)
                    if kind == "image"
                    else wandb.Video(value, fps=20, format="mp4")
                )

            wandb.log(logs, step=step)
            self._logger.debug("Logged %s(s) to W&B: %s", kind, list(logs.keys()))
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

            artifact = wandb.Artifact(name=name, type=art_type)
            artifact.add_file(path)
            self._wandb_run.log_artifact(artifact)
            self._logger.debug("Uploaded artifact to W&B: %s (%s)", name, path)
            return

        self._logger.warning("Unsupported event kind: %s", kind)
