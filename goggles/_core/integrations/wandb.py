"""WandB integration handler for Goggles logging framework."""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional, Set

import wandb


class WandBHandler:
    """Forward metric, image, and video events to Weights & Biases.

    Event compatibility with CoreGogglesLogger:
      - Metrics: event.kind == "metric" (or missing), payload = {name: value, ...}
      - Images/Videos: event.kind in {"image", "video"}, payload can be an array-like
        or a mapping {"name": array_like}.

    Attributes:
        name (str): Stable handler identifier.
        capabilities (set[str]): Supported kinds: {"metric", "image", "video"}.

    """

    name: str = "wandb"
    capabilities: Set[str] = frozenset({"metric", "image", "video"})

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
            kind (str): Kind of event ("metric", "image", "video", etc).

        Returns:
            bool: True if kind is supported, False otherwise.

        """
        return kind in self.capabilities

    def handle(self, event: Any) -> None:
        """Handle an event (metric/image/video) and log it to W&B.

        Args:
            event (Event): Event emitted by CoreGogglesLogger.

        Raises:
            ValueError: If a metric payload is not a mapping.

        """
        if not self._wandb_run:
            self._logger.warning("W&B handler not opened; ignoring event.")
            return

        # Robustly infer kind: MagicMock attributes are truthy by default.
        raw_kind = getattr(event, "kind", None)
        kind = raw_kind if isinstance(raw_kind, str) and raw_kind else "metric"

        # Step may be absent or a MagicMock; only pass along int or None.
        raw_step = getattr(event, "step", None)
        step = raw_step if (isinstance(raw_step, int) or raw_step is None) else None

        payload = getattr(event, "payload", None)

        if kind == "metric":
            if not isinstance(payload, Mapping):
                raise ValueError(
                    "Metric event payload must be a mapping of name→value."
                )
            if step is None:
                wandb.log(dict(payload))
            else:
                wandb.log(dict(payload), step=step)
            self._logger.debug("Logged metrics to W&B: %s", list(payload.keys()))
            return

        if kind in {"image", "video"}:
            # Normalize to {name: data}
            if isinstance(payload, Mapping):
                items = payload.items()
            else:
                items = [("data", payload)]

            logs: dict[str, Any] = {}
            for name, value in items:
                if kind == "image":
                    logs[name] = wandb.Image(value)
                else:
                    logs[name] = wandb.Video(value, fps=20, format="mp4")

            if step is None:
                wandb.log(logs)
            else:
                wandb.log(logs, step=step)
            self._logger.debug("Logged %s(s) to W&B: %s", kind, list(logs.keys()))
            return

        self._logger.warning("Unsupported event kind: %s", kind)
