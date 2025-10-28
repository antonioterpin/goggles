"""WandB integration handler for Goggles logging framework."""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional, Set

import wandb


class WandBHandler:
    """Handler that forwards metric events to Weights & Biases.

    Attributes:
        name (str): Stable handler identifier.
        capabilities (set[str]): Supported event kinds (only {"metric"}).

    """

    name: str = "wandb"
    capabilities: Set[str] = frozenset({"metric"})

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
            reinit (Optional[str]): W&B reinitialization mode. One of:
                "finish_previous", "return_previous", "create_new".
                If None, defaults to W&B’s internal behavior.

        Raises:
            ValueError: If reinit is not one of the accepted values.

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
        self._config = config or {}
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
            # String-based reinit supported in latest W&B
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
        """Return whether this handler can process the given kind."""
        return kind in self.capabilities

    def handle(self, event: Any) -> None:
        """Handle a metric event and log it to W&B.

        Args:
            event (Event): The event containing metrics.

        Raises:
            ValueError: If the event payload is not a mapping.

        """
        if not self._wandb_run:
            self._logger.warning("W&B handler not opened; ignoring event.")
            return

        if not isinstance(event.payload, Mapping):
            raise ValueError("Metric event payload must be a mapping of name→value.")

        step = getattr(event, "step", None)
        metrics = dict(event.payload)
        if step is not None:
            wandb.log(metrics, step=step)
        else:
            wandb.log(metrics)

        self._logger.debug("Logged metrics to W&B: %s", list(metrics.keys()))
