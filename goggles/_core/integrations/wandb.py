"""WandB handler config container (no-op).

In the simplified goggles branch, handlers are inert objects that only store
their configuration. `gg.attach(...)` will read these values and set global
defaults used by the simplified loggers (which will own the actual W&B runs).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, TypeAlias

from typing_extensions import Self

Reinit: TypeAlias = Literal[
    "default", "return_previous", "finish_previous", "create_new"
]


class WandBHandler:
    """No-op WandB handler that only stores configuration."""

    def __init__(
        self,
        project: str | None = None,
        entity: str | None = None,
        run_name: str | None = None,
        config: Mapping[str, Any] | None = None,
        group: str | None = None,
        reinit: Reinit = "create_new",
    ) -> None:
        """Initialize the WandB handler.

        Args:
            project: The W&B project name to log to.
            entity: The W&B entity (user or team) to log under.
            run_name: The name of the W&B run.
            config: A dictionary of configuration values to log with the run.
            group: The W&B group to associate this run with.
            reinit: How to handle existing runs when initializing a new run. Options are:
                - "default": Use W&B's default behavior (which may vary based on context).
                - "return_previous": If an active run exists, return it instead of creating a new one.
                - "finish_previous": If an active run exists, finish it before creating a new one.
                - "create_new": Always create a new run, even if an active run exists (this may lead to multiple active runs).

        Raises:
            ValueError: If `reinit` is not one of the valid options.
        """
        valid_reinit: set[str] = {
            "finish_previous",
            "return_previous",
            "create_new",
            "default",
        }
        if reinit not in valid_reinit:
            raise ValueError(
                f"Invalid reinit value '{reinit}'. Must be one of: "
                f"{', '.join(sorted(valid_reinit))}."
            )

        # Store inputs as attributes
        self.project = project
        self.entity = entity
        self.group = group
        self.run_name = run_name
        self.config = dict(config) if config is not None else {}
        self.reinit: Reinit = reinit

    def open(self) -> None:
        """Open the WandB handler. No-op for this implementation."""
        return None

    def close(self) -> None:
        """Close the WandB handler. No-op for this implementation."""
        return None

    def to_dict(self) -> dict:
        """Serialize the handler for later reconstruction.

        Returns:
            A dictionary containing the handler's configuration.
        """
        return {
            "cls": self.__class__.__name__,
            "data": {
                "project": self.project,
                "entity": self.entity,
                "run_name": self.run_name,
                "config": self.config,
                "reinit": self.reinit,
                "group": self.group,
            },
        }

    @classmethod
    def from_dict(cls, serialized: dict) -> Self:
        """Reconstruct a handler from its serialized representation.

        Args:
            serialized: A dictionary containing the handler's configuration.

        Returns:
            An instance of `WandBHandler` initialized with the provided configuration.
        """
        data = serialized.get("data", serialized)
        return cls(
            project=data.get("project"),
            entity=data.get("entity"),
            run_name=data.get("run_name"),
            config=data.get("config"),
            reinit=data.get("reinit", "create_new"),
            group=data.get("group"),
        )
