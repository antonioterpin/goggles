"""Console handler config container (no-op).

In the simplified goggles branch, handlers are inert objects that only store
their configuration. `gg.attach(...)` will read these values and set global
defaults used by the simplified loggers.
"""

import logging
from pathlib import Path
from typing import Literal

from typing_extensions import Self


class ConsoleHandler:
    """No-op console handler that only stores configuration."""

    def __init__(
        self,
        *,
        name: str = "goggles.console",
        level: int = logging.NOTSET,
        path_style: Literal["absolute", "relative"] = "relative",
        project_root: Path | None = None,
    ) -> None:
        """Initialize the console handler.

        Args:
            name: The name of the console handler.
            level: The logging level for the console handler.
            path_style: Whether to display absolute or relative file paths in log messages.
            project_root: The root directory to use for relative paths (if path_style is "relative").
        """
        self.name = name
        self.level = int(level)
        self.path_style = path_style
        self.project_root = Path(project_root or Path.cwd())

    def open(self) -> None:
        """Open the console handler. No-op for this implementation."""
        return None

    def close(self) -> None:
        """Close the console handler. No-op for this implementation."""
        return None

    def to_dict(self) -> dict:
        """Serialize the handler for later reconstruction.

        Returns:
            A dictionary containing the handler's configuration.
        """
        return {
            "cls": self.__class__.__name__,
            "data": {
                "name": self.name,
                "level": self.level,
                "path_style": self.path_style,
                "project_root": str(self.project_root),
            },
        }

    @classmethod
    def from_dict(cls, serialized: dict) -> Self:
        """Reconstruct a handler from its serialized representation.

        Args:
            serialized: A dictionary containing the handler's configuration.

        Returns:
            An instance of ConsoleHandler according to the serialized data.
        """
        data = serialized.get("data", serialized)
        return cls(
            name=data["name"],
            level=data["level"],
            path_style=data.get("path_style", "relative"),
            project_root=Path(data.get("project_root", Path.cwd())),
        )
