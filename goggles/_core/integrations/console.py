"""Console-based log handler for EventBus integration."""

import logging
from pathlib import Path
from typing import ClassVar, Literal
from typing_extensions import Self

from goggles.types import Event, Kind


class ConsoleHandler:
    """Handle 'log' events and output them to console using Python's logging API.

    Attributes:
        name: Stable handler identifier.
        capabilities: Supported event kinds (only {"log"}).

    """

    name: str = "goggles.console"
    capabilities: ClassVar[frozenset[Kind]] = frozenset({"log"})

    def __init__(
        self,
        *,
        name: str = "goggles.console",
        level: int = logging.NOTSET,
        path_style: Literal["absolute", "relative"] = "relative",
        project_root: Path | None = None,
    ) -> None:
        """Initialize the ConsoleHandler.

        Args:
            name: Stable handler identifier.
            level: Minimum log level to handle.
            path_style: Whether to print absolute
                or relative file paths. Defaults to "relative".
            project_root: Root path used for relative paths.

        """
        self.name = name
        self.level = int(level)
        self.path_style = path_style
        self.project_root = Path(project_root or Path.cwd())
        self._logger: logging.Logger

    def can_handle(self, kind: Kind) -> bool:
        """Return whether this handler can process the given kind.

        Args:
            kind: The event kind to check.

        Returns:
            True if the handler can process the kind, False otherwise.
        """
        return kind in self.capabilities

    def handle(self, event: Event) -> None:
        """Forward a log event to Python's logging system.

        Args:
            event: The log event to handle.
        """
        if event.kind != "log":
            raise ValueError(f"Unsupported event kind '{event.kind}'")

        level = int(event.level) if event.level else logging.NOTSET
        message = str(event.payload)

        # Derive display path
        path = Path(event.filepath)
        if self.path_style == "relative":
            try:
                path = path.relative_to(self.project_root)
            except ValueError:
                pass  # fallback to absolute if outside root
        path_str = f"{path}:{event.lineno}"

        # ANSI color codes for different log levels
        level_name = logging.getLevelName(level)
        color_codes = {
            "DEBUG": "\033[36m",  # Cyan
            "INFO": "\033[34m",  # Blue
            "WARNING": "\033[33m",  # Yellow
            "ERROR": "\033[31m",  # Red
            "CRITICAL": "\033[91m",  # Bright Red
        }
        reset_code = "\033[0m"  # Reset color

        color = color_codes.get(level_name, "")
        colored_message = f"{color}{path_str} - {message}{reset_code}"

        # We manually construct prefix since stacklevel=3 may mislead
        self._logger.log(level, colored_message, stacklevel=2)

    def open(self) -> None:
        """Initialize the handler (create logger and formatter)."""
        self._logger = logging.getLogger(self.name)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            self._logger.addHandler(handler)
        self._logger.setLevel(self.level or logging.INFO)

    def close(self) -> None:
        """Flush and release console handler resources."""
        for handler in self._logger.handlers:
            handler.flush()

    def to_dict(self) -> dict:
        """Serialize the handler for later reconstruction."""
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
        """Reconstruct a handler from its serialized representation."""
        data = serialized.get("data", serialized)
        return cls(
            name=data["name"],
            level=data["level"],
            path_style=data.get("path_style", "relative"),
            project_root=Path(data.get("project_root", Path.cwd())),
        )
