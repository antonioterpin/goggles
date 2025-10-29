"""Console integration for Goggles.

This module defines the ConsoleHandler to output log records to the console.
"""

"""Console-based log handler for EventBus integration."""

import logging
from typing import Set
from typing_extensions import Self

from goggles.types import Event, Kind


class ConsoleHandler:
    """Handle 'log' events and output them to console using Python's logging API.

    This handler is the final sink for textual log events emitted by the
    EventBus. It simply forwards messages to Python's logging system so that
    the reported file and line number correspond to the original caller.

    Attributes:
        name (str): Stable handler identifier.
        capabilities (set[str]): Supported event kinds (only `{"log"}`).

    """

    name: str = "goggles.console"
    capabilities: Set[str] = frozenset({"log"})

    def __init__(self, *, name: str = "goggles.console") -> None:
        """Initialize the ConsoleHandler.

        Args:
            name (str): Stable handler identifier. Defaults to "goggles.console".

        """
        self.name = name
        self._logger: logging.Logger

    def can_handle(self, kind: Kind) -> bool:
        """Return whether this handler can process the given kind.

        Args:
            kind (Kind): Kind of event ("log", "metric", "image", "artifact").

        Returns:
            bool: True if kind == "log", False otherwise.

        """
        return kind == "log"

    def handle(self, event: Event) -> None:
        """Forward a log event to Python's logging system.

        Args:
            event (Event): Event containing textual payload and metadata.

        Raises:
            ValueError: If the event kind is not "log".

        """
        if event.kind != "log":
            raise ValueError(f"Unsupported event kind '{event.kind}'")

        level = event.level or logging.INFO
        message = str(event.payload)
        self._logger.log(level, message, stacklevel=3)

    def open(self) -> None:
        """Initialize the handler (no-op for console)."""
        self._logger = logging.getLogger(self.name)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            self._logger.addHandler(handler)
        self._logger.setLevel(logging.NOTSET)

    def close(self) -> None:
        """Flush and release console handler resources."""
        for handler in self._logger.handlers:
            handler.flush()

    def to_dict(self) -> dict:
        """Serialize the handler.

        This method is needed during attachment. Will be called before binding.

        Returns:
            (dict) A dictionary that allows to instantiate the Handler.
                Must contain:
                    - "cls": The handler class name.
                    - "data": The handler data to be used in from_dict.

        """
        return {"cls": self.__class__.__name__, "data": {"name": self.name}}

    @classmethod
    def from_dict(cls, serialized: dict) -> Self:
        """De-serialize the handler.

        Args:
            serialized (dict): Serialized handler with handler.to_dict

        Returns:
            Self: The Handler instance.

        """
        return cls(name=serialized["name"])
