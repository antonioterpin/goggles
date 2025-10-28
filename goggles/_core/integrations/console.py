"""Console integration for Goggles.

This module defines the ConsoleHandler to output log records to the console.
"""

"""Console-based log handler for EventBus integration."""

import logging
from typing import Set

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

    name: str = "console"
    capabilities: Set[str] = frozenset({"log"})

    def __init__(self) -> None:
        """Initialize the console log handler."""
        self._logger = logging.getLogger(self.name)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            self._logger.addHandler(handler)
        self._logger.setLevel(logging.NOTSET)

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
        pass

    def close(self) -> None:
        """Flush and release console handler resources."""
        for handler in self._logger.handlers:
            handler.flush()
