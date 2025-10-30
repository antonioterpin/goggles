"""JSONL integration for Goggles logging framework."""

import json
import logging
import threading
from pathlib import Path
from typing import ClassVar, FrozenSet

from goggles.types import Event, Kind


class JsonlHandler:
    """Write log events to a JSONL file (UTF-8, one JSON per line).

    Thread-safe and line-buffered, ensuring atomic writes per event.

    Attributes:
        name (str): Stable handler identifier.
        capabilities (set[str]): Supported event kinds (only {"log"}).

    """

    name: str = "jsonl"
    capabilities: ClassVar[FrozenSet[Kind]] = frozenset({"log"})

    def __init__(self, path: Path, name: str = "jsonl") -> None:
        """Initialize the handler.

        Args:
            path (Path): Path to the JSONL output file.
            name (str): Handler identifier (for logging diagnostics).

        """
        self._path = path
        self._fp = None
        self._lock = threading.Lock()

    def open(self) -> None:
        """Open the JSONL file for appending (line-buffered)."""
        self._fp = open(self._path, "a", encoding="utf-8", buffering=1)

    def close(self) -> None:
        """Flush and close the JSONL file."""
        if self._fp and not self._fp.closed:
            with self._lock:
                self._fp.flush()
                self._fp.close()

    def can_handle(self, kind: Kind) -> bool:
        """Return True if this handler supports the given event kind.

        Args:
            kind (Kind): Kind of event ("log", "metric", "image", "artifact").

        Returns:
            bool: True if the kind is supported, False otherwise.

        """
        return kind in self.capabilities

    def handle(self, event: Event) -> None:
        """Write a single event to the JSONL file.

        Args:
            event (Event): The event to serialize.

        Raises:
            ValueError: If the event kind is not supported.
            RuntimeError: If the handler is not opened.

        """
        if not self.can_handle(event.kind):
            raise ValueError(f"Unsupported event kind: {event.kind}")
        if self._fp is None or self._fp.closed:
            raise RuntimeError("Handler not opened. Call open() first.")

        payload = {
            "kind": event.kind,
            "scope": event.scope,
            "payload": event.payload,
            "level": event.level,
            "step": event.step,
            "time": event.time,
            "filepath": event.filepath,
            "lineno": event.lineno,
        }

        if hasattr(event, "extra") and event.extra is not None:
            payload["extra"] = event.extra
        try:
            with self._lock:
                json.dump(payload, self._fp, ensure_ascii=False)
                self._fp.write("\n")
        except Exception:
            logging.getLogger(self.name).exception("Failed to write JSONL event")
