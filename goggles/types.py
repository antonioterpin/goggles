"""Types used in Goggles."""

import numpy as np
from typing import Literal, Any
from dataclasses import dataclass
from typing import TypeAlias

Kind = Literal[
    "log", "metric", "image", "video", "artifact", "histogram", "vector", "vector_field"
]

Metrics = dict[str, float | int]
Image: TypeAlias = np.ndarray
Video: TypeAlias = np.ndarray
Vector: TypeAlias = np.ndarray
VectorField: TypeAlias = np.ndarray


@dataclass(frozen=True)
class Event:
    """Structured event routed through the EventBus.

    Args:
        kind (Kind): Kind of event ("log", "metric", "image", "artifact").
        scope (str): Scope of the event ("global" or "run").
        payload (Any): Event payload.
        filepath (str): File path of the caller emitting the event.
        lineno (int): Line number of the caller emitting the event.
        level (int | None): Optional log level for "log" events.
        step (int | None): Optional global step index.
        time (float | None): Optional global timestamp.
        extra (dict[str, Any] | None): Optional extra metadata.

    """

    kind: Kind
    scope: str
    payload: Any
    filepath: str
    lineno: int
    level: int | None = None
    step: int | None = None
    time: float | None = None
    extra: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert Event to dictionary."""
        result = {
            "kind": self.kind,
            "scope": self.scope,
            "payload": self.payload,
            "filepath": self.filepath,
            "lineno": self.lineno,
        }

        # Only include optional fields if they are not None
        if self.level is not None:
            result["level"] = self.level
        if self.step is not None:
            result["step"] = self.step
        if self.time is not None:
            result["time"] = self.time
        if self.extra is not None:
            result["extra"] = self.extra

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Event":
        """Create Event from dictionary."""
        return cls(
            kind=data["kind"],
            scope=data["scope"],
            payload=data["payload"],
            filepath=data["filepath"],
            lineno=data["lineno"],
            level=data.get("level"),
            step=data.get("step"),
            time=data.get("time"),
            extra=data.get("extra"),
        )
