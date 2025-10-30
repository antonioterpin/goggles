"""Types used in Goggles."""

import numpy as np
from typing import Dict, Literal, Any, Optional
from dataclasses import dataclass
from typing import TypeAlias

Kind = Literal["log", "metric", "image", "video", "artifact"]

Metrics = Dict[str, float | int]
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
        level (Optional[int]): Optional log level for "log" events.
        step (Optional[int]): Optional global step index.
        time (Optional[float]): Optional global timestamp.
        extra (Optional[dict[str, Any]]): Optional extra metadata.

    """

    kind: Kind
    scope: str
    payload: Any
    filepath: str
    lineno: int
    level: Optional[int] = None
    step: Optional[int] = None
    time: Optional[float] = None
    extra: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
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
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
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
