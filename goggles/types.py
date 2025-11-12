"""Types used in Goggles."""

import numpy as np
from typing import Literal, Any
from typing_extensions import Self
from dataclasses import dataclass
from typing import TypeAlias

Kind = Literal[
    "log", "metric", "image", "video", "artifact", "histogram", "vector", "vector_field"
]

Metrics = dict[str, float | int | np.ndarray]
Image: TypeAlias = np.ndarray
Video: TypeAlias = np.ndarray
Vector: TypeAlias = np.ndarray
VectorField: TypeAlias = np.ndarray


@dataclass(frozen=True)
class Event:
    """Structured event routed through the EventBus.

    Args:
        kind: Kind of event ("log", "metric", "image", "artifact").
        scope: Scope of the event ("global" or "run").
        payload: Event payload.
        filepath: File path of the caller emitting the event.
        lineno: Line number of the caller emitting the event.
        level: Optional log level for "log" events.
        step: Optional global step index.
        time: Optional global timestamp.
        extra: Optional extra metadata.

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
        """Convert Event to dictionary.

        Returns:
            Dictionary representation of the Event.
        """
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
            for key, value in self.extra.items():
                result["extra." + key] = value

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create Event from dictionary.

        Args:
            data: Dictionary representation of an Event.

        Returns:
            Parsed Event instance.
        """
        extra_dict = {
            k[len("extra.") :]: v for k, v in data.items() if k.startswith("extra.")
        }
        return cls(
            kind=data["kind"],
            scope=data["scope"],
            payload=data["payload"],
            filepath=data["filepath"],
            lineno=data["lineno"],
            level=data.get("level"),
            step=data.get("step"),
            time=data.get("time"),
            extra=extra_dict if extra_dict else None,
        )
