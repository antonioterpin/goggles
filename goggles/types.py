"""Types used in Goggles."""

from typing import Literal, Any
from dataclasses import dataclass

Kind = Literal["log", "metric", "image", "artifact"]


@dataclass(frozen=True)
class Event:
    """Structured event routed through the EventBus.

    Args:
        kind (Kind): Kind of event ("log", "metric", "image", "artifact").
        scope (str): Scope of the event ("global" or "run").
        payload (Any): Event payload.
        level (int): Numeric log level for "log" events.
        step (int): Step number associated with the event.
        time (float): Timestamp of the event in seconds since epoch.

    """

    kind: Kind
    scope: str
    payload: Any
    level: int
    step: int
    time: float
