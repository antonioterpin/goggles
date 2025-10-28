"""Types used in Goggles."""

from typing import Dict, Literal, Any, Optional
from dataclasses import dataclass

Kind = Literal["log", "metric", "image", "video", "artifact"]

try:
    import jax.numpy as jnp

    jnparray = jnp.ndarray
except ImportError:
    jnparray = bytes
try:
    import numpy as np

    nparray = np.ndarray
except ImportError:
    nparray = bytes

Metrics = Dict[str, float | int]
Image = nparray | jnparray | bytes
Video = nparray | jnparray | bytes


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
        extra (Optional[dict[str, Any]]): Additional structured fields.

    """

    kind: Kind
    scope: str
    payload: Any
    level: Optional[int] = None
    step: Optional[int] = None
    time: Optional[float] = None
    extra: Optional[dict[str, Any]] = None
