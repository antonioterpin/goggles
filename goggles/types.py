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
    extra: Optional[dict[str, Any]] = None
