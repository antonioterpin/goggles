"""Internal event schema and validation utilities for Goggles.

Warning:
    This module is an implementation detail of Goggles' logging system.
    External code should **not** import from this module directly.

Overview:
    The `MetricEvent` dataclass represents the canonical message format for
    metrics emitted by Goggles' BoundLogger. It defines a consistent schema
    for scalars, images, videos, and histograms, and performs validation at
    construction time to prevent malformed or inconsistent event data from
    entering the logging pipeline.

Usage:
    This module is used internally by the logger's `.push()` method and by
    event queue handlers to validate and serialize metrics. Users should log
    data via the high-level API:

    >>> log = gg.get_logger(__name__)
    >>> log.push(step=10, metrics={"loss": 0.42, "psnr": 29.1})

    Direct construction of `MetricEvent` is not required or recommended.

"""

from __future__ import annotations

import dataclasses
import numbers
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Literal

import numpy as np


# ---------------------------------------------------------------------------
# Payload type aliases
# ---------------------------------------------------------------------------
ScalarPayload = float | int
ImagePayload = np.ndarray
VideoPayload = np.ndarray
HistogramPayload = np.ndarray | list[float] | tuple[float, ...]
MetricPayload = ScalarPayload | ImagePayload | VideoPayload | HistogramPayload


# ---------------------------------------------------------------------------
# MetricEvent Dataclass
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MetricEvent:
    """Schema and validation for a single metric event.

    The `MetricEvent` represents one structured metric emission in Goggles.
    Each event carries a key, type, step, timestamp, and validated payload,
    plus optional contextual metadata and tags.

    Attributes:
        key (str): Unique metric identifier (e.g., `"train/loss"`).
        type (Literal["scalar", "image", "video", "histogram"]):
            Metric type; determines payload validation.
        step (int): Global training step or iteration number (must be ≥ 0).
        payload (MetricPayload): Event payload. Expected structures:
            * scalar: int | float
            * image: np.ndarray (H, W[, C])
            * video: np.ndarray (T, H, W[, C])
            * histogram: np.ndarray (1D) or sequence of numeric values
        ts (float): UNIX timestamp in seconds. Defaults to `time.time()`.
        tags (list[str]): Flat list of string labels for grouping/filtering.
        context (dict[str, Any]): Structured metadata describing event context.

    Raises:
        ValueError: If key is empty, step is negative, or shape is invalid.
        TypeError: If payload, context, or tag types are invalid.

    Example:
        >>> event = MetricEvent(
        ...     key="train/loss",
        ...     type="scalar",
        ...     step=10,
        ...     payload=0.42,
        ...     tags=["train"],
        ...     context={"epoch": 1, "device": "cuda:0"},
        ... )
        >>> isinstance(event.to_dict(), dict)
        True

    """

    key: str
    type: Literal["scalar", "image", "video", "histogram"]
    step: int
    payload: MetricPayload
    ts: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)

    # -------------------------------------------------------------------------
    # Validation logic
    # -------------------------------------------------------------------------
    def __post_init__(self):
        """Validate field types and payload structure after initialization."""
        if not isinstance(self.key, str) or not self.key:
            raise ValueError("key must be a non-empty string.")
        if not isinstance(self.step, int) or self.step < 0:
            raise ValueError("step must be a non-negative integer.")
        if not isinstance(self.ts, (int, float)):
            raise TypeError("ts must be a numeric UNIX timestamp.")
        if not all(isinstance(t, str) for t in self.tags):
            raise TypeError("tags must be a list of strings.")
        if not isinstance(self.context, dict):
            raise TypeError("context must be a dictionary.")

        validator = getattr(self, f"_validate_{self.type}", None)
        if validator is None:
            raise ValueError(f"Unknown metric type: {self.type}")
        validator(self.payload)

    # -------------------------------------------------------------------------
    # Type-specific validation helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _validate_scalar(value: Any) -> None:
        """Validate scalar payload (must be numeric)."""
        if not isinstance(value, numbers.Number):
            raise TypeError(f"Scalar payload must be numeric, got {type(value)}")

    @staticmethod
    def _validate_image(value: Any) -> None:
        """Validate image payload shape and dtype.

        Args:
            value (Any): The image payload to validate.

        Raises:
            TypeError: If the payload is not a NumPy array or has non-numeric dtype
            ValueError: If the payload does not have 2 or 3 dimensions.

        """
        if not isinstance(value, np.ndarray):
            raise TypeError("Image payload must be a NumPy array.")
        if value.ndim not in (2, 3):
            raise ValueError("Image payload must have shape (H, W) or (H, W, C).")
        if not np.issubdtype(value.dtype, np.number):
            raise TypeError("Image payload must have numeric dtype.")

    @staticmethod
    def _validate_video(value: Any) -> None:
        """Validate video payload shape and dtype.

        Args:
            value (Any): The video payload to validate.

        Raises:
            TypeError: If the payload is not a NumPy array or has non-numeric dtype
            ValueError: If the payload does not have 3 or 4 dimensions.

        """
        if not isinstance(value, np.ndarray):
            raise TypeError("Video payload must be a NumPy array.")
        if value.ndim not in (3, 4):
            raise ValueError("Video payload must have shape (T, H, W) or (T, H, W, C).")
        if not np.issubdtype(value.dtype, np.number):
            raise TypeError("Video payload must have numeric dtype.")

    @staticmethod
    def _validate_histogram(value: Any) -> None:
        """Validate histogram payload (1D array or numeric sequence).

        Args:
            value (Any): The histogram payload to validate.

        Raises:
            TypeError: If the payload is not a 1D NumPy array or numeric sequence
            ValueError: If the NumPy array is not 1D.

        """
        if isinstance(value, np.ndarray):
            if value.ndim != 1:
                raise ValueError("Histogram payload must be 1D array or sequence.")
            if not np.issubdtype(value.dtype, np.number):
                raise TypeError("Histogram payload must have numeric dtype.")
        elif isinstance(value, (list, tuple)):
            if not all(isinstance(v, numbers.Number) for v in value):
                raise TypeError("Histogram sequence must contain only numeric values.")
        else:
            raise TypeError("Histogram payload must be array or numeric sequence.")

    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary representation of this event.

        Returns:
            dict[str, Any]: Dictionary with all dataclass fields recursively
                converted to built-in types suitable for serialization.

        """
        return dataclasses.asdict(self)
