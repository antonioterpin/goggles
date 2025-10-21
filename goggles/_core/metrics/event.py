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
ArrayLike = Any  # np.ndarray | jax.Array | torch.Tensor | etc.
MaterializedPayload = bytes  # PNG, JPEG, MP4, or serialized blob
MetricPayload = (
    float | int | ArrayLike | list[float] | tuple[float, ...] | MaterializedPayload
)


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
        """Validate field types and payload structure after initialization.

        Raises:
            ValueError: If key is empty, step is negative, or shape is invalid.
            TypeError: If payload, context, or tag types are invalid.

        """
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

        state = self.context.get("_state", "raw")
        if state == "materialized":
            if not isinstance(self.payload, bytes):
                raise TypeError("Materialized payloads must be bytes (e.g., PNG, MP4).")
            return  # Skip further validation for materialized payloads

        validator = getattr(self, f"_validate_{self.type}", None)
        if validator is None:
            raise ValueError(f"Unknown metric type: {self.type}")
        validator(self.payload)

    # -------------------------------------------------------------------------
    # Type-specific validation helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _validate_scalar(value: Any) -> None:
        """Validate scalar payload (must be numeric).

        Args:
            value (Any): The scalar payload to validate.

        Raises:
            TypeError: If the payload is not a numeric type.

        """
        if not isinstance(value, numbers.Number):
            raise TypeError(f"Scalar payload must be numeric, got {type(value)}")

    @staticmethod
    def _validate_image(value: Any) -> None:
        """Validate image payload with host-aware dtype check.

        Args:
            value (Any): Image payload to validate.

        Raises:
            TypeError: If the payload is not array-like or has invalid dtype.
            ValueError: If the payload does not appear to have 2 or 3 dimensions.

        """
        ndim = getattr(value, "ndim", None)
        shape = getattr(value, "shape", None)

        if ndim is None and shape is None:
            raise TypeError("Image payload must expose 'shape' or 'ndim' attributes.")

        if ndim is None and isinstance(shape, (tuple, list)):
            ndim = len(shape)

        if ndim not in (2, 3):
            raise ValueError("Image payload must have ndim 2 (H,W) or 3 (H,W,C).")

        # Only enforce dtype check if this is a NumPy array (safe on host)
        if isinstance(value, np.ndarray):
            if not np.issubdtype(value.dtype, np.number):
                raise TypeError(
                    f"Image payload must have numeric dtype, got {value.dtype}."
                )

    @staticmethod
    def _validate_video(value: Any) -> None:
        """Validate video payload with host-aware dtype check.

        Args:
            value (Any): Video payload to validate.

        Raises:
            TypeError: If the payload is not array-like or has invalid dtype.
            ValueError: If the payload does not appear to have 3 or 4 dimensions.

        """
        ndim = getattr(value, "ndim", None)
        shape = getattr(value, "shape", None)

        if ndim is None and shape is None:
            raise TypeError("Video payload must expose 'shape' or 'ndim' attributes.")

        if ndim is None and isinstance(shape, (tuple, list)):
            ndim = len(shape)

        if ndim not in (3, 4):
            raise ValueError("Video payload must have ndim 3 (T,H,W) or 4 (T,H,W,C).")

        if isinstance(value, np.ndarray):
            if not np.issubdtype(value.dtype, np.number):
                raise TypeError(
                    f"Video payload must have numeric dtype, got {value.dtype}."
                )

    @staticmethod
    def _validate_histogram(value: Any) -> None:
        """Validate histogram payload structure (safe for device arrays).

        Args:
            value (Any): Histogram payload to validate.

        Raises:
            TypeError: If payload is not array-like or numeric sequence.
            ValueError: If array is not 1D.

        """
        ndim = getattr(value, "ndim", None)
        shape = getattr(value, "shape", None)

        if ndim is not None or shape is not None:
            if ndim is None and isinstance(shape, (tuple, list)):
                ndim = len(shape)
            if ndim != 1:
                raise ValueError("Histogram payload must be 1-D array or sequence.")

            if isinstance(value, np.ndarray):
                if not np.issubdtype(value.dtype, np.number):
                    raise TypeError(
                        f"Histogram payload must have numeric dtype, got {value.dtype}."
                    )
            return

        if isinstance(value, (list, tuple)):
            if not all(isinstance(v, numbers.Number) for v in value):
                raise TypeError("Histogram sequence must contain only numeric values.")
            return

        raise TypeError("Histogram payload must be 1-D array-like or numeric sequence.")

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
