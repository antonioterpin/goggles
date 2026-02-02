"""Filter classes for processing raw array values (NumPy / JAX).

This module provides a small, typed filtering framework to post-process streams
of array-valued observations. All filters accept and return arrays, and are
compatible with both NumPy (`numpy.ndarray`) and JAX (`jax.Array`) inputs.

Design goals:
- Strong typing: filters operate on arrays only (no scalar floats).
- Minimal backend branching: use `get_backend()` and small helpers.
- No code duplication: shared ring-buffer logic for windowed filters.
- Safety: stateful filters do not allow mixing NumPy and JAX inputs within the
  same filter instance (common source of subtle bugs).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np

if TYPE_CHECKING:
    # For static type checkers (pyright/mypy). Requires jax available in the
    # type-checking environment, or configure your checker accordingly.
    from jax import Array as JaxArray
    import jax.numpy as jnp

    HAS_JAX = True
else:
    try:
        from jax import Array as JaxArray
        import jax.numpy as jnp

        HAS_JAX = True
    except ImportError:  # pragma: no cover
        HAS_JAX = False
        jnp = None  # type: ignore[assignment]

        class JaxArray:  # pragma: no cover
            """Runtime placeholder for jax.Array when JAX is unavailable."""

            pass


Array: TypeAlias = np.ndarray | JaxArray


def is_jax_array(x: Array) -> bool:
    """Return True iff `x` is a JAX array and JAX is installed.

    Args:
        x: Input array candidate.

    Returns:
        True if `x` is a JAX array (and JAX is available), else False.
    """
    return HAS_JAX and isinstance(x, JaxArray)


def get_backend(x: Array) -> ModuleType:
    """Return the numerical backend module for `x` (numpy or jax.numpy).

    Args:
        x: Input array.

    Returns:
        `jax.numpy` if `x` is a JAX array, otherwise `numpy`.
    """
    if is_jax_array(x):
        return jnp  # type: ignore[return-value]
    return np


def _buffer_set(buf: Array, idx: int, value: Array) -> Array:
    """Set `buf[idx] = value` in a backend-aware way.

    For NumPy, assignment is in-place. For JAX, this uses `.at[idx].set(value)`
    and returns a new array.

    Args:
        buf: Ring buffer array with leading dimension = window size.
        idx: Index to write.
        value: Sample to store at `idx`.

    Returns:
        Updated buffer (same object for NumPy, new object for JAX).
    """
    if is_jax_array(buf):
        return buf.at[idx].set(value)  # type: ignore[attr-defined]
    buf[idx] = value  # type: ignore[index]
    return buf


@dataclass(frozen=True)
class FilterConfig:
    """Declarative configuration for a filter.

    Attributes:
        type: Filter class name (key into `AVAILABLE_FILTERS`).
        parameters: Keyword parameters passed to the filter constructor.
    """

    type: str
    parameters: Mapping[str, Any]

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> FilterConfig:
        """Create a FilterConfig from a config dict.

        Args:
            cfg: Configuration dict with keys "type" and optional "parameters".

        Returns:
            FilterConfig instance.
        """
        return cls(type=str(cfg["type"]), parameters=dict(cfg.get("parameters", {})))

    def to_dict(self) -> dict[str, Any]:
        """Return a plain Python dict.

        Returns:
            Dictionary representation of the FilterConfig.
        """
        return {"type": self.type, "parameters": dict(self.parameters)}


class Filter(ABC):
    """Abstract base class for array filters.

    Filters are callable objects that transform an input array into an output
    array, potentially using internal state (e.g., moving averages).

    Subclasses must implement:
    - `step(data)`: apply one filtering step and return filtered array.
    - `reset()`: reset internal state.
    - `_name()`: short string name describing the filter instance.
    """

    def __init__(self, prefix: str = "") -> None:
        """Initialize the filter.

        Args:
            prefix: Optional prefix used when building `name` for debugging.
        """
        self.prefix = prefix

    def __call__(self, data: Array) -> Array:
        """Alias for `step()`.

        Args:
            data: Input array.

        Returns:
            Filtered output array.
        """
        return self.step(data)

    @abstractmethod
    def step(self, data: Array) -> Array:
        """Filter `data` and return the filtered array.

        Args:
            data: Input array.

        Returns:
            Filtered output array.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset any internal state held by the filter."""

    @property
    def name(self) -> str:
        """Return a human-readable name for this filter instance.

        Returns:
            Debug name including any configured prefix.
        """
        return self.prefix + self._name()

    @abstractmethod
    def _name(self) -> str:
        """Return the unprefixed name for this filter instance.

        Returns:
            Short descriptive name.
        """


class _BackendAware(Filter):
    """Filter mixin that enforces a single backend (NumPy or JAX) per instance.

    Stateful filters that allocate buffers should not be fed alternating NumPy
    and JAX arrays; this class detects that and raises early.

    Notes:
        Stateless filters may not need this; they can just use `get_backend()`.
    """

    def __init__(self, prefix: str = "") -> None:
        """Initialize the backend-aware filter.

        Args:
            prefix: Optional filter name prefix.
        """
        super().__init__(prefix=prefix)
        self._is_jax: bool | None = None

    def _xp(self, data: Array) -> ModuleType:
        """Return backend module for `data`, enforcing backend consistency.

        Args:
            data: Input array.

        Returns:
            `numpy` or `jax.numpy` matching `data`.

        Raises:
            TypeError: If the filter instance previously saw a different backend.
        """
        cur_is_jax = is_jax_array(data)
        if self._is_jax is None:
            self._is_jax = cur_is_jax
        elif self._is_jax != cur_is_jax:
            raise TypeError(
                "Cannot mix NumPy and JAX inputs within the same filter instance. "
                "Create separate filter objects per backend."
            )
        return jnp if cur_is_jax else np  # type: ignore[return-value]

    def reset(self) -> None:
        """Reset backend selection so the instance can be reused."""
        self._is_jax = None


class ScaleFilter(Filter):
    """Multiply input array by a constant scale factor."""

    def __init__(self, scale: float, prefix: str = "") -> None:
        """Initialize the scale filter.

        Args:
            scale: Multiplicative scaling factor.
            prefix: Optional filter name prefix.

        Raises:
            TypeError: If `scale` is not numeric.
        """
        if not isinstance(scale, (int, float)):
            raise TypeError("scale must be a numeric value")
        super().__init__(prefix=prefix)
        self.scale = float(scale)

    def step(self, data: Array) -> Array:
        """Scale the input array.

        Args:
            data: Input array.

        Returns:
            `data * scale`.
        """
        return data * self.scale

    def reset(self) -> None:
        """Reset filter state (no-op; this filter is stateless)."""
        return

    def _name(self) -> str:
        """Return a short descriptive name."""
        return f"ScaleFilter(scale={self.scale})"


class MinMaxFilter(Filter):
    """Affinely map values from [min_val, max_val] to [0, 1] and clip."""

    def __init__(self, min_val: float, max_val: float, prefix: str = "") -> None:
        """Initialize the MinMax filter.

        Args:
            min_val: Minimum input value mapped to 0.
            max_val: Maximum input value mapped to 1.
            prefix: Optional filter name prefix.

        Raises:
            ValueError: If `min_val >= max_val`.
        """
        if min_val >= max_val:
            raise ValueError("min_val must be less than max_val")
        super().__init__(prefix=prefix)
        self.min_val = float(min_val)
        self.max_val = float(max_val)

    def step(self, data: Array) -> Array:
        """Apply min-max normalization with clipping.

        Args:
            data: Input array.

        Returns:
            Array scaled to [0, 1] and clipped.
        """
        xp = get_backend(data)
        return xp.clip((data - self.min_val) / (self.max_val - self.min_val), 0.0, 1.0)

    def reset(self) -> None:
        """Reset filter state (no-op; this filter is stateless)."""
        return

    def _name(self) -> str:
        """Return a short descriptive name."""
        return f"MinMaxFilter({self.min_val},{self.max_val})"


class _WindowBufferFilter(_BackendAware):
    """Base class for windowed filters backed by a ring buffer.

    This implements:
    - backend-consistent buffer allocation (NumPy or JAX)
    - ring-buffer indexing
    - tracking number of samples seen
    - backend-aware writes (`_buffer_set`)

    Subclasses implement `step()` to compute a statistic over the valid window.
    """

    def __init__(self, window_size: int, prefix: str = "") -> None:
        """Initialize the windowed filter base.

        Args:
            window_size: Size of the moving window (number of samples stored).
            prefix: Optional filter name prefix.

        Raises:
            ValueError: If `window_size` is not a positive integer.
        """
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("window_size must be a positive integer")
        super().__init__(prefix=prefix)
        self.window_size = window_size
        self.buffer: Array | None = None
        self.index: int = 0
        self.n_seen: int = 0

    def _push(self, data: Array) -> tuple[ModuleType, int]:
        """Insert `data` into the ring buffer and return backend and valid length.

        Args:
            data: Input array to store.

        Returns:
            A tuple `(xp, valid_len)` where:
            - `xp` is `numpy` or `jax.numpy`
            - `valid_len` is the number of valid samples in the buffer
              (min(n_seen, window_size)).
        """
        xp = self._xp(data)
        if self.buffer is None:
            self.buffer = xp.zeros((self.window_size,) + data.shape, dtype=data.dtype)

        # buffer is guaranteed to be non-None here
        assert self.buffer is not None, "Buffer should be initialized"
        self.buffer = _buffer_set(self.buffer, self.index, data)
        self.index = (self.index + 1) % self.window_size
        self.n_seen += 1
        valid_len = min(self.n_seen, self.window_size)
        return xp, valid_len

    def reset(self) -> None:
        """Reset buffer and counters."""
        super().reset()
        self.buffer = None
        self.index = 0
        self.n_seen = 0


class AverageFilter(_WindowBufferFilter):
    """Compute a simple moving average over the last `window_size` samples."""

    def step(self, data: Array) -> Array:
        """Insert `data` and return the window mean.

        Args:
            data: Input array.

        Returns:
            Mean over the valid portion of the ring buffer.
        """
        xp, valid_len = self._push(data)
        buf = self.buffer
        assert buf is not None, "Buffer should be initialized"
        return xp.mean(buf[:valid_len], axis=0)

    def _name(self) -> str:
        """Return a short descriptive name."""
        return f"AverageFilter({self.window_size})"


class MedianFilter(_WindowBufferFilter):
    """Compute a moving median over the last `window_size` samples."""

    def step(self, data: Array) -> Array:
        """Insert `data` and return the window median.

        Args:
            data: Input array.

        Returns:
            Median over the valid portion of the ring buffer.
        """
        xp, valid_len = self._push(data)
        buf = self.buffer
        assert buf is not None, "Buffer should be initialized"
        return xp.median(buf[:valid_len], axis=0)

    def _name(self) -> str:
        """Return a short descriptive name."""
        return f"MedianFilter({self.window_size})"


class ExpAverageFilter(_BackendAware):
    """Exponential moving average.

    Recurrence:
        y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
    """

    def __init__(self, alpha: float, prefix: str = "") -> None:
        """Initialize the exponential moving average filter.

        Args:
            alpha: Smoothing factor in [0, 1].
            prefix: Optional filter name prefix.

        Raises:
            ValueError: If `alpha` is not in [0, 1].
        """
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in the range [0, 1]")
        super().__init__(prefix=prefix)
        self.alpha = float(alpha)
        self.value: Array | None = None

    def step(self, data: Array) -> Array:
        """Update and return the exponential moving average.

        Args:
            data: Input array.

        Returns:
            Updated EMA value.
        """
        self._xp(data)  # enforce backend consistency
        if self.value is None:
            self.value = data
        else:
            self.value = self.alpha * data + (1.0 - self.alpha) * self.value
        return self.value

    def reset(self) -> None:
        """Reset EMA state."""
        super().reset()
        self.value = None

    def _name(self) -> str:
        """Return a short descriptive name."""
        return f"ExpAverageFilter({self.alpha})"


class QuantizationFilter(_BackendAware):
    """Clamp to a range then quantize to nearest discrete level.

    The quantization levels are:
        min_value, min_value + step_size, ..., max_value

    Notes:
        Levels are cached per instance and rebuilt on `reset()`.
    """

    def __init__(
        self,
        min_value: float = -0.150,
        max_value: float = 0.150,
        step_size: float = 0.00015,
        prefix: str = "",
    ) -> None:
        """Initialize the quantization filter.

        Args:
            min_value: Minimum clamp value.
            max_value: Maximum clamp value.
            step_size: Quantization step.
            prefix: Optional filter name prefix.

        Raises:
            ValueError: If `step_size <= 0` or `min_value >= max_value`.
        """
        super().__init__(prefix=prefix)
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        if min_value >= max_value:
            raise ValueError("min_value must be < max_value")
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.step_size = float(step_size)
        self.levels: Array | None = None

    def step(self, data: Array) -> Array:
        """Clamp and quantize `data` to nearest level.

        Args:
            data: Input array.

        Returns:
            Quantized output array.
        """
        xp = self._xp(data)
        clipped = xp.clip(data, self.min_value, self.max_value)

        # levels shape: (n_levels, 1, 1, ..., 1) for broadcasting over data.ndim
        if self.levels is None:
            base = xp.arange(
                self.min_value, self.max_value + self.step_size, self.step_size
            )
            self.levels = base.reshape((-1,) + (1,) * data.ndim)

        levels = self.levels
        assert levels is not None, "Quantization levels should be initialized"

        idx = xp.argmin(xp.abs(levels - clipped), axis=0)
        gathered = xp.take_along_axis(levels, idx[None, ...], axis=0)
        return gathered.squeeze(axis=0)

    def reset(self) -> None:
        """Reset cached quantization levels and backend tracking."""
        super().reset()
        self.levels = None

    def _name(self) -> str:
        """Return a short descriptive name."""
        return (
            "QuantizationFilter("
            f"min={self.min_value}, max={self.max_value}, step={self.step_size}"
            ")"
        )


class ConcatFilter(Filter):
    """Apply multiple filters in sequence."""

    def __init__(self, filters: list[Filter], prefix: str = "") -> None:
        """Initialize the concatenation filter.

        Args:
            filters: List of filters to apply in order.
            prefix: Optional filter name prefix.
        """
        super().__init__(prefix=prefix)
        self.filters = filters

    def step(self, data: Array) -> Array:
        """Apply all filters sequentially.

        Args:
            data: Input array.

        Returns:
            Output after applying each filter in `filters` in order.
        """
        out = data
        for f in self.filters:
            out = f.step(out)
        return out

    def reset(self) -> None:
        """Reset each filter in the chain."""
        for f in self.filters:
            f.reset()

    def _name(self) -> str:
        """Return a short descriptive name."""
        return f"ConcatFilter({[f.name for f in self.filters]})"


AVAILABLE_FILTERS: dict[str, type[Filter]] = {
    "MinMaxFilter": MinMaxFilter,
    "AverageFilter": AverageFilter,
    "ExpAverageFilter": ExpAverageFilter,
    "MedianFilter": MedianFilter,
    "QuantizationFilter": QuantizationFilter,
    "ScaleFilter": ScaleFilter,
}
"""Registry mapping filter type names to filter classes.

Note:
    `ConcatFilter` is intentionally not included because it is built via
    `create_concat_filter()` from a list of `FilterConfig`.
"""


def create_concat_filter(
    filter_configs: list[FilterConfig],
    *,
    available_filters: dict[str, type[Filter]] = AVAILABLE_FILTERS,
) -> ConcatFilter:
    """Create a `ConcatFilter` from a list of declarative filter configs.

    Args:
        filter_configs: Ordered list of filter configurations. Each config's
            `type` must exist in `available_filters`, and its `parameters` must
            match the target constructor signature.
        available_filters: Optional custom filter registry to use instead of
            the default `AVAILABLE_FILTERS`.

    Returns:
        A `ConcatFilter` that applies the configured filters in sequence.

    Raises:
        ValueError: If an unknown filter type is encountered or parameters are invalid.
    """
    filters: list[Filter] = []
    for idx, cfg in enumerate(filter_configs):
        if cfg.type not in available_filters:
            raise ValueError(f"Unknown filter type: {cfg.type}")

        cls = available_filters[cfg.type]
        try:
            filters.append(cls(**cfg.parameters, prefix=f"[{idx}] "))  # type: ignore[arg-type]
        except Exception as e:
            raise ValueError(
                f"Invalid parameters for {cfg.type} (index {idx}): {e}"
            ) from e

    return ConcatFilter(filters)
