"""Filter classes for processing raw array values (NumPy / JAX).

This module provides a small, typed filtering framework to post-process streams
of array-valued observations. All filters accept and return arrays, and are
compatible with both NumPy (`numpy.ndarray`) and JAX (`jax.Array`) inputs.

Each filter exposes two equivalent interfaces:

- An ergonomic stateful API:
    - ``step(data) -> output`` (and ``__call__``)
    - ``reset()``

- A pure functional, JIT-compatible API:
    - ``init_state(data) -> state``: allocate the initial state pytree given a
      sample input that determines shape and dtype.
    - ``apply(state, data) -> (new_state, output)``: the pure step. Safe to
      pass through ``jax.jit`` and ``jax.lax.scan``.

The stateful ``step`` is implemented as a thin wrapper that lazily initializes
internal state via ``init_state`` and threads it through ``apply``.

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
from typing import TYPE_CHECKING, Any, TypeAlias, TypeGuard

import numpy as np

if TYPE_CHECKING:
    import jax.numpy as jnp
    from jax import Array as JaxArray

    HAS_JAX = True
else:
    try:
        import jax.numpy as jnp
        from jax import Array as JaxArray

        HAS_JAX = True
    except ImportError:  # pragma: no cover
        HAS_JAX = False
        jnp = None  # type: ignore[assignment]

        class JaxArray:  # pragma: no cover
            """Runtime placeholder for jax.Array when JAX is unavailable."""

            pass


Array: TypeAlias = np.ndarray | JaxArray
State: TypeAlias = Any


def is_jax_array(x: Array) -> TypeGuard[JaxArray]:
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
        assert jnp is not None
        return jnp
    return np


def _buffer_set(buf: Array, idx: Array, value: Array) -> Array:
    """Set `buf[idx] = value` in a backend-aware way.

    For NumPy, assignment is in-place. For JAX, this uses `.at[idx].set(value)`
    and returns a new array.

    Args:
        buf: Ring buffer array with leading dimension = window size.
        idx: Index to write (scalar int).
        value: Sample to store at `idx`.

    Returns:
        Updated buffer (same object for NumPy, new object for JAX).
    """
    if is_jax_array(buf):
        return buf.at[idx].set(value)

    np_buf = np.asarray(buf)
    np_buf[int(np.asarray(idx))] = np.asarray(value)
    return np_buf


def _broadcast_mask(mask: Array, ndim: int) -> Array:
    """Reshape a 1-D `(W,)` mask to broadcast over a `(W, *data_shape)` buffer.

    Args:
        mask: Mask of shape `(W,)`.
        ndim: Number of trailing dimensions in the buffer beyond `W`.

    Returns:
        Mask reshaped to `(W,) + (1,) * ndim`.
    """
    return mask.reshape((mask.shape[0],) + (1,) * ndim)


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
        return cls(
            type=str(cfg["type"]), parameters=dict(cfg.get("parameters", {}))
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain Python dict.

        Returns:
            Dictionary representation of the FilterConfig.
        """
        return {"type": self.type, "parameters": dict(self.parameters)}


class Filter(ABC):
    """Abstract base class for array filters.

    Filters are callable objects that transform an input array into an output
    array, optionally using internal state (e.g., moving averages).

    Subclasses must implement:
        - ``init_state(data)``: build the initial state pytree from a sample
          input. Stateless filters may return ``None``.
        - ``apply(state, data)``: pure step returning ``(new_state, output)``.
        - ``_name()``: short string name describing the filter instance.

    The default ``step`` implementation lazily initializes ``self._state`` via
    ``init_state`` and threads it through ``apply``. Stateful filters
    additionally enforce a single backend (NumPy or JAX) per instance.
    """

    def __init__(self, prefix: str = "") -> None:
        """Initialize the filter.

        Args:
            prefix: Optional prefix used when building `name` for debugging.
        """
        self.prefix = prefix
        self._state: State = None
        self._is_jax: bool | None = None

    def __call__(self, data: Array) -> Array:
        """Alias for `step()`.

        Args:
            data: Input array.

        Returns:
            Filtered output array.
        """
        return self.step(data)

    def step(self, data: Array) -> Array:
        """Filter `data` using internal state and return the filtered array.

        Args:
            data: Input array.

        Returns:
            Filtered output array.

        Raises:
            TypeError: If the filter previously saw a different array backend.
        """
        cur_is_jax = is_jax_array(data)
        if self._is_jax is None:
            self._is_jax = cur_is_jax
        elif self._is_jax != cur_is_jax:
            raise TypeError(
                "Cannot mix NumPy and JAX inputs within the same filter. "
                "Create separate filter objects per backend."
            )
        if self._state is None:
            self._state = self.init_state(data)
        self._state, out = self.apply(self._state, data)
        return out

    def reset(self) -> None:
        """Reset internal state and backend tracking."""
        self._state = None
        self._is_jax = None

    @abstractmethod
    def init_state(self, data: Array) -> State:
        """Allocate the initial state pytree for this filter.

        Args:
            data: A sample input used to determine shape and dtype of any
                allocated state arrays. The values are not consumed unless
                explicitly noted by a subclass.

        Returns:
            The initial state pytree. ``None`` for stateless filters.
        """

    @abstractmethod
    def apply(self, state: State, data: Array) -> tuple[State, Array]:
        """Apply one filtering step in functional form.

        Args:
            state: Current state pytree (as returned by `init_state` or a
                previous `apply`).
            data: Input array.

        Returns:
            A tuple `(new_state, output)`.
        """

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

    def init_state(self, data: Array) -> State:
        """Stateless: returns ``None``.

        Args:
            data: Sample input (unused).

        Returns:
            ``None``.
        """
        del data
        return None

    def apply(self, state: State, data: Array) -> tuple[State, Array]:
        """Scale the input array.

        Args:
            state: Unused (stateless).
            data: Input array.

        Returns:
            ``(state, data * scale)``.
        """
        return state, data * self.scale

    def _name(self) -> str:
        """Return a short descriptive name.

        Returns:
            Name string including the scale factor.
        """
        return f"ScaleFilter(scale={self.scale})"


class MinMaxFilter(Filter):
    """Affinely map values from [min_val, max_val] to [0, 1] and clip."""

    def __init__(
        self, min_val: float, max_val: float, prefix: str = ""
    ) -> None:
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

    def init_state(self, data: Array) -> State:
        """Stateless: returns ``None``.

        Args:
            data: Sample input (unused).

        Returns:
            ``None``.
        """
        del data
        return None

    def apply(self, state: State, data: Array) -> tuple[State, Array]:
        """Apply min-max normalization with clipping.

        Args:
            state: Unused (stateless).
            data: Input array.

        Returns:
            ``(state, normalized)`` where `normalized` is scaled to [0, 1] and
            clipped.
        """
        xp = get_backend(data)
        out = xp.clip(
            (data - self.min_val) / (self.max_val - self.min_val), 0.0, 1.0
        )
        return state, out

    def _name(self) -> str:
        """Return a short descriptive name.

        Returns:
            Name string including the min and max values.
        """
        return f"MinMaxFilter({self.min_val},{self.max_val})"


class _WindowBufferFilter(Filter):
    """Base class for windowed filters backed by a fixed-shape ring buffer.

    State is a 3-tuple ``(buffer, index, n_seen)``:
        - ``buffer``: shape ``(window_size, *data.shape)``, zero-initialized.
        - ``index``: int32 scalar, next write position (modulo window_size).
        - ``n_seen``: int32 scalar, number of samples observed so far.

    Subclasses implement ``_compute(xp, buf, mask_b, n_seen, data)`` to compute
    the windowed statistic. The buffer always has shape ``(window_size, ...)``;
    invalid (not-yet-written) slots are masked via ``mask_b``.
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

    def init_state(self, data: Array) -> State:
        """Allocate ring-buffer state matching `data`'s shape and dtype.

        Args:
            data: Sample input determining buffer shape and dtype.

        Returns:
            A tuple ``(buffer, index, n_seen)``.
        """
        xp = get_backend(data)
        buffer = xp.zeros((self.window_size, *data.shape), dtype=data.dtype)
        index = xp.asarray(0, dtype=xp.int32)
        n_seen = xp.asarray(0, dtype=xp.int32)
        return (buffer, index, n_seen)

    def _push(
        self,
        state: State,
        data: Array,
    ) -> tuple[State, ModuleType, Array, Array]:
        """Insert `data` into the buffer and return updated state and helpers.

        Args:
            state: Current ring-buffer state.
            data: Input array to insert.

        Returns:
            A tuple ``(new_state, xp, mask_b, n_seen_new)`` where:
                - ``new_state`` reflects `data` written at the current index.
                - ``xp`` is the backend module.
                - ``mask_b`` is a broadcasted validity mask of shape
                  ``(window_size,) + (1,) * data.ndim``.
                - ``n_seen_new`` is the post-push sample count.
        """
        buf, index, n_seen = state
        xp = get_backend(data)
        new_buf = _buffer_set(buf, index, data)
        new_index = (index + 1) % self.window_size
        n_seen_new = n_seen + 1
        valid_count = xp.minimum(n_seen_new, self.window_size)
        mask = xp.arange(self.window_size, dtype=xp.int32) < valid_count
        mask_b = _broadcast_mask(mask, data.ndim)
        return (new_buf, new_index, n_seen_new), xp, mask_b, valid_count


class AverageFilter(_WindowBufferFilter):
    """Compute a simple moving average over the last `window_size` samples."""

    def apply(self, state: State, data: Array) -> tuple[State, Array]:
        """Insert `data` and return the window mean.

        Args:
            state: Ring-buffer state.
            data: Input array.

        Returns:
            ``(new_state, mean)`` over the valid portion of the buffer.
        """
        new_state, xp, mask_b, valid_count = self._push(state, data)
        buf = new_state[0]
        divisor = xp.maximum(valid_count, 1).astype(buf.dtype)
        total = xp.sum(
            xp.where(mask_b, buf, xp.asarray(0, dtype=buf.dtype)), axis=0
        )
        return new_state, total / divisor

    def _name(self) -> str:
        """Return a short descriptive name.

        Returns:
            Name string including the window size.
        """
        return f"AverageFilter({self.window_size})"


class MedianFilter(_WindowBufferFilter):
    """Compute a moving median over the last `window_size` samples.

    During warm-up (when fewer than ``window_size`` samples have been
    observed), invalid buffer slots are filled with the most recent input so
    the median is well-defined and JIT-compatible. Once the window is full,
    this is identical to a standard moving median.
    """

    def apply(self, state: State, data: Array) -> tuple[State, Array]:
        """Insert `data` and return the window median.

        Args:
            state: Ring-buffer state.
            data: Input array.

        Returns:
            ``(new_state, median)`` over the buffer with invalid slots filled
            with `data`.
        """
        new_state, xp, mask_b, _ = self._push(state, data)
        buf = new_state[0]
        view = xp.where(mask_b, buf, data)
        return new_state, xp.median(view, axis=0)

    def _name(self) -> str:
        """Return a short descriptive name.

        Returns:
            Name string including the window size.
        """
        return f"MedianFilter({self.window_size})"


class ExpAverageFilter(Filter):
    """Exponential moving average.

    Recurrence:
        y[n] = alpha * x[n] + (1 - alpha) * y[n-1]

    The very first sample is returned unchanged (initialization).
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

    def init_state(self, data: Array) -> State:
        """Allocate EMA state.

        Args:
            data: Sample input determining the running value's shape and dtype.

        Returns:
            A tuple ``(value, initialized)`` where ``value`` is zero-allocated
            and ``initialized`` is a scalar ``False``.
        """
        xp = get_backend(data)
        value = xp.zeros_like(data)
        initialized = xp.asarray(False)
        return (value, initialized)

    def apply(self, state: State, data: Array) -> tuple[State, Array]:
        """Update and return the exponential moving average.

        Args:
            state: Current EMA state.
            data: Input array.

        Returns:
            ``(new_state, ema)``. On the first call, ``ema == data``.
        """
        value, initialized = state
        xp = get_backend(data)
        next_value = self.alpha * data + (1.0 - self.alpha) * value
        new_value = xp.where(initialized, next_value, data)
        new_initialized = xp.asarray(True)
        return (new_value, new_initialized), new_value

    def _name(self) -> str:
        """Return a short descriptive name.

        Returns:
            Name string including the alpha value.
        """
        return f"ExpAverageFilter({self.alpha})"


class QuantizationFilter(Filter):
    """Clamp to a range then quantize to nearest discrete level.

    The quantization levels are:
        min_value, min_value + step_size, ..., max_value

    Notes:
        Levels depend on the input array's number of dimensions; they are
        materialized in `init_state`.
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

    def init_state(self, data: Array) -> State:
        """Materialize quantization levels broadcast over `data.ndim`.

        Args:
            data: Sample input determining trailing broadcast dimensions.

        Returns:
            The levels array of shape ``(n_levels,) + (1,) * data.ndim``.
        """
        xp = get_backend(data)
        base = xp.arange(
            self.min_value, self.max_value + self.step_size, self.step_size
        )
        return base.reshape((-1,) + (1,) * data.ndim)

    def apply(self, state: State, data: Array) -> tuple[State, Array]:
        """Clamp and quantize `data` to nearest level.

        Args:
            state: Pre-computed levels array.
            data: Input array.

        Returns:
            ``(state, quantized)``.
        """
        xp = get_backend(data)
        levels = state
        clipped = xp.clip(data, self.min_value, self.max_value)
        idx = xp.argmin(xp.abs(levels - clipped), axis=0)
        gathered = xp.take_along_axis(levels, idx[None, ...], axis=0)
        return state, gathered.squeeze(axis=0)

    def _name(self) -> str:
        """Return a short descriptive name.

        Returns:
            Name string including the quantization parameters.
        """
        return (
            "QuantizationFilter("
            f"min={self.min_value}, max={self.max_value}, step={self.step_size}"
            ")"
        )


class RangeRejectFilter(Filter):
    """Replace out-of-range values using a fallback filter."""

    def __init__(
        self,
        min_value: float,
        max_value: float,
        fallback_filter: list[Mapping[str, Any] | FilterConfig],
        available_filters: dict[str, type[Filter]] | None = None,
        prefix: str = "",
    ) -> None:
        """Initialize the range reject filter.

        Args:
            min_value: Minimum valid value (inclusive).
            max_value: Maximum valid value (inclusive).
            fallback_filter:
                List of filter configurations (dict or `FilterConfig`) to apply
                to out-of-range values. These are concatenated in order using
                `create_concat_filter()`.
            available_filters: Optional filter registry.
            prefix: Optional filter name prefix.

        Raises:
            ValueError: If `min_value >= max_value`.
        """
        if min_value >= max_value:
            raise ValueError("min_value must be < max_value")

        super().__init__(prefix=prefix)

        if available_filters is None:
            available_filters = AVAILABLE_FILTERS

        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.midpoint = (self.min_value + self.max_value) * 0.5

        fallback_cfgs = [
            fc if isinstance(fc, FilterConfig) else FilterConfig.from_config(fc)
            for fc in fallback_filter
        ]

        self.fallback = create_concat_filter(
            fallback_cfgs,
            available_filters=available_filters,
        )

    def init_state(self, data: Array) -> State:
        """Allocate state for this filter and its fallback chain.

        Args:
            data: Sample input.

        Returns:
            ``(last_valid, fallback_state)``.
        """
        xp = get_backend(data)
        last_valid = xp.full_like(data, self.midpoint)
        fallback_state = self.fallback.init_state(data)
        return (last_valid, fallback_state)

    def apply(self, state: State, data: Array) -> tuple[State, Array]:
        """Identify out-of-range values and replace them.

        Args:
            state: ``(last_valid, fallback_state)``.
            data: Input array.

        Returns:
            ``(new_state, output)`` where out-of-range entries are replaced by
            the fallback's output.
        """
        xp = get_backend(data)
        last_valid, fallback_state = state

        valid = xp.logical_and(
            data >= self.min_value,
            data <= self.max_value,
        )

        safe_data = xp.where(valid, data, last_valid)
        new_fallback_state, fallback_value = self.fallback.apply(
            fallback_state, safe_data
        )
        new_last_valid = xp.where(valid, data, last_valid)
        out = xp.where(valid, data, fallback_value)
        return (new_last_valid, new_fallback_state), out

    def _name(self) -> str:
        return (
            f"RangeRejectFilter("
            f"{self.min_value},{self.max_value},"
            f"fallback={self.fallback.name})"
        )


class StdRejectFilter(_WindowBufferFilter):
    """Replace outliers using a fallback filter.

    Per element (index), a value is considered valid if:
        abs(x_t - mean(x_{t-N:t-1})) <= std_factor * std(x_{t-N:t-1})

    where statistics are computed over the last `window_size` accepted samples
    for that element.
    """

    def __init__(
        self,
        std_factor: float,
        window_size: int,
        fallback_filter: list[Mapping[str, Any] | FilterConfig],
        available_filters: dict[str, type[Filter]] | None = None,
        prefix: str = "",
    ) -> None:
        """Initialize the std-based reject filter.

        Args:
            std_factor: Outlier threshold multiplier for standard deviation.
            window_size: Number of past samples used to estimate mean/std.
            fallback_filter:
                List of filter configurations (dict or `FilterConfig`) to apply
                to outlier values. These are concatenated in order using
                `create_concat_filter()`.
            available_filters: Optional filter registry.
            prefix: Optional filter name prefix.

        Raises:
            ValueError: If `std_factor <= 0`.
        """
        if std_factor <= 0:
            raise ValueError("std_factor must be > 0")

        super().__init__(window_size=window_size, prefix=prefix)

        if available_filters is None:
            available_filters = AVAILABLE_FILTERS

        self.std_factor = float(std_factor)

        fallback_cfgs = [
            fc if isinstance(fc, FilterConfig) else FilterConfig.from_config(fc)
            for fc in fallback_filter
        ]

        self.fallback = create_concat_filter(
            fallback_cfgs,
            available_filters=available_filters,
        )

    def init_state(self, data: Array) -> State:
        """Allocate ring-buffer + last-valid + fallback state.

        Args:
            data: Sample input.

        Returns:
            ``(buffer, index, n_seen, last_valid, last_valid_initialized,
            fallback_state)``.
        """
        xp = get_backend(data)
        buffer, index, n_seen = super().init_state(data)
        last_valid = xp.zeros_like(data)
        last_valid_initialized = xp.asarray(False)
        fallback_state = self.fallback.init_state(data)
        return (
            buffer,
            index,
            n_seen,
            last_valid,
            last_valid_initialized,
            fallback_state,
        )

    def apply(self, state: State, data: Array) -> tuple[State, Array]:
        """Identify outliers and replace them using the fallback filter.

        Args:
            state: Filter state; see `init_state` for layout.
            data: Input array.

        Returns:
            ``(new_state, output)``. Outliers are replaced by the fallback's
            output once the window has filled.
        """
        (
            buf,
            index,
            n_seen,
            last_valid,
            last_valid_initialized,
            fallback_state,
        ) = state
        xp = get_backend(data)

        # Always compute mean/std on the current buffer (zeroed in invalid
        # slots). These are only consumed in the post-warmup branch via
        # `where`-based selection, so warmup behavior is unaffected by the
        # zero-padding bias.
        valid_count = xp.minimum(n_seen, self.window_size)
        mask = xp.arange(self.window_size, dtype=xp.int32) < valid_count
        mask_b = _broadcast_mask(mask, data.ndim)
        zero = xp.asarray(0, dtype=buf.dtype)
        masked_buf = xp.where(mask_b, buf, zero)
        divisor = xp.maximum(valid_count, 1).astype(buf.dtype)
        mean = xp.sum(masked_buf, axis=0) / divisor
        centered = xp.where(mask_b, buf - mean[None], zero)
        variance = xp.sum(centered * centered, axis=0) / divisor
        std = xp.sqrt(variance)

        warmup = n_seen < self.window_size
        post_warmup_valid = xp.abs(data - mean) <= self.std_factor * std
        all_valid = xp.ones_like(data, dtype=post_warmup_valid.dtype)
        valid = xp.where(warmup, all_valid, post_warmup_valid)

        safe_data = xp.where(warmup, data, xp.where(valid, data, last_valid))

        new_fallback_state, fallback_value = self.fallback.apply(
            fallback_state, safe_data
        )

        new_last_valid_post = xp.where(
            last_valid_initialized,
            xp.where(valid, data, fallback_value),
            mean,
        )
        new_last_valid = xp.where(warmup, last_valid, new_last_valid_post)
        new_last_valid_initialized = xp.where(
            warmup, last_valid_initialized, xp.asarray(True)
        )

        out = xp.where(warmup, data, xp.where(valid, data, fallback_value))

        # Push policy: during warmup push raw data; post-warmup push
        # `new_last_valid` only if at least one element was valid (preserves
        # std collapse safeguard from the eager implementation).
        any_valid = xp.any(valid)
        do_push = xp.logical_or(warmup, any_valid)
        push_value = xp.where(warmup, data, new_last_valid)
        pushed_buf = _buffer_set(buf, index, push_value)
        new_buf = xp.where(do_push, pushed_buf, buf)
        new_index = xp.where(do_push, (index + 1) % self.window_size, index)
        new_n_seen = xp.where(do_push, n_seen + 1, n_seen)

        new_state = (
            new_buf,
            new_index,
            new_n_seen,
            new_last_valid,
            new_last_valid_initialized,
            new_fallback_state,
        )
        return new_state, out

    def _name(self) -> str:
        return (
            f"StdRejectFilter("
            f"std_factor={self.std_factor},"
            f"window_size={self.window_size},"
            f"fallback={self.fallback.name})"
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

    def init_state(self, data: Array) -> State:
        """Allocate per-child state.

        All built-in filters preserve shape and dtype, so each child's
        `init_state` is called with the original `data`.

        Args:
            data: Sample input.

        Returns:
            Tuple of child states, one per filter.
        """
        return tuple(f.init_state(data) for f in self.filters)

    def apply(self, state: State, data: Array) -> tuple[State, Array]:
        """Apply all filters sequentially.

        Args:
            state: Tuple of child states.
            data: Input array.

        Returns:
            ``(new_state, output)`` after applying each filter in order.
        """
        new_states = []
        out = data
        for f, s in zip(self.filters, state, strict=True):
            s_new, out = f.apply(s, out)
            new_states.append(s_new)
        return tuple(new_states), out

    def _name(self) -> str:
        """Return a short descriptive name.

        Returns:
            Name string listing the names of the concatenated filters.
        """
        return f"ConcatFilter({[f.name for f in self.filters]})"


AVAILABLE_FILTERS: dict[str, type[Filter]] = {
    "MinMaxFilter": MinMaxFilter,
    "AverageFilter": AverageFilter,
    "ExpAverageFilter": ExpAverageFilter,
    "MedianFilter": MedianFilter,
    "QuantizationFilter": QuantizationFilter,
    "ScaleFilter": ScaleFilter,
    "RangeRejectFilter": RangeRejectFilter,
    "StdRejectFilter": StdRejectFilter,
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
        ValueError:
            If an unknown filter type is encountered or parameters are invalid.
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
