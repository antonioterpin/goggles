"""Filter classes for processing raw values."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax
import numpy as np
import jax.numpy as jnp
from types import ModuleType


def get_backend(x) -> ModuleType:
    """Returns the appropriate numerical backend (numpy or jax.numpy) based on 'x'.

    Args:
        x: Input array.

    Returns:
        The numerical backend module (numpy or jax.numpy).
    """
    if isinstance(x, jax.Array):
        return jnp
    return np


@dataclass(frozen=True)
class FilterConfig:
    """Configuration for a filter."""

    type: str
    parameters: dict


class Filter(ABC):
    """Base class for filters.

    All filters must implement the following methods:

    - ``step(data)``: Processes the input ``data`` and returns the filtered value.
    - ``reset()``: Resets the filter state.
    - ``_name()``: Returns a string representation of the filter.
    """

    def __init__(self, prefix: str = "") -> None:
        """Initializes the Filter base class.

        Args:
            prefix (str): Prefix for the filter name.
        """
        self.prefix = prefix

    def __call__(self, data: np.ndarray | jax.Array) -> np.ndarray | jax.Array:
        """Convenience method to call the step method.

        Args:
            data: The input value to be filtered.

        Returns:
            The filtered value.
        """
        return self.step(data)

    @abstractmethod
    def step(self, data: np.ndarray | jax.Array) -> np.ndarray | jax.Array:
        """Filters the input 'data' and returns the filtered value.

        Args:
            data: The input value to be filtered.

        Returns:
            The filtered value.
        """

    @abstractmethod
    def reset(self) -> None:
        """Resets the filter state."""

    @property
    def name(self) -> str:
        """Returns a string representation of the filter.

        Returns:
            A string representation of the filter.
        """
        return self.prefix + self._name()

    @abstractmethod
    def _name(self) -> str:
        """Returns a string representation of the filter.

        Returns:
            A string representation of the filter.
        """


class ScaleFilter(Filter):
    """Scales the input data by a constant factor."""

    def __init__(self, scale: float, prefix: str = ""):
        """Initializes the ScaleFilter.

        Args:
            scale: Multiplicative scale factor.
            prefix: Prefix for the filter name.
        """
        if not isinstance(scale, (int, float)):
            raise ValueError("scale must be a numeric value")
        self.scale = float(scale)
        super().__init__(prefix=prefix)

    def step(self, data: np.ndarray | jax.Array) -> np.ndarray | jax.Array:
        """Scales the input data.

        Args:
            data: Input array.

        Returns:
            Scaled data.
        """
        return data * self.scale

    def reset(self) -> None:
        """Resets the filter state.

        ScaleFilter is stateless, so this is a no-op.
        """

    def _name(self) -> str:
        """Returns a string representation of the filter."""
        return f"ScaleFilter(scale={self.scale})"


class MinMaxFilter(Filter):
    """Scales 'data' by 'max_abs_value' and clips to [0, 1]."""

    def __init__(self, min_val: float, max_val: float, prefix: str = ""):
        """Initializes the MinMaxFilter.

        Args:
            min_val: The minimum value to scale 'data' by.
            max_val: The maximum value to scale 'data' by.
            prefix: Prefix for the filter name.
        """
        if min_val >= max_val:
            raise ValueError("min_val must be less than max_val")
        self.min_val = min_val
        self.max_val = max_val
        super().__init__(prefix=prefix)

    def step(self, data: np.ndarray | jax.Array | float) -> np.ndarray | jax.Array:
        """Scales 'data' by 'max_abs_value' and clips to [0, 1].

        Args:
            data: The input value to be filtered.

        Returns:
            The filtered value.
        """
        xp = get_backend(data)
        out = xp.clip((data - self.min_val) / (self.max_val - self.min_val), 0, 1)
        return out

    def reset(self) -> None:
        """Resets the filter state. No state to reset for MinMaxFilter."""

    def _name(self) -> str:
        """Returns a string representation of the filter."""
        return f"MinMaxFilter({self.min_val},{self.max_val})"


class AverageFilter(Filter):
    """Computes a simple moving average of the last 'window_size' values."""

    def __init__(self, window_size: int, prefix: str = ""):
        """Initializes the AverageFilter.

        Args:
            window_size: The number of values to average.
            prefix: Prefix for the filter name.
        """
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("window_size must be positive")
        self.window_size = window_size
        self.buffer: np.ndarray | jax.Array | None = None
        self.index = 0  # Current index in the buffer
        self.count = 0  # Number of values added to the buffer
        super().__init__(prefix=prefix)

    def step(self, data: np.ndarray | jax.Array) -> np.ndarray | jax.Array:
        """Computes a simple moving average of the last 'window_size' values.

        Args:
            data: The input value to be filtered.

        Returns:
            The filtered value.
        """
        self.count += 1
        xp = get_backend(data)
        if self.buffer is None:
            self.buffer = xp.zeros((self.window_size,) + data.shape, dtype=data.dtype)

        if isinstance(data, jax.Array):
            self.buffer = self.buffer.at[self.index].set(data)  # type: ignore[attr-defined]
        else:
            self.buffer[self.index] = data  # type: ignore[index]

        valid_length = min(self.count, self.window_size)
        valid_buffer = self.buffer[:valid_length]  # type: ignore[index]
        out = xp.mean(valid_buffer, axis=0)
        self.index = (self.index + 1) % self.window_size
        return out

    def reset(self) -> None:
        """Resets the filter state. Clears the buffer."""
        self.buffer = None
        self.index = 0
        self.count = 0

    def _name(self) -> str:
        """Returns a string representation of the filter."""
        return f"AverageFilter({self.window_size})"


class ExpAverageFilter(Filter):
    """Exponential moving average filter.

    Computes an exponential moving average with smoothing factor 'alpha'.
    y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
    """

    def __init__(self, alpha: float, prefix: str = ""):
        """Initializes the ExpAverageFilter.

        Args:
            alpha: The smoothing factor.
            prefix: Prefix for the filter name.
        """
        if alpha < 0 or alpha > 1:
            raise ValueError("alpha must be in the range [0, 1]")
        self.alpha = alpha
        self.value: np.ndarray | jax.Array | None = None
        super().__init__(prefix=prefix)

    def step(self, data: np.ndarray | jax.Array) -> np.ndarray | jax.Array:
        """Computes an exponential moving average of 'data'.

        Args:
            data: The input value to be filtered.

        Returns:
            The filtered value.
        """
        if self.value is None:
            # Initialize the average to the first input
            self.value = data
        self.value = self.alpha * data + (1 - self.alpha) * self.value
        return self.value

    def reset(self) -> None:
        """Resets the filter state. Clears the average value."""
        self.value = None

    def _name(self) -> str:
        """Returns a string representation of the filter."""
        return f"ExpAverageFilter({self.alpha})"


class MedianFilter(Filter):
    """Computes a median of the last 'window_size' values."""

    def __init__(self, window_size: int, prefix: str = ""):
        """Initializes the MedianFilter.

        Args:
            window_size: The number of values to compute the median of.
            prefix: Prefix for the filter name.
        """
        if (
            # TODO: Update once goggles validation supports it
            not isinstance(window_size, int)
            or window_size <= 0
        ):
            raise ValueError("window_size must be positive")
        self.window_size = window_size
        self.buffer: np.ndarray | jax.Array | None = None
        self.index = 0  # Current index in the buffer
        self.count = 0  # Number of values added to the buffer
        super().__init__(prefix=prefix)

    def step(self, data: np.ndarray | jax.Array) -> np.ndarray | jax.Array:
        """Computes the median of the last 'window_size' values.

        Args:
            data: The input value to be filtered.

        Returns:
            The filtered value.
        """
        self.count += 1
        xp = get_backend(data)
        if self.buffer is None:
            self.buffer = xp.zeros((self.window_size,) + data.shape, dtype=data.dtype)

        if isinstance(data, jax.Array):
            self.buffer = self.buffer.at[self.index].set(data)  # type: ignore[attr-defined]
        else:
            self.buffer[self.index] = data  # type: ignore[index]
        self.index = (self.index + 1) % self.window_size

        valid_length = min(self.count, self.window_size)
        valid_buffer = self.buffer[:valid_length]  # type: ignore[index]
        out = xp.median(valid_buffer, axis=0)
        return out

    def reset(self) -> None:
        """Resets the filter state. Clears the buffer."""
        self.buffer = None
        self.index = 0
        self.count = 0

    def _name(self) -> str:
        """Returns a string representation of the filter."""
        return f"MedianFilter({self.window_size})"


class QuantizationFilter(Filter):
    """Quantization filter for values.

    Clamps the value to [min_value, max_value] and quantizes it
    in steps of 'step_size'.
    """

    def __init__(
        self,
        min_value: float = -0.150,
        max_value: float = 0.150,
        step_size: float = 0.00015,
        prefix: str = "",
    ) -> None:
        """Initializes the QuantizationFilter.

        Args:
            min_value: The minimum value to clamp to.
            max_value: The maximum value to clamp to.
            step_size: The quantization step size.
            prefix: Prefix for the filter name.
        """
        self.min_value = min_value
        self.max_value = max_value
        self.step_size = step_size
        self.levels: np.ndarray | jax.Array | None = None

        super().__init__(prefix=prefix)

    def step(self, data: np.ndarray | jax.Array) -> np.ndarray | jax.Array:
        """Clamps the value to [min_value, max_value] and quantizes it.

        Args:
            data: The input value to be filtered.

        Returns:
            The filtered value.
        """
        xp = get_backend(data)

        # Create an array of all possible quantization levels. if not already created.
        # For example: -0.150, -0.14985, -0.14970, ..., 0.150.
        if self.levels is None:
            self.levels = xp.arange(
                self.min_value, self.max_value + self.step_size, self.step_size
            )[..., xp.newaxis]

        # Step 1: Clamp to [min_value, max_value].
        clipped = xp.clip(data, self.min_value, self.max_value)
        # Step 2: Find the nearest quantization level.
        idx = xp.argmin(xp.abs(self.levels - clipped), axis=0)
        out = self.levels[idx]  # type: ignore[index]
        return out.squeeze(axis=-1)

    def reset(self) -> None:
        """Resets the filter state. No state to reset for QuantizationFilter."""

    def _name(self) -> str:
        """Returns a string representation of the filter."""
        return f"QuantizationFilter(min={self.min_value}, max={self.max_value}, step={self.step_size})"


class ConcatFilter(Filter):
    """Applies multiple filters in sequence.

    It passes the output of each filter as the input to the next filter.
    """

    def __init__(self, filters: list[Filter]) -> None:
        """Initializes the ConcatFilter.

        Args:
            filters: A list of Filter objects to apply in sequence.
        """
        self.filters = filters
        super().__init__(prefix="")

    def step(self, data: np.ndarray | jax.Array) -> np.ndarray | jax.Array:
        """Applies the filters in sequence to 'data'.

        Args:
            data: The input value to be filtered.

        Returns:
            The filtered value.
        """
        for f in self.filters:
            data = f.step(data)
        return data

    def reset(self) -> None:
        """Resets the filter state. Resets each filter in the sequence."""
        for f in self.filters:
            f.reset()

    def _name(self) -> str:
        """Returns a string representation of the filter."""
        return f"ConcatFilter({[f.name for f in self.filters]})"


AVAILABLE_FILTERS = {
    "MinMaxFilter": MinMaxFilter,
    "AverageFilter": AverageFilter,
    "ExpAverageFilter": ExpAverageFilter,
    "MedianFilter": MedianFilter,
    "QuantizationFilter": QuantizationFilter,
    "ConcatFilter": ConcatFilter,
}


def create_concat_filter(filter_configs: list[FilterConfig]) -> ConcatFilter:
    """Creates a ConcatFilter from a list of filter configurations.

    Args:
        filter_configs: A list of filter configurations.

    Returns:
        A ConcatFilter object.
    """
    filters = []
    for id, config in enumerate(filter_configs):
        try:
            filter_class = AVAILABLE_FILTERS[config.type]
            filters.append(filter_class(**config.parameters, prefix=f"[{id}] "))
        except Exception as e:
            raise ValueError(
                f"Invalid filter class or parameters for {config.type}. {e}"
            ) from e

    return ConcatFilter(filters)
