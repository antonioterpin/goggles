"""Utility functions for history slicing and inspection."""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jaxlib.xla_client as xc

Device = xc.Device

from .types import History


def slice_history(
    history: History,
    start: int,
    length: int,
    fields: Sequence[str] | None = None,
) -> History:
    """Return a temporal slice [start : start+length] for selected fields.

    Args:
        history: Mapping field -> array of shape (B, T, ...).
        start: Starting timestep (0-based).
        length: Number of timesteps to include (> 0).
        fields: One or more field names to slice.
            If a single string is provided, only that field is sliced.
            If a list or tuple is provided, all listed fields are sliced.
            If None, all fields in `history` are sliced.

    Returns:
        Mapping of sliced arrays with shape (B, length, ...).

    Raises:
        ValueError: If `length` <= 0, `start` out of bounds, or slice exceeds T.
        KeyError: If `fields` is not present in `history`.
        TypeError: If `history` is empty or contains tensors with rank < 2.

    """
    # Validate length and history
    if length <= 0:
        raise ValueError("length must be > 0")
    if not history:
        raise TypeError("history must be a non-empty mapping")

    # Validate reference array and slice bounds
    any_arr = next(iter(history.values()))
    if any_arr.ndim < 2:
        raise TypeError("history arrays must have rank >= 2 (B, T, ...)")
    T = any_arr.shape[1]
    if start < 0 or start + length > T:
        raise ValueError(f"Invalid slice [{start}:{start+length}] for T={T}")

    # Normalize and validate `fields`
    if fields is None:
        keys = list(history.keys())
    elif isinstance(fields, str):
        keys = [fields]
    elif isinstance(fields, (list, tuple)):
        if not fields:
            raise ValueError("fields list is empty.")
        if not all(isinstance(f, str) for f in fields):
            raise TypeError("All field names must be strings.")
        keys = list(fields)
    else:
        raise TypeError("fields must be a string, list/tuple of strings, or None")

    # Check that all requested fields exist
    missing = set(keys) - set(history)
    if missing:
        raise KeyError(f"Unknown fields: {missing}")

    # Validate ranks for selected fields
    for k in keys:
        if history[k].ndim < 2:
            raise TypeError(f"Field {k!r} must have rank >= 2 (B, T, ...)")

    return {k: history[k][:, start : start + length, ...] for k in keys}


def peek_last(history: History, k: int = 1) -> History:
    """Return the last `k` timesteps for all fields.

    Args:
        history: Mapping field -> array of shape (B, T, *payload).
        k: Number of trailing timesteps to select (1 ≤ k ≤ T).

    Returns:
        Mapping field -> sliced array of shape (B, k, *payload).

    Raises:
        ValueError: If `k` < 1 or `k` > T for any field.
        TypeError: If `history` is empty or contains tensors with rank < 2.

    """
    if not history:
        raise TypeError("history must be a non-empty mapping")

    any_arr = next(iter(history.values()))
    if any_arr.ndim < 2:
        raise TypeError("history arrays must have rank >= 2 (B, T, ...)")
    T = any_arr.shape[1]

    if k < 1 or k > T:
        raise ValueError(f"k must be in [1, T]; got k={k}, T={T}")

    # Use negative slicing for clarity and to keep JAX-friendly semantics.
    return {k_name: v[:, -k:, ...] for k_name, v in history.items()}


def to_device(
    history: History,
    devices: Sequence[Device] | None = None,  # type: ignore[type-arg]
    keys: tuple[str, ...] | None = None,
) -> History:
    """Move selected history arrays to one or more JAX devices.

    This function moves JAX arrays contained in a dictionary (or subset of it)
    to a target device or set of devices. If multiple devices are provided,
    arrays are distributed in a simple round-robin fashion across them.

    Non-array values (e.g., metadata, scalars, strings) are left unchanged.

    Args:
        history: Mapping field to array (or PyTree of arrays).
        devices: Target devices. Defaults to first device.
        keys: Subset of fields to move. If None, move all.

    Returns:
        Copy of the history with selected arrays placed on the target device(s).

    """
    devices = devices or jax.devices()

    # Select subset of keys if specified
    subset = history if keys is None else {k: history[k] for k in keys if k in history}

    moved = {}
    for i, (k, v) in enumerate(subset.items()):
        device = devices[i % len(devices)]  # Round-robin device selection

        # Recursively move PyTree leaves to the device
        moved[k] = jax.tree_util.tree_map(
            lambda x: jax.device_put(x, device) if isinstance(x, jax.Array) else x,
            v,
        )

    # If all keys were moved, just return the moved version
    if keys is None:
        return moved

    # Otherwise, merge moved subset back into the original dict
    return {**history, **moved}


def to_host(
    history: History,
    keys: tuple[str, ...] | None = None,
) -> History:
    """Copy selected history arrays from device to host memory.

    Recursively retrieves device arrays from JAX devices and copies them
    into host (NumPy) memory. Non-array values are left unchanged.

    Args:
        history (History): Mapping field to array (or PyTree of arrays).
        keys (tuple[str, ...] | None): Subset of fields to copy. If None, all.

    Returns:
        History: Copy of the history with arrays stored in host (NumPy) memory.

    Example:
        >>> host_history = to_host(device_history)
        >>> type(host_history["loss"])
        <class 'numpy.ndarray'>

    """
    subset = history if keys is None else {k: history[k] for k in keys if k in history}

    moved = {}
    for k, v in subset.items():
        # Recursively copy all arrays from device to host
        moved[k] = jax.tree_util.tree_map(
            lambda x: jax.device_get(x) if isinstance(x, jax.Array) else x,
            v,
        )

    if keys is None:
        return moved
    return {**history, **moved}
