"""Utility functions for history slicing and inspection."""

from __future__ import annotations

from typing import Optional

from .types import Array, HistoryDict


def slice_history(
    history: HistoryDict,
    start: int,
    length: int,
    field: Optional[str] = None,
) -> HistoryDict | Array:
    """Return a temporal slice [start : start+length] along the time axis.

    Args:
        history (HistoryDict): Mapping field -> array of shape (B, T, *payload).
        start (int): Starting timestep (0-based).
        length (int): Number of timesteps to include (> 0).
        field (Optional[str]): If provided, slice only this field and return the
            array instead of a dict.

    Returns:
        HistoryDict | Array: A dict with the same keys and sliced values, or a
        single array if `field` is specified.

    Raises:
        ValueError: If `length` <= 0, `start` out of bounds, or slice exceeds T.
        KeyError: If `field` is not present in `history`.
        TypeError: If `history` is empty or contains tensors with rank < 2.
    """
    if length <= 0:
        raise ValueError("length must be > 0")
    if not history:
        raise TypeError("history must be a non-empty mapping")

    # Infer T from any field and validate ranks.
    any_arr = next(iter(history.values()))
    if any_arr.ndim < 2:
        raise TypeError("history arrays must have rank >= 2 (B, T, ...)")
    T = any_arr.shape[1]

    if start < 0 or start >= T:
        raise ValueError(f"start={start} out of bounds for T={T}")
    if start + length > T:
        raise ValueError(f"Slice [{start}:{start+length}] exceeds T={T}")

    if field is not None:
        if field not in history:
            raise KeyError(f"Unknown field {field!r}")
        arr = history[field]
        if arr.ndim < 2:
            raise TypeError(f"Field {field!r} must have rank >= 2 (B, T, ...)")
        return arr[:, start : start + length, ...]
    else:
        return {k: v[:, start : start + length, ...] for k, v in history.items()}


def peek_last(history: HistoryDict, k: int = 1) -> HistoryDict:
    """Return the last `k` timesteps for all fields.

    Args:
        history (HistoryDict): Mapping field -> array of shape (B, T, *payload).
        k (int): Number of trailing timesteps to select (1 ≤ k ≤ T).

    Returns:
        HistoryDict: Mapping field -> sliced array of shape (B, k, *payload).

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
