"""Utility functions for history slicing and inspection."""

from __future__ import annotations

from typing import Optional, Sequence

from .types import Array, History


def slice_history(
    history: History,
    start: int,
    length: int,
    fields: Optional[Sequence[str] | str] = None,
) -> History | Array:
    """Return a temporal slice [start : start+length] for selected fields.

    Args:
        history (History): Mapping field -> array of shape (B, T, ...).
        start (int): Starting timestep (0-based).
        length (int): Number of timesteps to include (> 0).
        fields (Optional[Sequence[str] | str]): One or more field names to slice.
            If a single string is provided, only that field is sliced.
            If a list or tuple is provided, all listed fields are sliced.
            If None, all fields in `history` are sliced.

    Returns:
        History: Mapping of sliced arrays with shape (B, length, ...).

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
        history (History): Mapping field -> array of shape (B, T, *payload).
        k (int): Number of trailing timesteps to select (1 ≤ k ≤ T).

    Returns:
        History: Mapping field -> sliced array of shape (B, k, *payload).

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
