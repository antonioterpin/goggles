# goggles/history/buffer.py
"""Creation and update interfaces for GPU-resident history buffers."""

from __future__ import annotations
from typing import Dict, Optional
import jax.numpy as jnp
from .spec import HistorySpec

PRNGKey = jnp.ndarray
Array = jnp.ndarray
HistoryDict = Dict[str, Array]


def create_history(
    spec: HistorySpec, batch_size: int, rng: Optional[PRNGKey] = None
) -> HistoryDict:
    """Allocate GPU-resident history tensors following (B, T, *shape).

    Args:
        spec: HistorySpec describing each field.
        batch_size: Batch size (B).
        rng: Optional PRNG key for randomized initialization.

    Returns:
        Mapping field → array shaped (B, T, *shape).

    Raises:
        NotImplementedError: Placeholder until implementation sub-issue.
    """
    raise NotImplementedError


def update_history(
    history: HistoryDict,
    new_data: Dict[str, Array],
    reset_mask: Optional[Array] = None,
) -> HistoryDict:
    """Shift and append new items along the temporal axis.

    Args:
        history: Current history dict (B, T, *shape).
        new_data: New entries per field.
        reset_mask: Optional boolean mask for resets (B,).

    Returns:
        Updated history dict.

    Raises:
        NotImplementedError: Placeholder until implementation sub-issue.
    """
    raise NotImplementedError
