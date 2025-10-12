# goggles/history/utils.py
"""Utility functions for history slicing and inspection."""

from __future__ import annotations
from typing import Dict, Optional
import jax.numpy as jnp
from goggles.history.types import Array, HistoryDict


def slice_history(
    history: HistoryDict,
    start: int,
    length: int,
    field: Optional[str] = None,
) -> HistoryDict | Array:
    """Return a temporal slice [start : start+length] along the time axis."""
    raise NotImplementedError


def peek_last(history: HistoryDict, k: int = 1) -> HistoryDict:
    """Return the last `k` timesteps for all fields."""
    raise NotImplementedError
