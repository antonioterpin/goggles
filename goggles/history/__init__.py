# goggles/history/__init__.py
"""GPU-resident temporal history buffers for JAX pipelines.

This package provides typed specifications and interfaces for constructing,
updating, and slicing temporal histories stored on device.  It defines only
the public API contracts — functional implementations are provided in later
development stages.

Public API:
    - HistoryFieldSpec
    - HistorySpec
    - create_history
    - update_history
    - slice_history
    - peek_last
"""
from __future__ import annotations

try:
    import jax  # noqa: F401
    import jax.numpy as jnp  # noqa: F401
except ImportError as e:
    raise ImportError(
        "The 'goggles.history' module requires JAX. "
        "Install with `pip install goggles[jax]`."
    ) from e

from .spec import HistoryFieldSpec, HistorySpec
from .buffer import create_history, update_history
from .utils import slice_history, peek_last

__all__ = [
    "HistoryFieldSpec",
    "HistorySpec",
    "create_history",
    "update_history",
    "slice_history",
    "peek_last",
]
