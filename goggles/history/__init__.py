"""Device-resident temporal history buffers for JAX pipelines.

This package provides typed specifications and interfaces for constructing,
updating, and slicing temporal histories stored on device.

Public API:
    - HistoryFieldSpec
    - HistorySpec
    - create_history
    - update_history
    - slice_history
    - peek_last
"""

from __future__ import annotations

from .buffer import create_history, update_history
from .spec import HistoryFieldSpec, HistorySpec
from .utils import peek_last, slice_history, to_device, to_host

__all__ = [
    "HistoryFieldSpec",
    "HistorySpec",
    "create_history",
    "peek_last",
    "slice_history",
    "to_device",
    "to_host",
    "update_history",
]
