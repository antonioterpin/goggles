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

from .spec import HistoryFieldSpec, HistorySpec
from .buffer import create_history, update_history
from .utils import slice_history, peek_last, to_device, to_host

__all__ = [
    "HistoryFieldSpec",
    "HistorySpec",
    "create_history",
    "update_history",
    "slice_history",
    "peek_last",
    "to_device",
    "to_host",
]
