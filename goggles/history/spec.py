# goggles/history/spec.py
"""Type specifications for GPU-resident history buffers."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Literal, Mapping
import jax.numpy as jnp

InitMode = Literal["zeros", "ones", "randn", "none"]


@dataclass(frozen=True)
class HistoryFieldSpec:
    """Describe one temporal field stored on device.

    Attributes:
        length: Number of stored timesteps for this field.
        shape: Per-timestep payload shape (no batch/time dims).
        dtype: Array dtype.
        init: Initialization policy ("zeros" | "ones" | "randn" | "none").
    """

    length: int
    shape: tuple[int, ...]
    dtype: jnp.dtype = jnp.float32
    init: InitMode = "zeros"


@dataclass(frozen=True)
class HistorySpec:
    """Bundle multiple named history field specs.

    Attributes:
        fields: Mapping from field name to `HistoryFieldSpec`.
    """

    fields: Mapping[str, HistoryFieldSpec]

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "HistorySpec":
        """Construct a HistorySpec from a nested config dictionary.

        Args:
            config: Dict mapping field name to kwargs for `HistoryFieldSpec`.

        Returns:
            Parsed `HistorySpec` object.
        """
        raise NotImplementedError
