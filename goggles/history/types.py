"""Shared type aliases for history package."""

from __future__ import annotations
from typing import Dict
import jax.numpy as jnp

PRNGKey = jnp.ndarray
Array = jnp.ndarray
History = Dict[str, Array]
