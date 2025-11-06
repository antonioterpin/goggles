"""Shared type aliases for history package."""

from __future__ import annotations
from typing import TypeAlias
import jax

PRNGKey: TypeAlias = "jax.Array"
Array: TypeAlias = "jax.Array"
History: TypeAlias = dict[str, Array]
