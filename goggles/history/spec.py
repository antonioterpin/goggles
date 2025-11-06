"""Type specifications for device-resident history buffers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
from collections.abc import Mapping

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
        fields: Mapping from field name to spec

    """

    fields: Mapping[str, HistoryFieldSpec]

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> HistorySpec:
        """Construct a HistorySpec from a nested config dictionary.

        Args:
            config: Dict mapping field name to kwargs for
                `HistoryFieldSpec` or to an already-built `HistoryFieldSpec`. Each
                kwargs dict must include:
                - "length" (int): Number of timesteps (T >= 1).
                - "shape" (Sequence[int] | tuple[int, ...]): Per-timestep shape.
                Optional keys:
                - "dtype": Anything accepted by `jnp.dtype` (default float32).
                - "init": One of {"zeros", "ones", "randn", "none"}.

        Returns:
            Parsed specification bundle.

        Raises:
            TypeError: If `config` is not a mapping, or a field entry has
                unsupported type, or shapes/dtypes have invalid types.
            ValueError: If required keys are missing or values are invalid
                (e.g., length < 1, negative dims, unknown init mode).

        """
        if not isinstance(config, Mapping):
            raise TypeError("config must be a Mapping[str, Any].")

        allowed_inits: tuple[str, ...] = ("zeros", "ones", "randn", "none")
        out: dict[str, HistoryFieldSpec] = {}

        for name, spec in config.items():
            if isinstance(spec, HistoryFieldSpec):
                # Validate basic invariants even if user provided an instance.
                if not isinstance(spec.length, int) or spec.length < 1:
                    raise ValueError(
                        f"{name!r}.length must be an int >= 1, got {spec.length}."
                    )
                if any((not isinstance(d, int) or d < 0) for d in spec.shape):
                    raise ValueError(
                        f"{name!r}.shape must be a tuple of non-negative ints, "
                        f"got {spec.shape}."
                    )
                if spec.init not in allowed_inits:
                    raise ValueError(
                        f"{name!r}.init must be one of {allowed_inits}, got {spec.init}."
                    )
                out[name] = spec
                continue

            if not isinstance(spec, Mapping):
                raise TypeError(
                    f"Field {name!r} must be a Mapping or HistoryFieldSpec, "
                    f"got {type(spec).__name__}."
                )

            # Required keys
            if "length" not in spec or "shape" not in spec:
                raise ValueError(
                    f"Field {name!r} must define 'length' and 'shape'. Got keys: "
                    f"{list(spec.keys())}"
                )

            # Validate length
            length = spec["length"]
            if not isinstance(length, int) or length < 1:
                raise ValueError(f"{name!r}.length must be an int >= 1, got {length}.")

            # Validate shape
            shape_val = spec["shape"]
            if not isinstance(shape_val, (tuple, list)):
                raise TypeError(
                    f"{name!r}.shape must be a tuple/list of ints, "
                    f"got {type(shape_val).__name__}."
                )
            shape_tuple = tuple(int(d) for d in shape_val)
            if any(d < 0 for d in shape_tuple):
                raise ValueError(
                    f"{name!r}.shape must contain non-negative ints, got {shape_tuple}."
                )

            # Optional keys: dtype/init

            # Validate dtype
            try:
                dtype = jnp.dtype(spec.get("dtype", jnp.float32))
            except Exception as e:
                raise TypeError(
                    f"{name!r}.dtype is not a valid JAX dtype: {spec.get('dtype')!r}."
                ) from e

            # Validate init
            init = spec.get("init", "zeros")
            if init not in allowed_inits:
                raise ValueError(
                    f"{name!r}.init must be one of {allowed_inits}, got {init!r}."
                )

            out[name] = HistoryFieldSpec(
                length=length, shape=shape_tuple, dtype=dtype, init=init
            )

        return cls(fields=out)
