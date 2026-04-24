"""Type specifications for device-resident history buffers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

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

        May propagate ``ValueError`` from :func:`_validate_field_spec` /
        :func:`_field_spec_from_mapping` when required keys are missing
        or values are invalid (e.g. ``length < 1``, negative dims,
        unknown init mode).

        Args:
            config: Dict mapping field name to kwargs for
                `HistoryFieldSpec` or to an already-built `HistoryFieldSpec`.
                Each kwargs dict must include:
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

        """
        if not isinstance(config, Mapping):
            raise TypeError("config must be a Mapping[str, Any].")

        out: dict[str, HistoryFieldSpec] = {}
        for name, spec in config.items():
            if isinstance(spec, HistoryFieldSpec):
                _validate_field_spec(name, spec)
                out[name] = spec
            elif isinstance(spec, Mapping):
                out[name] = _field_spec_from_mapping(name, spec)
            else:
                raise TypeError(
                    f"Field {name!r} must be a Mapping or HistoryFieldSpec, "
                    f"got {type(spec).__name__}."
                )
        return cls(fields=out)


_ALLOWED_INITS: tuple[str, ...] = ("zeros", "ones", "randn", "none")


def _validate_field_spec(name: str, spec: HistoryFieldSpec) -> None:
    """Validate an already-built :class:`HistoryFieldSpec`.

    Args:
        name: Field name (used in error messages).
        spec: Field spec to validate in place.

    Raises:
        ValueError: If ``length``/``shape``/``init`` fail the invariants.
    """
    if not isinstance(spec.length, int) or spec.length < 1:
        raise ValueError(
            f"{name!r}.length must be an int >= 1, got {spec.length}."
        )
    if any((not isinstance(d, int) or d < 0) for d in spec.shape):
        raise ValueError(
            f"{name!r}.shape must be a tuple of non-negative ints, "
            f"got {spec.shape}."
        )
    if spec.init not in _ALLOWED_INITS:
        raise ValueError(
            f"{name!r}.init must be one of {_ALLOWED_INITS}, got {spec.init}."
        )


def _field_spec_from_mapping(
    name: str, spec: Mapping[str, Any]
) -> HistoryFieldSpec:
    """Build a :class:`HistoryFieldSpec` from a kwargs mapping.

    Args:
        name: Field name (used in error messages).
        spec: Mapping with ``length``/``shape`` plus optional
            ``dtype``/``init`` keys.

    Returns:
        A validated :class:`HistoryFieldSpec`.

    Raises:
        TypeError: If ``shape`` or ``dtype`` are malformed.
        ValueError: If required keys are missing or values are invalid.
    """
    if "length" not in spec or "shape" not in spec:
        raise ValueError(
            f"Field {name!r} must define 'length' and 'shape'. "
            f"Got keys: {list(spec.keys())}."
        )

    length = spec["length"]
    if not isinstance(length, int) or length < 1:
        raise ValueError(f"{name!r}.length must be an int >= 1, got {length}.")

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

    try:
        dtype = jnp.dtype(spec.get("dtype", jnp.float32))
    except Exception as e:
        raise TypeError(
            f"{name!r}.dtype is not a valid JAX dtype: {spec.get('dtype')!r}."
        ) from e

    init = spec.get("init", "zeros")
    if init not in _ALLOWED_INITS:
        raise ValueError(
            f"{name!r}.init must be one of {_ALLOWED_INITS}, got {init!r}."
        )

    return HistoryFieldSpec(
        length=length, shape=shape_tuple, dtype=dtype, init=init
    )
