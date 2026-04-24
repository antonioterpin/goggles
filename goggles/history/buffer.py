"""Creation and update interfaces for device-resident history buffers."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .spec import HistorySpec
from .types import Array, History, PRNGKey


def _apply_reset(
    hist_row: Array,
    new_row: Array,
    reset: Array,
    init_mode: str,
    key: PRNGKey | None = None,
) -> Array:
    """Shift and optionally reset a single history row.

    This uses JAX-friendly control flow (lax.cond) so it can be jitted/vmap'd
    without attempting Python-level boolean conversions of tracers.

    Args:
        hist_row: Array shaped (T, *shape) for a single batch row.
        new_row: Array shaped (1, *shape), appended at the end along time.
        reset: Boolean scalar (0-dim) indicating whether to reset this row.
        init_mode: One of {"zeros", "ones", "randn", "none"}.
        key: PRNGKey | None (shape (2,)) used when `init_mode == "randn"`.

    Returns:
        Array with the same shape as `hist_row`, updated for this step.
    """
    shifted_row = jnp.concatenate([hist_row[1:], new_row], axis=0)

    if init_mode == "none":
        return shifted_row

    def do_reset(_):
        if init_mode == "zeros":
            return jnp.zeros_like(hist_row)
        if init_mode == "ones":
            return jnp.ones_like(hist_row)
        if init_mode == "randn":
            if key is None:
                raise ValueError("init_mode 'randn' requires a PRNG key")
            return jax.random.normal(key, hist_row.shape, hist_row.dtype)
        raise ValueError(f"Unknown init mode {init_mode!r}")

    return jax.lax.cond(reset, do_reset, lambda _: shifted_row, operand=None)


def create_history(
    spec: HistorySpec, batch_size: int, rng: PRNGKey | None = None
) -> History:
    """Allocate device-resident history tensors following (B, T, *shape).

    Args:
        spec: Describing each field.
        batch_size: Batch size (B).
        rng: Optional PRNG key for randomized initialization
            of the buffers (e.g., for initial values or noise).

    Returns:
        dict (History): Mapping field name to array shaped (B, T, *shape).

    Raises:
        ValueError: If batch_size <= 0 or invalid spec values.

    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    history: History = {}
    for name, field in spec.fields.items():
        # Validate length
        if field.length <= 0:
            raise ValueError(f"Invalid history length for field '{name}'")
        shape = (batch_size, field.length, *field.shape)

        # Initialize according to policy
        if field.init == "zeros":
            arr = jnp.zeros(shape, field.dtype)
        elif field.init == "ones":
            arr = jnp.ones(shape, field.dtype)
        elif field.init == "randn":
            if rng is None:
                raise ValueError(f"Field '{name}' requires rng for randn init")
            rng, sub = jax.random.split(rng)
            arr = jax.random.normal(sub, shape, field.dtype)
        elif field.init == "none":
            arr = jnp.empty(shape, field.dtype)
        else:
            raise ValueError(
                f"Unknown init mode {field.init!r} for field '{name}'"
            )
        history[name] = arr
    return history


def _validate_update_entry(name: str, hist: Array, new: Array) -> None:
    """Validate that ``new`` can be appended to ``hist`` for field ``name``.

    Args:
        name: Field name (used in error messages).
        hist: Existing history array ``(B, T, *shape)``.
        new: New entry array ``(B, 1, *shape)``.

    Raises:
        ValueError: If rank, append length, or dtype mismatch the existing
            history.
    """
    if new.ndim != hist.ndim:
        raise ValueError(
            f"Dim mismatch for field '{name}': {new.shape} vs {hist.shape}"
        )
    if new.shape[1] != 1:
        raise ValueError(f"Append length must be 1 for field '{name}'")
    if new.dtype != hist.dtype:
        raise ValueError(f"Dtype mismatch for field '{name}'")


def _resolve_reset_keys(
    name: str, rng: jax.Array | None, init_mode: str, batch_size: int
) -> Array:
    """Return per-batch PRNG keys for reset initialization.

    Args:
        name: Field name (used in error messages).
        rng: Either a single PRNGKey of shape ``(2,)`` or per-batch keys of
            shape ``(B, 2)``.
        init_mode: Field init mode; randomized modes (``"randn"``) require
            a non-``None`` ``rng``.
        batch_size: Leading batch dimension ``B`` of the target history.

    Returns:
        A ``(B, 2)`` uint32 array of PRNG keys. Dummy zeros are returned
        when ``init_mode`` is not randomized — callers pass the result to
        ``jax.vmap`` for uniformity.

    Raises:
        ValueError: If ``rng`` is missing for a randomized init or has an
            unsupported shape.
    """
    if init_mode != "randn":
        return jnp.zeros((batch_size, 2), dtype=jnp.uint32)
    if rng is None:
        raise ValueError(f"Field '{name}' requires rng for randn reset")
    rng_arr = jnp.asarray(rng)
    if rng_arr.ndim == 1:
        return jax.random.split(rng_arr, batch_size)
    if rng_arr.ndim == 2 and rng_arr.shape[0] == batch_size:
        return rng_arr
    raise ValueError(
        "rng must be a PRNGKey (shape (2,)) or per-batch keys "
        f"with shape (B, 2); got {tuple(rng_arr.shape)}"
    )


def update_history(
    history: History,
    new_data: dict[str, Array],
    reset_mask: Array | None = None,
    spec: HistorySpec | None = None,
    rng: jax.Array | None = None,
) -> History:
    """Shift and append new items along the temporal axis.

    Note: this function can be jitted and vmapped over batch dimensions.
    RNG handling:
    if `rng` is provided, it may be either a single PRNGKey or an array of
    per-batch keys with shape (B, 2). This lets callers supply already-sharded
    keys for multi-device/pmap scenarios.

    Args:
        history: Current history dict (B, T, *shape).
        new_data: New entries per field, shaped (B, 1, *shape).
        reset_mask: Optional boolean mask for resets (B,).
        spec: Optional spec describing reset initialization.
        rng: Optional PRNG key for randomized resets.

    Returns:
        History: Updated history dict.

    Raises:
        ValueError: If shapes, dtypes, or append lengths are invalid.

    """
    updated: History = {}

    for name, hist in history.items():
        if name not in new_data:
            raise ValueError(f"Missing new data for field '{name}'")
        new = new_data[name]
        _validate_update_entry(name, hist, new)

        if reset_mask is None:
            updated[name] = jnp.concatenate([hist[:, 1:, ...], new], axis=1)
            continue

        if reset_mask.ndim != 1 or reset_mask.shape[0] != hist.shape[0]:
            raise ValueError(
                f"Invalid reset_mask shape {reset_mask.shape}, expected (B,)"
            )

        init_mode = (
            spec.fields[name].init
            if spec is not None
            and hasattr(spec, "fields")
            and name in spec.fields
            else "zeros"
        )
        keys = _resolve_reset_keys(name, rng, init_mode, hist.shape[0])

        def apply(h, n, r, k, init_mode=init_mode):
            return _apply_reset(h, n, r, init_mode, k)

        updated[name] = jax.vmap(apply)(
            hist, new[:, 0:1, ...], reset_mask, keys
        )

    return updated
