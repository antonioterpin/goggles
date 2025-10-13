"""Creation and update interfaces for device-resident history buffers."""
import jax
import jax.numpy as jnp
from typing import Dict, Optional
from .spec import HistorySpec
from .types import PRNGKey, Array, HistoryDict


def create_history(
    spec: HistorySpec, batch_size: int, rng: Optional[PRNGKey] = None
) -> HistoryDict:
    """Allocate device-resident history tensors following (B, T, *shape).

    Args:
        spec (HistorySpec): Describing each field.
        batch_size (int): Batch size (B).
        rng (Optional[PRNGKey]): Optional PRNG key for randomized initialization
            of the buffers (e.g., for initial values or noise).

    Returns:
        dict (HistoryDict): Mapping field name to array shaped (B, T, *shape).

    Raises:
        ValueError: If batch_size <= 0 or invalid spec values.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    history: HistoryDict = {}
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
            raise ValueError(f"Unknown init mode {field.init!r} for field '{name}'")
        history[name] = arr
    return history


def update_history(
    history: HistoryDict,
    new_data: Dict[str, Array],
    reset_mask: Optional[Array] = None,
    spec: Optional[HistorySpec] = None,
    rng: Optional[jax.Array] = None,
) -> HistoryDict:
    """Shift and append new items along the temporal axis.

    Note: this function can be jitted and vmapped over batch dimensions.
    RNG handling: if `rng` is provided, it may be either a single PRNGKey
    or an array of per-batch keys with shape (B, 2). This lets callers
    supply already-sharded keys for multi-device/pmap scenarios.

    Args:
        history (HistoryDict): Current history dict (B, T, *shape).
        new_data (Dict[str, Array]): New entries per field, shaped (B, 1, *shape).
        reset_mask (Optional[Array]): Optional boolean mask for resets (B,).
        spec (Optional[HistorySpec]): Optional spec describing reset initialization.
        rng (Optional[jax.Array]): Optional PRNG key for randomized resets.

    Returns:
        HistoryDict: Updated history dict.

    Raises:
        ValueError: If shapes, dtypes, or append lengths are invalid.
    """
    updated: HistoryDict = {}

    for name, hist in history.items():
        if name not in new_data:
            raise ValueError(f"Missing new data for field '{name}'")
        new = new_data[name]

        # Validate shapes/dtypes
        if new.ndim != hist.ndim:
            raise ValueError(
                f"Dim mismatch for field '{name}': {new.shape} vs {hist.shape}"
            )
        if new.shape[1] != 1:
            raise ValueError(f"Append length must be 1 for field '{name}'")
        if new.dtype != hist.dtype:
            raise ValueError(f"Dtype mismatch for field '{name}'")

        # Determine init mode for resets
        if spec is not None and name in spec.fields:
            init_mode = spec.fields[name].init
        else:
            init_mode = "zeros"

        def apply_reset(hist_row, new_row, reset, key=None):
            """Shift and optionally reset a single history row.

            This uses JAX-friendly control flow (lax.cond) so it can be jitted/vmap'd
            without attempting Python-level boolean conversions of tracers.
            """
            shifted_row = jnp.concatenate([hist_row[1:], new_row], axis=0)

            # If 'none', never reset (always return shifted_row).
            if init_mode == "none":
                return shifted_row

            # True-branch: produce an initialized buffer for this row.
            def do_reset(args):
                hrow, k = args
                if init_mode == "zeros":
                    return jnp.zeros_like(hrow)
                elif init_mode == "ones":
                    return jnp.ones_like(hrow)
                elif init_mode == "randn":
                    # At runtime (outside jitted/traced contexts) we enforce rng
                    if rng is None:
                        raise ValueError(f"Field '{name}' requires rng for randn reset")
                    return jax.random.normal(k, hrow.shape, hrow.dtype)
                else:
                    raise ValueError(
                        f"Unknown init mode {init_mode!r} for field '{name}'"
                    )

            # Use lax.cond so 'reset' can be a traced boolean.
            return jax.lax.cond(
                reset, do_reset, lambda args: shifted_row, (hist_row, key)
            )

        if reset_mask is None:
            updated_field = jnp.concatenate([hist[:, 1:, ...], new], axis=1)
        else:
            if reset_mask.ndim != 1 or reset_mask.shape[0] != hist.shape[0]:
                raise ValueError(
                    f"Invalid reset_mask shape {reset_mask.shape}, expected (B,)"
                )

            # Determine init mode for resets
            if init_mode == "randn":
                # If rng is None but no resets requested: no error; we use dummy keys.
                if rng is None:
                    if bool(jnp.any(reset_mask)):
                        raise ValueError(f"Field '{name}' requires rng for randn reset")
                    dummy_keys = jnp.zeros((hist.shape[0], 2), dtype=jnp.uint32)
                    updated_field = jax.vmap(apply_reset)(
                        hist, new, reset_mask, dummy_keys
                    )
                else:
                    rng_arr = jnp.asarray(rng)
                    # Single key -> split into per-batch keys.
                    if rng_arr.ndim == 1:
                        subkeys = jax.random.split(rng_arr, hist.shape[0])
                        updated_field = jax.vmap(apply_reset)(
                            hist, new, reset_mask, subkeys
                        )
                    # Per-batch keys -> use directly (must match batch size).
                    elif rng_arr.ndim == 2 and rng_arr.shape[0] == hist.shape[0]:
                        updated_field = jax.vmap(apply_reset)(
                            hist, new, reset_mask, rng_arr
                        )
                    else:
                        raise ValueError(
                            f"rng must be a PRNGKey or an array of per-batch keys "
                            f"with shape (B,2); got {rng_arr.shape}"
                        )
            else:
                dummy_keys = jnp.zeros((hist.shape[0], 2), dtype=jnp.uint32)
                updated_field = jax.vmap(apply_reset)(hist, new, reset_mask, dummy_keys)

        updated[name] = updated_field

    return updated
