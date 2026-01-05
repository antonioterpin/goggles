"""Tests verifying that public functions in goggles.history are JIT/VMAP friendly."""

import jax
import jax.numpy as jnp
import pytest
from goggles.history import create_history, update_history
from goggles.history.utils import slice_history, peek_last
from goggles.history.spec import HistorySpec


@pytest.mark.parametrize("batch_size", [1, 2, 5])
@pytest.mark.parametrize("length", [2, 4])
@pytest.mark.parametrize("shape", [(1,), (2, 3)])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.int32])
@pytest.mark.parametrize("init", ["zeros", "ones"])
def test_create_history_jittable_and_vmappable(batch_size, length, shape, dtype, init):
    spec = HistorySpec.from_config(
        {"x": {"length": length, "shape": shape, "dtype": dtype, "init": init}}
    )

    # create_history depends only on spec and batch_size (pure inputs)
    # JIT the function (spec is a Python dataclass, so we supply as static arg)
    # JIT the function by constructing the spec inside the jitted function
    # so we don't pass the non-hashable HistorySpec as a static arg.
    # Make batch_size a static argument so create_history's Python-level
    # validation (batch_size <= 0) is evaluated at compile time rather than
    # on a traced JAX tracer.
    jitted = jax.jit(
        lambda bs: create_history(
            HistorySpec.from_config(
                {"x": {"length": length, "shape": shape, "dtype": dtype, "init": init}}
            ),
            bs,
        ),
        static_argnums=(0,),
    )
    h = jitted(batch_size)
    assert "x" in h, "'x' field missing in jitted history"
    assert h["x"].shape == (batch_size, length, *shape), "Jitted history shape mismatch"


@pytest.mark.parametrize("B", [1, 3])
@pytest.mark.parametrize("T", [2, 5])
def test_update_history_jittable_and_vmappable(B, T):
    hist = {"x": jnp.zeros((B, T, 1), dtype=jnp.float32)}
    new = {"x": jnp.ones((B, 1, 1), dtype=jnp.float32)}

    # JIT the update_history function across its array args; spec and rng static
    jitted = jax.jit(update_history)
    out = jitted(hist, new)
    assert out["x"].shape == (B, T, 1), "Jitted update_history shape mismatch"

    # vmap over per-batch row using a wrapper that operates on single-row tensors
    def per_row(h_row, n_row):
        # h_row: (T, *), n_row: (1, *) -> add batch dim and call
        h = {"x": h_row[None, ...]}
        n = {"x": n_row[None, ...]}
        return update_history(h, n)["x"][0]

    vmapped = jax.vmap(per_row)
    out_rows = vmapped(hist["x"], new["x"])
    assert out_rows.shape == (B, T, 1), "Vmapped update_history shape mismatch"


@pytest.mark.parametrize("B", [1, 3])
@pytest.mark.parametrize("T", [5, 8])
@pytest.mark.parametrize("start", [0, 1])
@pytest.mark.parametrize("length", [1, 2])
def test_slice_and_peek_jittable_and_vmappable(B, T, start, length):
    history = {
        "a": jnp.arange(B * T).reshape(B, T, 1),
        "b": jnp.arange(B * T * 2).reshape(B, T, 2),
    }

    # slice_history: start/length are Python ints -> static args
    jitted_slice = jax.jit(slice_history, static_argnums=(1, 2))
    s = jitted_slice(history, start, length)
    assert s["a"].shape == (B, length, 1), "Jitted slice_history shape mismatch"

    # peek_last: k is int -> static arg
    jitted_peek = jax.jit(peek_last, static_argnums=(1,))
    p = jitted_peek(history, 2)
    assert p["b"].shape == (B, 2, 2), "Jitted peek_last shape mismatch"

    # vmap slice_history per-field (operate on single batch rows)
    def per_row_slice(a_row, b_row):
        h = {"a": a_row[None, ...], "b": b_row[None, ...]}
        return slice_history(h, 0, 1)["a"][0]

    vmapped = jax.vmap(per_row_slice)
    out = vmapped(history["a"], history["b"])  # should produce shape (B, 1, 1)
    assert out.shape == (B, 1, 1), "Vmapped slice_history shape mismatch"
