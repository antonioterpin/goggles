"""Tests for the functional, JIT-compatible API of `goggles.filters`.

Each filter exposes:
    - ``init_state(data)``: allocate the initial state pytree.
    - ``apply(state, data) -> (new_state, output)``: pure functional step.

These tests verify that:
    1. ``apply`` reproduces the output of the stateful ``step`` over a
       sequence of inputs (eager equivalence).
    2. ``apply`` is JIT-compatible under ``jax.jit`` and produces the same
       outputs as the eager path.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from goggles.filters import (
    AverageFilter,
    ConcatFilter,
    ExpAverageFilter,
    Filter,
    FilterConfig,
    MedianFilter,
    MinMaxFilter,
    QuantizationFilter,
    RangeRejectFilter,
    ScaleFilter,
    StdRejectFilter,
    create_concat_filter,
)


def _filter_factories() -> list[tuple[str, Callable[[], Filter]]]:
    """Return (id, factory) pairs covering every public filter.

    Returns:
        A list of ``(id, factory)`` pairs. Each ``factory`` constructs a fresh
        filter instance.
    """
    return [
        ("ScaleFilter", lambda: ScaleFilter(scale=2.0)),
        ("MinMaxFilter", lambda: MinMaxFilter(min_val=-10.0, max_val=10.0)),
        ("AverageFilter", lambda: AverageFilter(window_size=3)),
        ("MedianFilter", lambda: MedianFilter(window_size=3)),
        ("ExpAverageFilter", lambda: ExpAverageFilter(alpha=0.4)),
        (
            "QuantizationFilter",
            lambda: QuantizationFilter(
                min_value=-1.0, max_value=1.0, step_size=0.1
            ),
        ),
        (
            "RangeRejectFilter",
            lambda: RangeRejectFilter(
                min_value=-1.0,
                max_value=1.0,
                fallback_filter=[
                    {"type": "ScaleFilter", "parameters": {"scale": 0.0}}
                ],
            ),
        ),
        (
            "StdRejectFilter",
            lambda: StdRejectFilter(
                std_factor=2.0,
                window_size=3,
                fallback_filter=[
                    {"type": "ScaleFilter", "parameters": {"scale": 0.0}}
                ],
            ),
        ),
        (
            "ConcatFilter",
            lambda: ConcatFilter(
                filters=[
                    ScaleFilter(scale=2.0),
                    AverageFilter(window_size=2),
                ]
            ),
        ),
        (
            "create_concat_filter",
            lambda: create_concat_filter(
                [
                    FilterConfig(
                        type="MinMaxFilter",
                        parameters={"min_val": -1.0, "max_val": 1.0},
                    ),
                    FilterConfig(
                        type="ExpAverageFilter", parameters={"alpha": 0.5}
                    ),
                ]
            ),
        ),
    ]


FILTER_IDS = [name for name, _ in _filter_factories()]
FILTER_FACTORIES = [factory for _, factory in _filter_factories()]


def _sample_sequence(
    rng: np.random.Generator, length: int, shape: tuple[int, ...]
) -> np.ndarray:
    """Draw a small float32 sequence for stateful tests.

    Args:
        rng: NumPy random generator.
        length: Number of samples in the sequence.
        shape: Per-sample array shape.

    Returns:
        Array of shape ``(length, *shape)`` with float32 values in
        ``[-2, 2]``.
    """
    return rng.uniform(-2.0, 2.0, size=(length, *shape)).astype(np.float32)


@pytest.mark.parametrize("factory", FILTER_FACTORIES, ids=FILTER_IDS)
def test_apply_matches_step_numpy(factory: Callable[[], Filter]) -> None:
    """`apply` over an explicit state must match `step` element-wise.

    Args:
        factory: Zero-arg constructor for a filter instance.
    """
    rng = np.random.default_rng(0)
    seq = _sample_sequence(rng, length=8, shape=(4,))

    eager = factory()
    eager_outs = [eager.step(x) for x in seq]

    func = factory()
    state = func.init_state(seq[0])
    func_outs: list[Any] = []
    for x in seq:
        state, out = func.apply(state, x)
        func_outs.append(np.asarray(out))

    for i, (e, f) in enumerate(zip(eager_outs, func_outs, strict=True)):
        np.testing.assert_allclose(
            np.asarray(e),
            np.asarray(f),
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"step/apply diverged at index {i}",
        )


@pytest.mark.parametrize("factory", FILTER_FACTORIES, ids=FILTER_IDS)
def test_apply_is_jittable(factory: Callable[[], Filter]) -> None:
    """`apply` must run under `jax.jit` and match the eager output.

    Args:
        factory: Zero-arg constructor for a filter instance.
    """
    rng = np.random.default_rng(1)
    seq = jnp.asarray(_sample_sequence(rng, length=6, shape=(3,)))

    f_eager = factory()
    state = f_eager.init_state(seq[0])
    eager_outs: list[Any] = []
    cur = state
    for x in seq:
        cur, out = f_eager.apply(cur, x)
        eager_outs.append(out)

    f_jit = factory()
    jitted = jax.jit(f_jit.apply)
    state_j = f_jit.init_state(seq[0])
    jit_outs: list[Any] = []
    for x in seq:
        state_j, out = jitted(state_j, x)
        jit_outs.append(out)

    for i, (a, b) in enumerate(zip(eager_outs, jit_outs, strict=True)):
        np.testing.assert_allclose(
            np.asarray(a),
            np.asarray(b),
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"jit/apply diverged at index {i}",
        )


@pytest.mark.parametrize("factory", FILTER_FACTORIES, ids=FILTER_IDS)
def test_apply_does_not_mutate_filter(
    factory: Callable[[], Filter],
) -> None:
    """`apply` must be pure: calling it must not advance internal state.

    Args:
        factory: Zero-arg constructor for a filter instance.
    """
    rng = np.random.default_rng(2)
    seq = _sample_sequence(rng, length=4, shape=(2,))

    f = factory()
    state = f.init_state(seq[0])

    # Drive `apply` twice from the same state; outputs must match because
    # `apply` is pure -- no hidden state on `f` should advance between calls.
    state_after_first, out_a = f.apply(state, seq[0])
    state_after_second, out_b = f.apply(state, seq[0])

    np.testing.assert_allclose(
        np.asarray(out_a),
        np.asarray(out_b),
        rtol=1e-6,
        err_msg="apply produced different outputs for identical (state, data)",
    )

    # Now mix with the eager path: starting fresh, `step` should produce the
    # same first output as `apply(init_state, x)`.
    f2 = factory()
    eager_first = f2.step(seq[0])
    np.testing.assert_allclose(
        np.asarray(eager_first),
        np.asarray(out_a),
        rtol=1e-6,
        err_msg="step's first output differs from apply(init_state, x)",
    )

    # Reset is harmless: ensure the state machine restarts cleanly.
    _ = state_after_first
    _ = state_after_second


def test_jit_apply_chained_state_matches_eager() -> None:
    """End-to-end check: a jit-compiled scan over apply matches eager."""
    f = AverageFilter(window_size=4)
    seq = jnp.asarray(
        np.random.default_rng(3)
        .uniform(-1.0, 1.0, size=(10, 5))
        .astype(np.float32)
    )

    init = f.init_state(seq[0])

    def body(state: Any, x: Any) -> tuple[Any, Any]:
        return f.apply(state, x)

    _, jit_outs = jax.lax.scan(body, init, seq)

    eager = AverageFilter(window_size=4)
    eager_outs = jnp.stack([eager.step(x) for x in seq])

    np.testing.assert_allclose(
        np.asarray(jit_outs),
        np.asarray(eager_outs),
        rtol=1e-5,
        atol=1e-6,
        err_msg="jax.lax.scan over apply diverged from eager step",
    )
