"""Tests for history utils: peek_last and slice_history."""

import jax.numpy as jnp
import numpy as np
import pytest

from goggles.history.utils import peek_last, slice_history


def _make_history(B, T, shapes):
    """Build a toy HistoryDict with deterministic contents."""
    history = {}
    for i, (name, shape) in enumerate(shapes):
        size = B * T * int(np.prod(shape) if shape else 1)
        data = jnp.arange(i * 10, i * 10 + size, dtype=jnp.int32).reshape(
            (B, T, *shape)
        )
        history[name] = data
    return history


@pytest.mark.parametrize("B,T", [(1, 3), (2, 4), (3, 6)])
@pytest.mark.parametrize(
    "shapes",
    [
        (("a", ()), ("b", (2,))),  # scalar per-timestep + vector
        (("img", (4, 4, 3)), ("flow", (4, 4, 2))),  # image-like tensors
    ],
)
@pytest.mark.parametrize(
    "start,length",
    [
        (0, 1),  # first single step
        (0, 2),  # prefix
        (1, 1),  # middle single step
        (1, 2),  # middle window
    ],
)
def test_slice_history_dict_and_field_shapes(B, T, shapes, start, length):
    assume_ok = start + length <= T
    if not assume_ok:
        pytest.skip("Invalid window for this T; exercised in error tests below.")

    history = _make_history(B, T, shapes)

    # Dict mode: shapes match (B, length, *payload)
    sliced = slice_history(history, start=start, length=length)
    for (name, shape), arr in zip(shapes, map(lambda k: sliced[k[0]], shapes)):
        assert arr.shape == (B, length, *shape), f"{name} shape mismatch"
        # Values match direct slicing
        np.testing.assert_array_equal(
            np.asarray(arr), np.asarray(history[name][:, start : start + length, ...])
        )

    # Single-field mode: same checks for one key
    field_name, field_shape = shapes[0]
    sliced_one = slice_history(history, start=start, length=length, field=field_name)
    assert sliced_one.shape == (B, length, *field_shape)
    np.testing.assert_array_equal(
        np.asarray(sliced_one),
        np.asarray(history[field_name][:, start : start + length, ...]),
    )


@pytest.mark.parametrize("B,T", [(1, 3), (2, 5)])
@pytest.mark.parametrize(
    "shapes",
    [
        (("a", (3,)), ("b", (2, 2))),
    ],
)
@pytest.mark.parametrize(
    "bad_args",
    [
        {"start": 0, "length": 0},  # length <= 0
        {"start": -1, "length": 1},  # start < 0
        {"start": 10, "length": 1},  # start >= T
        {"start": 1, "length": 100},  # start+length > T
    ],
)
def test_slice_history_errors(B, T, shapes, bad_args):
    history = _make_history(B, T, shapes)

    # Validate error conditions
    with pytest.raises((ValueError, TypeError)):
        slice_history(history, **bad_args)

    # Unknown field
    if bad_args == {"start": 0, "length": 1}:
        with pytest.raises(KeyError):
            slice_history(history, field="__nope__", **bad_args)


@pytest.mark.parametrize("B,T", [(1, 3), (2, 4), (3, 5)])
@pytest.mark.parametrize(
    "shapes",
    [
        (("a", ()), ("b", (2,))),
        (("img", (8, 8, 1)), ("flow", (8, 8, 2))),
    ],
)
@pytest.mark.parametrize("k", [1, 2])
def test_peek_last_valid(B, T, shapes, k):
    assume_ok = 1 <= k <= T
    if not assume_ok:
        pytest.skip("k outside valid range; covered in error tests.")

    history = _make_history(B, T, shapes)
    last = peek_last(history, k=k)
    for (name, shape), arr in zip(shapes, map(lambda k: last[k[0]], shapes)):
        assert arr.shape == (B, k, *shape), f"{name} shape mismatch"
        np.testing.assert_array_equal(
            np.asarray(arr), np.asarray(history[name][:, -k:, ...])
        )


@pytest.mark.parametrize("B,T", [(1, 3), (2, 4)])
@pytest.mark.parametrize(
    "shapes",
    [
        (("a", (1,)), ("b", (2, 2))),
    ],
)
@pytest.mark.parametrize("k", [0, 5])  # k < 1 and k > T (for some T above)
def test_peek_last_errors(B, T, shapes, k):
    history = _make_history(B, T, shapes)
    with pytest.raises(ValueError):
        peek_last(history, k=k)


def test_slice_history_empty_raises():
    with pytest.raises(TypeError):
        slice_history({}, start=0, length=1)


def test_peek_last_empty_raises():
    with pytest.raises(TypeError):
        peek_last({})


def test_slice_history_rank_and_field_errors_additional():
    # rank <2 should raise
    with pytest.raises(TypeError):
        slice_history({"a": jnp.zeros((3,))}, start=0, length=1)

    # unknown field
    history = {"x": jnp.zeros((1, 3, 1), dtype=jnp.int32)}
    with pytest.raises(KeyError):
        slice_history(history, start=0, length=1, field="nope")

    # field with rank <2
    with pytest.raises(TypeError):
        slice_history(
            {"a": jnp.zeros((1, 3, 1)), "b": jnp.zeros((3,))},
            start=0,
            length=1,
            field="b",
        )


def test_peek_last_rank_error_additional():
    with pytest.raises(TypeError):
        peek_last({"a": jnp.zeros((3,))}, k=1)
