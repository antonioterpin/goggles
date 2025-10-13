"""Unit tests for goggles.history.buffer module."""

import pytest
import jax
import jax.numpy as jnp
from goggles.history.buffer import create_history, update_history
from goggles.history.spec import HistorySpec, HistoryFieldSpec


@pytest.fixture
def sample_spec():
    return HistorySpec.from_config(
        {
            "images": {
                "length": 3,
                "shape": (4, 4, 1),
                "dtype": jnp.float32,
                "init": "zeros",
            },
            "flow": {
                "length": 2,
                "shape": (2, 2, 2),
                "dtype": jnp.float32,
                "init": "ones",
            },
        }
    )


@pytest.fixture
def base_spec():
    return HistorySpec.from_config(
        {
            "zeros_field": {
                "length": 3,
                "shape": (2,),
                "dtype": jnp.float32,
                "init": "zeros",
            },
            "ones_field": {
                "length": 3,
                "shape": (2,),
                "dtype": jnp.float32,
                "init": "ones",
            },
            "randn_field": {
                "length": 3,
                "shape": (2,),
                "dtype": jnp.float32,
                "init": "randn",
            },
            "none_field": {
                "length": 3,
                "shape": (2,),
                "dtype": jnp.float32,
                "init": "none",
            },
        }
    )


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_create_history_basic(sample_spec, batch_size):
    history = create_history(sample_spec, batch_size=batch_size)
    assert set(history.keys()) == {"images", "flow"}
    # shapes should follow (B, length, *shape)
    assert history["images"].shape[0] == batch_size
    assert history["images"].shape[1] == 3
    assert history["flow"].shape[0] == batch_size
    assert history["flow"].shape[1] == 2
    assert jnp.all(history["images"] == 0)
    assert jnp.all(history["flow"] == 1)


@pytest.mark.parametrize("batch_size,length,vec_shape", [(1, 4, (2,)), (2, 4, (2,))])
def test_create_history_with_randn(batch_size, length, vec_shape):
    spec = HistorySpec.from_config(
        {
            "noise": {
                "length": length,
                "shape": vec_shape,
                "dtype": jnp.float32,
                "init": "randn",
            }
        }
    )
    rng = jax.random.PRNGKey(0)
    history = create_history(spec, batch_size=batch_size, rng=rng)
    assert "noise" in history
    assert history["noise"].shape[0] == batch_size
    assert history["noise"].shape[1] == length
    assert history["noise"].shape[2:] == vec_shape
    assert jnp.isfinite(history["noise"]).all()


def test_create_history_invalid_batch_raises(sample_spec):
    with pytest.raises(ValueError, match="batch_size must be > 0"):
        create_history(sample_spec, batch_size=0)


@pytest.mark.parametrize("batch_size,time_len", [(1, 3), (2, 3), (4, 5)])
def test_update_history_basic_shift_and_append(batch_size, time_len):
    hist = {"x": jnp.zeros((batch_size, time_len, 1))}
    new = {"x": jnp.ones((batch_size, 1, 1))}
    updated = update_history(hist, new)
    expected = jnp.concatenate(
        [jnp.zeros((batch_size, time_len - 1, 1)), jnp.ones((batch_size, 1, 1))], axis=1
    )
    assert jnp.allclose(updated["x"], expected)


def test_update_history_invalid_shape_raises():
    hist = {"x": jnp.zeros((2, 3, 1))}
    new = {"x": jnp.ones((2, 2, 1))}
    with pytest.raises(ValueError, match="Append length must be 1"):
        update_history(hist, new)


def test_update_history_dtype_mismatch_raises():
    hist = {"x": jnp.zeros((2, 3, 1), dtype=jnp.float32)}
    new = {"x": jnp.ones((2, 1, 1), dtype=jnp.int32)}
    with pytest.raises(ValueError, match="Dtype mismatch"):
        update_history(hist, new)


@pytest.mark.parametrize("batch_size,time_len", [(2, 3), (3, 4)])
def test_update_history_reset_mask(batch_size, time_len):
    hist = {
        "x": jnp.arange(batch_size * time_len)
        .reshape(batch_size, time_len, 1)
        .astype(jnp.float32)
    }
    new = {"x": jnp.ones((batch_size, 1, 1), dtype=jnp.float32) * 99}
    # alternate reset mask: True for first, False for others
    reset_mask = jnp.array([True] + [False] * (batch_size - 1))
    updated = update_history(hist, new, reset_mask=reset_mask, spec=None)

    # For the first batch element (reset), should be all zeros
    assert jnp.all(updated["x"][0] == 0)
    # For a non-reset batch element, last entry should equal the appended value
    assert updated["x"][1, -1, 0] == 99


def test_update_history_missing_field_raises():
    hist = {"a": jnp.zeros((2, 3, 1))}
    new = {"b": jnp.ones((2, 1, 1))}
    with pytest.raises(ValueError, match="Missing new data"):
        update_history(hist, new)


@pytest.mark.parametrize("batch_size,time_len", [(2, 3), (4, 3)])
def test_update_history_vmappable_jittable(batch_size, time_len):
    hist = {"x": jnp.zeros((batch_size, time_len, 1))}
    new = {"x": jnp.ones((batch_size, 1, 1))}

    # JIT across full batch
    _ = jax.jit(update_history)(hist, new)

    # vmap across batch dimension manually
    def per_batch(h, n):
        out = update_history({"x": h[None, ...]}, {"x": n[None, ...]})
        return out["x"][0]

    _ = jax.vmap(per_batch)(hist["x"], new["x"])


@pytest.mark.parametrize("batch_size,time_len", [(2, 3), (3, 4)])
def test_update_history_reset_with_zeros(base_spec, batch_size, time_len):
    hist = {
        "zeros_field": jnp.arange(batch_size * time_len * 2)
        .reshape(batch_size, time_len, 2)
        .astype(jnp.float32)
    }
    new = {"zeros_field": jnp.ones((batch_size, 1, 2), dtype=jnp.float32) * 99}
    reset_mask = jnp.array([True] + [False] * (batch_size - 1))

    updated = update_history(hist, new, reset_mask, base_spec)

    # Reset batch fully zeros
    assert jnp.all(updated["zeros_field"][0] == 0)
    # Non-reset batch shifted and appended
    assert jnp.all(updated["zeros_field"][1, -1] == 99)


@pytest.mark.parametrize("batch_size,time_len", [(2, 3), (3, 4)])
def test_update_history_reset_with_ones(base_spec, batch_size, time_len):
    hist = {"ones_field": jnp.zeros((batch_size, time_len, 2), dtype=jnp.float32)}
    new = {"ones_field": jnp.ones((batch_size, 1, 2), dtype=jnp.float32) * 5}
    reset_mask = jnp.array([True] + [False] * (batch_size - 1))

    updated = update_history(hist, new, reset_mask, base_spec)

    # Reset batch filled with ones
    assert jnp.allclose(
        updated["ones_field"][0], jnp.ones_like(updated["ones_field"][0])
    )
    # Non-reset batch last frame == 5
    assert jnp.allclose(updated["ones_field"][1, -1], 5.0)


@pytest.mark.parametrize("batch_size,time_len", [(2, 3), (3, 4)])
def test_update_history_reset_with_randn_requires_rng(base_spec, batch_size, time_len):
    hist = {"randn_field": jnp.zeros((batch_size, time_len, 2), dtype=jnp.float32)}
    new = {"randn_field": jnp.ones((batch_size, 1, 2), dtype=jnp.float32)}
    reset_mask = jnp.array([True] + [False] * (batch_size - 1))

    # Should raise if rng missing
    with pytest.raises(ValueError, match="requires rng"):
        update_history(hist, new, reset_mask, base_spec)

    rng = jax.random.PRNGKey(0)
    updated = update_history(hist, new, reset_mask, base_spec, rng=rng)
    # Ensure reset batch contains non-zero random values
    assert not jnp.allclose(updated["randn_field"][0], 0)
    # Ensure non-reset batch is correctly appended
    assert jnp.allclose(updated["randn_field"][1, -1], 1.0)


@pytest.mark.parametrize("batch_size,time_len", [(2, 3), (3, 4)])
def test_update_history_reset_with_none_mode(base_spec, batch_size, time_len):
    hist = {
        "none_field": jnp.arange(batch_size * time_len * 2)
        .reshape(batch_size, time_len, 2)
        .astype(jnp.float32)
    }
    new = {"none_field": jnp.ones((batch_size, 1, 2), dtype=jnp.float32) * 10}
    reset_mask = jnp.array([True] + [False] * (batch_size - 1))

    updated = update_history(hist, new, reset_mask, base_spec)

    # "none" mode should keep original values (no reset): compare shifted rows
    assert jnp.allclose(
        updated["none_field"][0],
        jnp.concatenate([hist["none_field"][0, 1:], new["none_field"][0]], axis=0),
    )
    assert jnp.allclose(updated["none_field"][1, -1], 10.0)


def test_update_history_reset_invalid_mode_raises():
    # Creating a spec with invalid init should raise
    with pytest.raises(ValueError, match="must be one of"):
        HistorySpec.from_config(
            {
                "bad_field": {
                    "length": 3,
                    "shape": (2,),
                    "dtype": jnp.float32,
                    "init": "foo",
                }
            }
        )


def test_update_history_accepts_per_batch_keys(base_spec):
    # Ensure update_history accepts an array of per-batch PRNGKeys (B, 2)
    B, T = 2, 3
    hist = {"randn_field": jnp.zeros((B, T, 2), dtype=jnp.float32)}
    new = {"randn_field": jnp.ones((B, 1, 2), dtype=jnp.float32)}
    reset_mask = jnp.array([True, False])

    # Create per-batch keys and pass them directly
    base = jax.random.PRNGKey(123)
    per_batch = jax.random.split(base, B)

    updated = update_history(hist, new, reset_mask, base_spec, rng=per_batch)
    assert updated["randn_field"].shape == (B, T, 2)


def test_create_history_invalid_field_length():
    spec = HistorySpec(
        fields={"a": HistoryFieldSpec(length=0, shape=(1,), dtype=jnp.float32)}
    )
    with pytest.raises(ValueError, match="Invalid history length"):
        create_history(spec, batch_size=1)


def test_create_history_unknown_init_fieldspec():
    bad = HistoryFieldSpec(length=2, shape=(1,), dtype=jnp.float32, init="badinit")
    spec = HistorySpec(fields={"a": bad})
    with pytest.raises(ValueError, match="Unknown init mode"):
        create_history(spec, batch_size=1)


@pytest.mark.parametrize("batch_size,time_len", [(2, 3)])
def test_update_history_dim_mismatch_raises(batch_size, time_len):
    # new has different ndim than hist -> should raise dim mismatch
    hist = {"x": jnp.zeros((batch_size, time_len, 1))}
    # make new with one fewer dimension
    new = {"x": jnp.ones((batch_size, 1))}
    with pytest.raises(ValueError, match="Dim mismatch"):
        update_history(hist, new)


@pytest.mark.parametrize("batch_size,length,vec_shape", [(1, 2, (2,)), (2, 3, (2,))])
def test_create_history_none_init(batch_size, length, vec_shape):
    spec = HistorySpec.from_config(
        {
            "x": {
                "length": length,
                "shape": vec_shape,
                "dtype": jnp.float32,
                "init": "none",
            }
        }
    )
    history = create_history(spec, batch_size=batch_size)
    assert "x" in history
    arr = history["x"]
    assert arr.shape == (batch_size, length, *vec_shape)
    assert arr.dtype == jnp.float32


@pytest.mark.parametrize("batch_size,length,vec_shape", [(1, 2, (2,)), (2, 3, (2,))])
def test_create_history_randn_requires_rng_raises(batch_size, length, vec_shape):
    spec = HistorySpec.from_config(
        {
            "n": {
                "length": length,
                "shape": vec_shape,
                "dtype": jnp.float32,
                "init": "randn",
            }
        }
    )
    # Should raise when rng missing
    with pytest.raises(ValueError, match="requires rng"):
        create_history(spec, batch_size=batch_size)


@pytest.mark.parametrize("batch_size,time_len", [(2, 3), (3, 4)])
def test_update_history_invalid_reset_mask_shape(batch_size, time_len):
    hist = {"x": jnp.zeros((batch_size, time_len, 1))}
    new = {"x": jnp.ones((batch_size, 1, 1))}
    # Wrong shape reset_mask
    bad_mask = jnp.array([[True]])
    with pytest.raises(ValueError, match="Invalid reset_mask shape"):
        update_history(hist, new, reset_mask=bad_mask)


@pytest.mark.parametrize("batch_size,time_len", [(2, 3)])
def test_update_history_rng_wrong_shape_raises(batch_size, time_len):
    hist = {"x": jnp.zeros((batch_size, time_len, 1))}
    new = {"x": jnp.ones((batch_size, 1, 1))}
    reset_mask = jnp.array([True] + [False] * (batch_size - 1))
    # rng with wrong first dim
    bad_rng = jnp.zeros((batch_size + 1, 2), dtype=jnp.uint32)
    spec = HistorySpec.from_config(
        {
            "x": {
                "length": time_len,
                "shape": (1,),
                "dtype": jnp.float32,
                "init": "randn",
            }
        }
    )
    with pytest.raises(ValueError, match="rng must be a PRNGKey or an array"):
        update_history(hist, new, reset_mask, spec=spec, rng=bad_rng)


@pytest.mark.parametrize("batch_size,time_len", [(2, 3)])
def test_update_history_randn_no_resets_uses_dummy_keys(batch_size, time_len):
    # If init is randn but reset_mask has no True values and rng is None,
    # update_history should proceed using dummy keys (no error).
    spec = HistorySpec.from_config(
        {
            "r": {
                "length": time_len,
                "shape": (1,),
                "dtype": jnp.float32,
                "init": "randn",
            }
        }
    )
    hist = {"r": jnp.zeros((batch_size, time_len, 1), dtype=jnp.float32)}
    new = {"r": jnp.ones((batch_size, 1, 1), dtype=jnp.float32)}
    # Use a numpy boolean array (or Python list) so outer host-level checks can
    # evaluate without JAX tracer behavior. No rng provided, but no resets -> ok.
    import numpy as _np

    reset_mask = _np.array([False, False])
    out = update_history(hist, new, reset_mask, spec=spec, rng=None)
    # No resets => should behave like shift+append for each batch
    assert out["r"].shape == (batch_size, time_len, 1)


@pytest.mark.parametrize("batch_size,time_len", [(2, 3)])
def test_update_history_unknown_init_mode_in_do_reset(batch_size, time_len):
    # Build a spec where init_mode is unknown; expect update_history to raise
    bad_spec = HistorySpec(
        fields={
            "b": HistoryFieldSpec(length=3, shape=(1,), dtype=jnp.float32, init="foo")
        }
    )
    hist = {"b": jnp.zeros((batch_size, time_len, 1), dtype=jnp.float32)}
    new = {"b": jnp.ones((batch_size, 1, 1), dtype=jnp.float32)}
    reset_mask = jnp.array([True, False])
    with pytest.raises(ValueError, match="Unknown init mode"):
        update_history(hist, new, reset_mask, spec=bad_spec)
