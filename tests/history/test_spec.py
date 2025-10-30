"""Unit tests for goggles.history.spec module."""

import pytest
import jax.numpy as jnp
from goggles.history.spec import HistorySpec, HistoryFieldSpec


@pytest.mark.parametrize(
    "images_len,images_shape,flow_len,flow_shape,flow_init",
    [
        (4, (64, 64, 3), 2, (64, 64, 2), "ones"),
        (3, (32, 32, 1), 1, (32, 32, 2), "ones"),
    ],
)
def test_from_config_basic_dict(
    images_len, images_shape, flow_len, flow_shape, flow_init
):
    config = {
        "images": {"length": images_len, "shape": images_shape},
        "flow": {
            "length": flow_len,
            "shape": flow_shape,
            "dtype": jnp.float32,
            "init": flow_init,
        },
    }
    spec = HistorySpec.from_config(config)

    assert isinstance(spec, HistorySpec)
    assert set(spec.fields.keys()) == {"images", "flow"}

    img = spec.fields["images"]
    assert isinstance(img, HistoryFieldSpec)
    assert img.length == images_len
    assert img.shape == images_shape
    assert img.dtype == jnp.float32
    assert img.init == "zeros"

    flow = spec.fields["flow"]
    assert flow.init == flow_init
    assert flow.dtype == jnp.float32


def test_from_config_with_existing_fieldspec():
    field_spec = HistoryFieldSpec(
        length=5, shape=(8, 8), dtype=jnp.float32, init="randn"
    )
    spec = HistorySpec.from_config({"feat": field_spec})

    assert spec.fields["feat"].length == 5
    assert spec.fields["feat"].init == "randn"


@pytest.mark.parametrize("bad_config", [None, 123, [1, 2]])
def test_invalid_config_type(bad_config):
    with pytest.raises(TypeError):
        HistorySpec.from_config(bad_config)


@pytest.mark.parametrize(
    "config",
    [
        {"a": {"shape": (2, 3)}},  # missing length
        {"a": {"length": 2}},  # missing shape
    ],
)
def test_missing_required_keys(config):
    with pytest.raises(ValueError, match="must define 'length' and 'shape'"):
        HistorySpec.from_config(config)


@pytest.mark.parametrize(
    "config",
    [
        {"a": {"length": 0, "shape": (2,)}},
        {"a": {"length": -1, "shape": (2,)}},
    ],
)
def test_invalid_length(config):
    with pytest.raises(ValueError, match="length must be an int >= 1"):
        HistorySpec.from_config(config)


def test_invalid_shape_type_and_values():
    bad_shape_config = {"a": {"length": 2, "shape": "not_a_tuple"}}
    with pytest.raises(TypeError, match="shape must be a tuple/list"):
        HistorySpec.from_config(bad_shape_config)

    neg_dim_config = {"a": {"length": 2, "shape": (4, -1)}}
    with pytest.raises(ValueError, match="non-negative"):
        HistorySpec.from_config(neg_dim_config)


def test_invalid_dtype():
    config = {"a": {"length": 2, "shape": (3,), "dtype": "not_a_dtype"}}
    with pytest.raises(TypeError, match="dtype is not a valid JAX dtype"):
        HistorySpec.from_config(config)


@pytest.mark.parametrize("bad_init", ["invalid", "", 123])
def test_invalid_init(bad_init):
    config = {"a": {"length": 2, "shape": (3,), "init": bad_init}}
    with pytest.raises(ValueError, match="init must be one of"):
        HistorySpec.from_config(config)


def test_invalid_field_type_in_config():
    config = {"a": 123}
    with pytest.raises(TypeError, match="must be a Mapping or HistoryFieldSpec"):
        HistorySpec.from_config(config)


def test_from_config_with_invalid_fieldspec_instances():
    # length invalid
    bad_len = HistoryFieldSpec(length=0, shape=(2,), dtype=jnp.float32, init="zeros")
    with pytest.raises(ValueError, match="length must be an int >= 1"):
        HistorySpec.from_config({"a": bad_len})

    # negative dim in shape
    bad_shape = HistoryFieldSpec(
        length=2, shape=(4, -1), dtype=jnp.float32, init="zeros"
    )
    with pytest.raises(ValueError, match="shape must be a tuple of non-negative ints"):
        HistorySpec.from_config({"a": bad_shape})

    # invalid init
    bad_init = HistoryFieldSpec(length=2, shape=(2,), dtype=jnp.float32, init="bad")
    with pytest.raises(ValueError, match="init must be one of"):
        HistorySpec.from_config({"a": bad_init})
