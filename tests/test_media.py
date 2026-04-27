"""Unit tests for goggles.media helpers."""

from __future__ import annotations

import numpy as np
from ruamel.yaml import YAML

from goggles.media import yaml_dump


def _parse(text: str) -> object:
    yaml = YAML(typ="safe", pure=True)
    return yaml.load(text)


def test_yaml_dump_roundtrips_plain_mapping() -> None:
    payload = {"name": "cfg", "values": [1, 2, 3], "nested": {"a": 1}}
    assert _parse(yaml_dump(payload)) == payload


def test_yaml_dump_normalizes_numpy_arrays_and_scalars() -> None:
    payload = {
        "array_1d": np.array([1, 2, 3], dtype=np.int32),
        "array_2d": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "int_scalar": np.int64(42),
        "float_scalar": np.float64(3.25),
        "bool_scalar": np.bool_(True),
        "zero_dim": np.array(7),
    }
    loaded = _parse(yaml_dump(payload))
    assert loaded == {
        "array_1d": [1, 2, 3],
        "array_2d": [[1.0, 2.0], [3.0, 4.0]],
        "int_scalar": 42,
        "float_scalar": 3.25,
        "bool_scalar": True,
        "zero_dim": 7,
    }


def test_yaml_dump_handles_nested_numpy_in_lists_and_dicts() -> None:
    payload = {
        "xs": [np.float32(1.5), np.int16(2), np.array([3, 4])],
        "ys": {"a": np.bool_(False), "b": [np.array([[1], [2]])]},
    }
    loaded = _parse(yaml_dump(payload))
    assert loaded == {
        "xs": [1.5, 2, [3, 4]],
        "ys": {"a": False, "b": [[[1], [2]]]},
    }


def test_yaml_dump_accepts_top_level_numpy_values() -> None:
    assert _parse(yaml_dump(np.array([1, 2, 3], dtype=np.int32))) == [1, 2, 3]
    assert _parse(yaml_dump(np.float32(1.5))) == 1.5
    assert _parse(yaml_dump(np.int64(7))) == 7


def test_yaml_dump_normalizes_numpy_scalar_keys() -> None:
    payload = {np.int64(1): "one", np.float32(2.0): "two"}
    assert _parse(yaml_dump(payload)) == {1: "one", 2.0: "two"}


def test_yaml_dump_normalizes_object_array_elements() -> None:
    payload = np.array([np.int64(1), np.float32(2.5), "s"], dtype=object)
    assert _parse(yaml_dump(payload)) == [1, 2.5, "s"]
