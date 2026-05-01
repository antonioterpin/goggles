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
    parsed = _parse(yaml_dump(payload))
    assert parsed == payload, (
        f"Plain mapping should roundtrip via yaml_dump; "
        f"expected {payload!r}, got {parsed!r}"
    )


def test_yaml_dump_normalizes_numpy_arrays_and_scalars() -> None:
    payload = {
        "array_1d": np.array([1, 2, 3], dtype=np.int32),
        "array_2d": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "int_scalar": np.int64(42),
        "float_scalar": np.float64(3.25),
        "bool_scalar": np.bool_(True),
        "zero_dim": np.array(7),
    }
    expected = {
        "array_1d": [1, 2, 3],
        "array_2d": [[1.0, 2.0], [3.0, 4.0]],
        "int_scalar": 42,
        "float_scalar": 3.25,
        "bool_scalar": True,
        "zero_dim": 7,
    }
    loaded = _parse(yaml_dump(payload))
    assert loaded == expected, (
        f"yaml_dump should normalize numpy arrays/scalars to native Python; "
        f"expected {expected!r}, got {loaded!r}"
    )


def test_yaml_dump_handles_nested_numpy_in_lists_and_dicts() -> None:
    payload = {
        "xs": [np.float32(1.5), np.int16(2), np.array([3, 4])],
        "ys": {"a": np.bool_(False), "b": [np.array([[1], [2]])]},
    }
    expected = {
        "xs": [1.5, 2, [3, 4]],
        "ys": {"a": False, "b": [[[1], [2]]]},
    }
    loaded = _parse(yaml_dump(payload))
    assert loaded == expected, (
        f"yaml_dump should normalize numpy values nested in lists/dicts; "
        f"expected {expected!r}, got {loaded!r}"
    )


def test_yaml_dump_accepts_top_level_numpy_values() -> None:
    arr_loaded = _parse(yaml_dump(np.array([1, 2, 3], dtype=np.int32)))
    assert arr_loaded == [1, 2, 3], (
        f"Top-level int32 ndarray should normalize to a list; "
        f"got {arr_loaded!r}"
    )
    f32_loaded = _parse(yaml_dump(np.float32(1.5)))
    assert f32_loaded == 1.5, (
        f"Top-level np.float32 should normalize to a Python float; "
        f"got {f32_loaded!r}"
    )
    i64_loaded = _parse(yaml_dump(np.int64(7)))
    assert i64_loaded == 7, (
        f"Top-level np.int64 should normalize to a Python int; "
        f"got {i64_loaded!r}"
    )


def test_yaml_dump_normalizes_numpy_scalar_keys() -> None:
    payload = {np.int64(1): "one", np.float32(2.0): "two"}
    parsed = _parse(yaml_dump(payload))
    expected = {1: "one", 2.0: "two"}
    assert parsed == expected, (
        f"yaml_dump should normalize numpy scalar keys to native Python; "
        f"expected {expected!r}, got {parsed!r}"
    )


def test_yaml_dump_normalizes_object_array_elements() -> None:
    payload = np.array([np.int64(1), np.float32(2.5), "s"], dtype=object)
    parsed = _parse(yaml_dump(payload))
    expected = [1, 2.5, "s"]
    assert parsed == expected, (
        f"yaml_dump should normalize numpy elements inside object arrays; "
        f"expected {expected!r}, got {parsed!r}"
    )
