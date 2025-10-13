# tests/test_history_api.py
"""API surface tests for goggles.history scaffolding.

These tests verify that the public symbols, types, and module structure
exist and can be imported before functional implementation.
"""

import inspect
import pytest

import goggles.history as gh


def test_public_api_symbols_exist():
    """Ensure all expected names are in the public API."""
    expected = {
        "HistoryFieldSpec",
        "HistorySpec",
        "create_history",
        "update_history",
        "slice_history",
        "peek_last",
    }
    for name in expected:
        assert hasattr(gh, name), f"Missing public symbol: {name}"

def test_history_field_spec_signature():
    """Check HistoryFieldSpec constructor signature."""
    from goggles.history import HistoryFieldSpec

    sig = inspect.signature(HistoryFieldSpec)
    params = list(sig.parameters.keys())
    for p in ["length", "shape"]:
        assert p in params
    assert "dtype" in params and "init" in params


def test_history_spec_from_config_signature():
    from goggles.history import HistorySpec

    assert hasattr(HistorySpec, "from_config")

    raw_attr = HistorySpec.__dict__.get("from_config")
    assert isinstance(raw_attr, classmethod), "from_config should be a @classmethod"

    sig = inspect.signature(raw_attr.__func__)
    params = list(sig.parameters.keys())
    assert params[0] == "cls", "First argument of from_config should be 'cls'"


def test_create_update_raise_not_implemented():
    """Stubbed API functions should raise NotImplementedError for now."""
    from goggles.history import HistorySpec, create_history, update_history

    spec = HistorySpec(fields={})
    with pytest.raises(NotImplementedError):
        create_history(spec, batch_size=1)
    with pytest.raises(NotImplementedError):
        update_history({}, {})


def test_utils_raise_not_implemented():
    """Utility stubs should also raise NotImplementedError."""
    from goggles.history import slice_history, peek_last

    with pytest.raises(NotImplementedError):
        slice_history({}, 0, 1)
    with pytest.raises(NotImplementedError):
        peek_last({}, 1)


def test_type_aliases_exist():
    """Check shared type aliases are available in types.py."""
    import goggles.history.types as ht

    for name in ["PRNGKey", "Array", "HistoryDict"]:
        assert hasattr(ht, name)
        obj = getattr(ht, name)
        assert obj is not None
