""""Tests for goggles.history module initialization and public API."""
import builtins
import importlib
import sys
import inspect

import goggles.history as gh
import pytest


def test_import_error_message_for_missing_jax(monkeypatch):
    orig_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "jax" or name.startswith("jax."):
            raise ImportError("no jax here")
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    # Remove cached modules so import will re-execute module code.
    cached = {
        k: sys.modules.get(k) for k in list(sys.modules) if k.startswith("goggles")
    }
    for k in list(cached.keys()):
        sys.modules.pop(k, None)

    try:
        with pytest.raises(ImportError) as exc:
            importlib.import_module("goggles.history")
        assert "The 'goggles.history' module requires JAX" in str(exc.value)
    finally:
        monkeypatch.setattr(builtins, "__import__", orig_import)
        for k, v in cached.items():
            if v is not None:
                sys.modules[k] = v
        importlib.reload(importlib.import_module("goggles.history"))


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


def test_type_aliases_exist():
    """Check shared type aliases are available in types.py."""
    import goggles.history.types as ht

    for name in ["PRNGKey", "Array", "HistoryDict"]:
        assert hasattr(ht, name)
        obj = getattr(ht, name)
        assert obj is not None
