"""Smoke tests for Goggles API reachability and backward-compatibility.

These tests ensure:
- The public API symbols are importable from `goggles`.
- The new structured API (`run`, `get_logger`, etc.) can be called without error.
- Legacy aliases (`info`, `debug`, `scalar`, etc.) are still callable.
- No import-time side effects (other than the expected single deprecation warning).

We do NOT test real logging behavior, file creation, or W&B integration here.
"""

import importlib
import sys
import warnings
import inspect
import logging
import pytest


def test_public_exports_match_contract():
    import goggles

    expected = {
        "RunContext",
        "BoundLogger",
        "configure",
        "run",
        "current_run",
        "get_logger",
        "scalar",
        "image",
        "video",
    }
    assert set(goggles.__all__) == expected
    for name in expected:
        assert hasattr(goggles, name)


from dataclasses import is_dataclass, fields
from goggles import RunContext


def test_runcontext_structure():
    assert is_dataclass(RunContext)
    assert getattr(RunContext, "__dataclass_params__").frozen
    field_names = [f.name for f in fields(RunContext)]
    expected = [
        "run_id",
        "run_name",
        "log_dir",
        "created_at",
        "pid",
        "host",
        "python",
        "metadata",
        "wandb",
    ]
    assert field_names == expected


def test_import_root_no_side_effects(monkeypatch):
    """Importing goggles should not attach handlers to root."""
    root_before = list(logging.getLogger().handlers)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        import goggles  # noqa: F401
    root_after = list(logging.getLogger().handlers)

    # Should have exactly one DeprecationWarning
    dep_warnings = [rec for rec in w if issubclass(rec.category, DeprecationWarning)]
    assert len(dep_warnings) == 0, "Expected 0 deprecation warning at import."

    # Root handlers unchanged
    assert root_before == root_after, "Importing goggles modified root handlers."


def test_public_exports_match_api_contract():
    """Ensure top-level API matches declared __all__ surface."""
    import goggles

    expected = {
        "RunContext",
        "BoundLogger",
        "configure",
        "run",
        "get_logger",
        "scalar",
        "image",
        "video",
        # legacy functions should still exist
        "info",
        "debug",
        "warning",
        "error",
    }
    missing = expected - set(dir(goggles))
    assert not missing, f"Missing expected symbols: {missing}"


@pytest.mark.parametrize(
    "name",
    [
        "RunContext",
        "BoundLogger",
        "configure",
        "run",
        "get_logger",
        "scalar",
        "image",
        "video",
        "info",
        "debug",
        "warning",
        "error",
    ],
)
def test_symbols_are_callable_or_types(name):
    """Each public symbol should be importable and callable/type-checkable."""
    import goggles

    obj = getattr(goggles, name)
    if callable(obj):
        # Just call it in a safe way, catching NotImplemented or harmless errors
        try:
            sig = inspect.signature(obj)
            # For parameterless functions, call directly
            if not sig.parameters:
                obj()
            else:
                # Try minimal dummy args
                if name in {"info", "debug", "warning", "error"}:
                    obj("test message")
                elif name in {"scalar"}:
                    obj("test", 0.1)
                elif name in {"image", "video"}:
                    obj("tag", None)
                elif name in {"run"}:
                    # run() returns a context manager; don't enter it yet
                    cm = obj("test_run")
                    assert hasattr(cm, "__enter__") or hasattr(cm, "__aenter__")
                elif name == "get_logger":
                    log = obj("test")
                    assert hasattr(log, "info")
                elif name == "configure":
                    obj()
        except NotImplementedError:
            # Allowed for unimplemented stubs
            pass
        except Exception as e:
            pytest.fail(f"Calling goggles.{name}() raised unexpected error: {e}")


def test_reimport_safe(monkeypatch):
    if "goggles" in list(sys.modules):
        sys.modules.pop("goggles")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        import goggles  # noqa: F401

        importlib.reload(goggles)
    dep = [x for x in w if issubclass(x.category, DeprecationWarning)]
    # Two warnings allowed (import + reload), but nothing fatal
    assert len(dep) <= 2


def _root_state():
    """Capture a snapshot of the root logger state."""
    root = logging.getLogger()
    return {
        "level": root.level,
        "handlers": tuple(root.handlers),
        "propagate": getattr(root, "propagate", None),
    }


def test_import_has_no_side_effects(monkeypatch):
    """Importing goggles must not modify the root logger or attach handlers."""
    before = _root_state()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        import goggles  # noqa: F401

    after = _root_state()

    # Root handler set and level unchanged
    assert before == after, "Importing goggles changed global logging state"

    # Only allowed warning type: DeprecationWarning (optional)
    for rec in w:
        assert issubclass(rec.category, DeprecationWarning)

    # Check that importing added a NullHandler *only* to its own logger
    g_logger = logging.getLogger("goggles")
    nulls = [h for h in g_logger.handlers if isinstance(h, logging.NullHandler)]
    assert nulls, "Goggles logger missing NullHandler"
    assert all(isinstance(h, logging.NullHandler) for h in g_logger.handlers)


def test_reimport_is_idempotent():
    """Reloading goggles should not add extra handlers."""
    import goggles

    g_logger = logging.getLogger("goggles")
    before = len(g_logger.handlers)
    importlib.reload(goggles)
    after = len(g_logger.handlers)
    assert before == after, "Reimporting goggles duplicated handlers"
