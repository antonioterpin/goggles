"""Tests for goggles' outer API."""

import logging
from typing import ClassVar
import pytest

import goggles as gg


# ---------------------------------------------------------------------
# Logger creation
# ---------------------------------------------------------------------


def test_get_logger_returns_expected_protocols():
    """Verify that get_logger returns proper protocol instances."""
    plain = gg.get_logger("plain")
    with_metrics = gg.get_logger("metrics", with_metrics=True)

    assert isinstance(plain, gg.TextLogger)
    assert not isinstance(plain, gg.GogglesLogger)
    assert isinstance(with_metrics, gg.GogglesLogger)


def test_logger_bind_creates_new_instance():
    """Ensure that bind() returns a derived logger."""
    log = gg.get_logger("base")
    bound = log.bind(scope="run", extra_field="value")
    assert isinstance(bound, gg.TextLogger)
    assert bound is not log


# ---------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------


def test_timeit_measures_execution_time(monkeypatch):
    """Check that @timeit executes and returns the wrapped result."""
    called = {}

    @gg.timeit()
    def fn(x):
        called["x"] = x
        return x + 1

    assert fn(1) == 2
    assert called["x"] == 1


def test_timeit_nested_does_not_conflict():
    """Verify nested @timeit decorators behave correctly."""

    @gg.timeit()
    @gg.timeit()
    def inner(x):
        return x * 2

    assert inner(3) == 6


def test_trace_on_error_logs_and_rethrows():
    """trace_on_error should log and re-raise the original error."""

    @gg.trace_on_error()
    def boom(x):
        return 1 / x

    with pytest.raises(ZeroDivisionError):
        boom(0)


def test_trace_on_error_with_kwargs():
    """Ensure kwargs are included in traced error info."""

    @gg.trace_on_error()
    def boom_kw(x=0):
        return 1 / x

    with pytest.raises(ZeroDivisionError):
        boom_kw(x=0)


# ---------------------------------------------------------------------
# Handlers and EventBus
# ---------------------------------------------------------------------


class DummyHandler:
    name = "dummy"
    capabilities: ClassVar[frozenset[gg.Kind]] = frozenset({"log", "metric"})
    handled = []

    def can_handle(self, kind):
        return True

    def handle(self, event):
        self.handled.append(event)

    def open(self):
        self.handled.clear()

    def close(self):
        self.handled.append("closed")

    def to_dict(self):
        return {"cls": "DummyHandler", "data": {}}

    @classmethod
    def from_dict(cls, serialized):
        return cls()


def test_attach_and_emit(monkeypatch):
    """Attach dummy handler and ensure events are received."""
    gg.register_handler(DummyHandler)
    handler = DummyHandler()
    gg.attach(handler, scopes=["global"])

    log = gg.get_logger("demo")
    log.info("event test")
    gg.finish()

    assert "closed" in DummyHandler.handled
    assert any(hasattr(e, "msg") for e in DummyHandler.handled if e != "closed")


def test_eventbus_emit_routing_and_detach():
    """Check EventBus routes correctly and detaches handlers."""
    bus = gg.EventBus()
    handler = DummyHandler()
    bus.attach([handler.to_dict()], scopes=["scope"])
    assert "scope" in bus.scopes
    assert handler.name in bus.handlers

    # Emit dummy event
    event_dict = {
        "kind": "log",
        "scope": "scope",
        "msg": "msg",
        "data": {},
        "timestamp": 0.0,
    }
    bus.emit(event_dict)

    bus.detach("dummy", "scope")
    assert "scope" not in bus.scopes
    assert "dummy" not in bus.handlers


def test_detach_raises_for_invalid_scope():
    bus = gg.EventBus()
    handler = DummyHandler()
    bus.attach([handler.to_dict()], scopes=["valid"])
    with pytest.raises(ValueError):
        bus.detach("dummy", "invalid")


def test_eventbus_emit_ignores_unknown_scope():
    bus = gg.EventBus()
    bus.emit(
        {
            "kind": "log",
            "scope": "no_handlers",
            "msg": "none",
            "data": {},
            "timestamp": 0.0,
        }
    )  # Should not raise


# ---------------------------------------------------------------------
# LocalStorageHandler I/O
# ---------------------------------------------------------------------


def test_localstorage_handler_writes_json(tmp_path):
    path = tmp_path / "logs"
    handler = gg.LocalStorageHandler(path=path, name="api.jsonl")
    gg.attach(handler, scopes=["global"])
    log = gg.get_logger("io")
    log.info("testing write")

    gg.finish()

    jsonls = list(path.glob("*.jsonl"))
    assert jsonls
    data = jsonls[0].read_text()
    assert "testing write" in data


# ---------------------------------------------------------------------
# Environment and registry
# ---------------------------------------------------------------------


def test_environment_overrides(monkeypatch):
    monkeypatch.setenv("GOGGLES_ASYNC", "false")
    monkeypatch.setenv("GOGGLES_PORT", "9999")
    monkeypatch.setenv("GOGGLES_HOST", "remote")

    import importlib

    importlib.reload(gg)
    assert gg.GOGGLES_ASYNC is False
    assert gg.GOGGLES_PORT == "9999"
    assert gg.GOGGLES_HOST == "remote"


def test_get_handler_class_error():
    with pytest.raises(KeyError):
        gg._get_handler_class("UnknownHandler")


def test_register_handler_and_lookup():
    class MyHandler(DummyHandler):
        pass

    gg.register_handler(MyHandler)
    found = gg._get_handler_class("MyHandler")
    assert found is MyHandler


# ---------------------------------------------------------------------
# Metrics and composite logger
# ---------------------------------------------------------------------


def test_goggles_logger_scalar_and_push(tmp_path):
    handler = gg.LocalStorageHandler(path=tmp_path, name="metrics.jsonl")
    gg.attach(handler, scopes=["global"])

    tlog = gg.get_logger("train", with_metrics=True)
    tlog.scalar("loss", 0.1, step=1)
    tlog.push({"accuracy": 0.9}, step=1)
    gg.finish()

    data = next(tmp_path.glob("*.jsonl")).read_text()
    assert "loss" in data and "accuracy" in data


def test_logger_levels_mapping(monkeypatch):
    """Ensure TextLogger.log dispatches correctly by severity."""
    log = gg.get_logger("severity")

    # Should not raise for standard levels
    for lvl in [gg.DEBUG, gg.INFO, gg.WARNING, gg.ERROR, gg.CRITICAL]:
        log.log(lvl, "msg", step=0)


# ---------------------------------------------------------------------
# Finish and import safety
# ---------------------------------------------------------------------


def test_finish_multiple_times_is_safe():
    """finish() should be idempotent and safe to call repeatedly."""
    gg.finish()
    gg.finish()  # second call should not crash


def test_import_adds_nullhandler(monkeypatch):
    """Ensure module attaches a NullHandler at import time."""
    import importlib
    import goggles as gg1

    importlib.reload(gg1)
    logger = logging.getLogger(gg1.__name__)
    assert any(isinstance(h, logging.NullHandler) for h in logger.handlers)
