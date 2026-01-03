import sys
import types

import pytest
import goggles.__init__ as gg


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


class DummyHandler:
    """Minimal handler implementation for testing EventBus attach/detach/emit."""

    capabilities = frozenset({"metric", "log"})

    def __init__(self, name="dummy"):
        self.name = name
        self.opened = False
        self.closed = False
        self.handled_events = []

    def can_handle(self, kind):
        return kind in self.capabilities

    def handle(self, event):
        self.handled_events.append(event)

    def open(self):
        self.opened = True

    def close(self):
        self.closed = True

    def to_dict(self):
        return {"cls": "DummyHandler", "data": {"name": self.name}}

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


@pytest.fixture(autouse=True)
def clean_registry(monkeypatch):
    gg._HANDLER_REGISTRY.clear()
    yield
    gg._HANDLER_REGISTRY.clear()


# ---------------------------------------------------------------------------
# get_logger
# ---------------------------------------------------------------------------


def test_get_logger_returns_text_and_goggles(monkeypatch):
    dummy_text = object()
    dummy_metrics = object()
    monkeypatch.setattr(gg, "_make_text_logger", lambda n, s, t: dummy_text)
    monkeypatch.setattr(gg, "_make_goggles_logger", lambda n, s, t: dummy_metrics)

    assert gg.get_logger("x") is dummy_text
    assert gg.get_logger("x", with_metrics=True) is dummy_metrics


# ---------------------------------------------------------------------------
# _get_handler_class and register_handler
# ---------------------------------------------------------------------------


def test_register_and_get_handler_from_registry():
    gg.register_handler(DummyHandler)
    assert gg._get_handler_class("DummyHandler") is DummyHandler


def test_get_handler_class_falls_back_to_globals(monkeypatch):
    monkeypatch.setitem(gg.__dict__, "GlobalHandler", DummyHandler)
    assert gg._get_handler_class("GlobalHandler") is DummyHandler


def test_get_handler_class_raises_keyerror():
    with pytest.raises(KeyError):
        gg._get_handler_class("UnknownHandler")


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------


def test_eventbus_attach_and_emit(monkeypatch):
    gg.register_handler(DummyHandler)
    bus = gg.EventBus()
    handler_data = {"cls": "DummyHandler", "data": {"name": "h1"}}
    bus.attach([handler_data], scopes=["train"])

    assert "h1" in bus.handlers
    assert "train" in bus.scopes
    assert "h1" in bus.scopes["train"]

    # Make gg.Event be SimpleNamespace so isinstance(event, Event) passes.
    monkeypatch.setattr(gg, "Event", types.SimpleNamespace)
    event = types.SimpleNamespace(scope="train", kind="log")
    bus.emit(event)

    assert (
        bus.handlers["h1"].handled_events
        and bus.handlers["h1"].handled_events[0] is event
    )


def test_eventbus_emit_ignores_scope_and_invalid_type(monkeypatch):
    bus = gg.EventBus()
    dummy = DummyHandler()
    bus.handlers[dummy.name] = dummy
    bus.scopes["train"] = {dummy.name}

    # Wrong type -> TypeError
    with pytest.raises(TypeError):
        bus.emit(123)

    # No scope match — first ensure isinstance(event, Event) holds
    monkeypatch.setattr(gg, "Event", types.SimpleNamespace)
    event = types.SimpleNamespace(scope="other", kind="log")
    bus.emit(event)
    assert dummy.handled_events == []


def test_eventbus_detach_and_shutdown():
    gg.register_handler(DummyHandler)
    bus = gg.EventBus()
    handler_data = {"cls": "DummyHandler", "data": {"name": "h1"}}
    bus.attach([handler_data], scopes=["train"])
    assert bus.handlers["h1"].opened

    # Detach removes handler
    bus.detach("h1", "train")
    assert "h1" not in bus.handlers

    # Detach again should raise
    with pytest.raises(ValueError):
        bus.detach("h1", "train")

    # Reattach multiple and shutdown
    bus.attach([handler_data], scopes=["train", "test"])
    bus.shutdown()
    assert not bus.handlers


# ---------------------------------------------------------------------------
# attach/detach/finish wrapper functions
# ---------------------------------------------------------------------------


def test_attach_detach_finish_call_bus(monkeypatch):
    mock_bus = types.SimpleNamespace(
        attach=lambda h, s: setattr(mock_bus, "attached", (h, s)),
        detach=lambda n, s: setattr(mock_bus, "detached", (n, s)),
        shutdown=lambda: types.SimpleNamespace(
            result=lambda: setattr(mock_bus, "shut", True)
        ),
    )
    monkeypatch.setattr(gg, "get_bus", lambda: mock_bus)

    dummy = DummyHandler()
    gg.attach(dummy, scopes=["run"])
    assert mock_bus.attached[1] == ["run"]

    gg.detach("x", "scope")
    assert mock_bus.detached == ("x", "scope")

    gg.finish()
    assert getattr(mock_bus, "shut", False) is True


# ---------------------------------------------------------------------------
# get_bus caching + import hook
# ---------------------------------------------------------------------------


def test_get_bus_caches_implementation(monkeypatch):
    """Ensure get_bus imports once, caches the callable, and returns its value."""
    # Reset cache
    monkeypatch.setattr(gg, "__impl_get_bus", None, raising=True)

    # Create fake modules to satisfy: from ._core.routing import get_bus
    fake_core = types.ModuleType("goggles._core")
    fake_routing = types.ModuleType("goggles._core.routing")

    calls = {"n": 0}

    def fake_get_bus():
        calls["n"] += 1
        return "client"

    fake_routing.get_bus = fake_get_bus  # what __init__.get_bus imports

    # Inject into sys.modules so the relative import resolves to our fake
    sys.modules["goggles._core"] = fake_core
    sys.modules["goggles._core.routing"] = fake_routing

    # First call imports and caches
    result1 = gg.get_bus()
    assert result1 == "client"
    assert gg.__impl_get_bus is fake_get_bus
    assert calls["n"] == 1  # called once

    # Second call uses cached callable (no new import) and calls it again
    result2 = gg.get_bus()
    assert result2 == "client"
    assert calls["n"] == 2
