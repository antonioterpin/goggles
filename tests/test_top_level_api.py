import sys
import threading
import time
import types
from typing import Any, ClassVar, cast

import pytest

import goggles.__init__ as gg

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


class DummyHandler:
    """Minimal handler implementation for testing EventBus attach/detach/emit.

    Attributes:
        capabilities: Supported event kinds for this test handler.
    """

    capabilities: ClassVar[frozenset[gg.Kind]] = cast(
        frozenset[gg.Kind], frozenset({"metric", "log"})
    )

    def __init__(self, name: str = "dummy") -> None:
        """Initialize a dummy handler instance.

        Args:
            name: Handler name used as registry key.
        """
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
    def from_dict(cls, serialized: dict[Any, Any]):
        return cls(**serialized)


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
    monkeypatch.setattr(
        gg, "_make_text_logger", lambda n, s, level, **t: dummy_text
    )
    monkeypatch.setattr(
        gg, "_make_goggles_logger", lambda n, s, level, **t: dummy_metrics
    )

    assert gg.get_logger("x") is dummy_text, (
        "get_logger('x') should return the text logger"
    )
    assert gg.get_logger("x", with_metrics=True) is dummy_metrics, (
        "get_logger('x', with_metrics=True) should return the goggles logger"
    )


# ---------------------------------------------------------------------------
# _get_handler_class and register_handler
# ---------------------------------------------------------------------------


def test_register_and_get_handler_from_registry():
    gg.register_handler(DummyHandler)
    assert gg._get_handler_class("DummyHandler") is DummyHandler, (
        "Should find DummyHandler in registry"
    )


def test_get_handler_class_falls_back_to_globals(monkeypatch):
    monkeypatch.setitem(gg.__dict__, "GlobalHandler", DummyHandler)
    assert gg._get_handler_class("GlobalHandler") is DummyHandler, (
        "Should find GlobalHandler in globals"
    )


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

    assert "h1" in bus.handlers, "'h1' handler should be attached to the bus"
    assert "train" in bus.scopes, (
        "'train' scope should be registered in the bus"
    )
    assert "h1" in bus.scopes["train"], (
        "'h1' handler should be in the 'train' scope"
    )

    event = gg.Event("log", "train", "msg", filepath=__file__, lineno=1)
    bus.emit(event)

    attached = cast(DummyHandler, bus.handlers["h1"])
    assert attached.handled_events and attached.handled_events[0] is event, (
        "The event should have been handled by handler 'h1'"
    )


def test_eventbus_emit_ignores_scope_and_invalid_type(monkeypatch):
    gg.register_handler(DummyHandler)
    bus = gg.EventBus()
    dummy = DummyHandler(name="dummy")
    bus.attach([dummy.to_dict()], scopes=["train"])
    attached = cast(DummyHandler, bus.handlers[dummy.name])

    # Wrong type -> TypeError
    with pytest.raises(TypeError):
        bus.emit(cast(Any, 123))

    # No scope match.
    event = gg.Event("log", "other", "msg", filepath=__file__, lineno=1)
    bus.emit(event)
    assert attached.handled_events == [], (
        "Should have no handled events for mismatched scope"
    )


def test_eventbus_detach_and_shutdown():
    gg.register_handler(DummyHandler)
    bus = gg.EventBus()
    handler_data = {"cls": "DummyHandler", "data": {"name": "h1"}}
    bus.attach([handler_data], scopes=["train"])
    assert cast(DummyHandler, bus.handlers["h1"]).opened, (
        "Handler 'h1' should be opened upon attachment"
    )

    # Detach removes handler
    bus.detach("h1", "train")
    assert "h1" not in bus.handlers, (
        "Handler 'h1' should be removed after detach"
    )

    # Detach again should raise
    with pytest.raises(ValueError):
        bus.detach("h1", "train")

    # Reattach multiple and shutdown
    bus.attach([handler_data], scopes=["train", "test"])
    bus.shutdown()
    assert not bus.handlers, "Bus should have no handlers after shutdown"


class _SlowCloseHandler:
    """Test handler whose ``close()`` blocks for a configurable delay.

    Attributes:
        capabilities: Event kinds claimed (just metric/log here).
    """

    capabilities: ClassVar[frozenset[gg.Kind]] = cast(
        frozenset[gg.Kind], frozenset({"metric", "log"})
    )

    def __init__(self, name: str = "slow", close_delay: float = 0.0) -> None:
        """Initialize a handler whose ``close`` blocks for ``close_delay``.

        Args:
            name: Handler name used as registry key.
            close_delay: Seconds ``close`` sleeps before returning.
        """
        self.name = name
        self.close_delay = close_delay
        self.closed = False
        self.close_started = threading.Event()

    def can_handle(self, kind: gg.Kind) -> bool:
        return kind in self.capabilities

    def handle(self, event: gg.Event) -> None:
        del event

    def open(self) -> None: ...

    def close(self) -> None:
        self.close_started.set()
        if self.close_delay > 0:
            time.sleep(self.close_delay)
        self.closed = True

    def to_dict(self) -> dict:
        return {
            "cls": "_SlowCloseHandler",
            "data": {"name": self.name, "close_delay": self.close_delay},
        }

    @classmethod
    def from_dict(cls, serialized: dict) -> "_SlowCloseHandler":
        return cls(**serialized)


def test_eventbus_shutdown_default_waits_for_slow_close():
    """``shutdown()`` with no timeout must wait for ``close()`` to return."""
    gg.register_handler(_SlowCloseHandler)
    bus = gg.EventBus()
    handler = _SlowCloseHandler(name="h1", close_delay=0.3)
    bus.attach([handler.to_dict()], scopes=["train"])
    attached = cast(_SlowCloseHandler, bus.handlers["h1"])
    bus.shutdown()  # timeout=None
    assert attached.closed, "Default shutdown must wait for close() to complete"
    assert not bus.handlers, "Handlers must be cleared after shutdown"


def test_eventbus_shutdown_timeout_abandons_hung_close():
    """``shutdown(timeout=T)`` must return without blocking on a hung close."""
    gg.register_handler(_SlowCloseHandler)
    bus = gg.EventBus()
    handler = _SlowCloseHandler(name="h1", close_delay=5.0)
    bus.attach([handler.to_dict()], scopes=["train"])
    attached = cast(_SlowCloseHandler, bus.handlers["h1"])
    start = time.monotonic()
    bus.shutdown(timeout=0.2)
    elapsed = time.monotonic() - start
    assert elapsed < 2.0, (
        f"Bounded shutdown must not block on hung close(); took {elapsed:.2f}s"
    )
    assert attached.close_started.is_set(), (
        "close() must have been invoked even though we did not wait for it"
    )
    assert not attached.closed, (
        "close() should still be running (sleep=5s, timeout=0.2s)"
    )
    assert not bus.handlers, (
        "Handlers must still be cleared after timed-out shutdown"
    )


def test_eventbus_shutdown_closes_all_handlers_in_parallel():
    """Slow handlers must close concurrently, not one-after-another.

    Each handler sleeps ``delay`` seconds in close(). Serial close would
    take ~``n*delay``; parallel close should finish in roughly ``delay``.
    """
    gg.register_handler(_SlowCloseHandler)
    bus = gg.EventBus()
    n = 4
    delay = 0.3
    for i in range(n):
        handler = _SlowCloseHandler(name=f"h{i}", close_delay=delay)
        bus.attach([handler.to_dict()], scopes=[f"s{i}"])
    start = time.monotonic()
    bus.shutdown()
    elapsed = time.monotonic() - start
    # Generous slack for thread scheduling jitter, but well below the
    # serial baseline of n*delay = 1.2s.
    assert elapsed < n * delay * 0.75, (
        f"Close threads should run in parallel; serial baseline {n * delay}s, "
        f"parallel observed {elapsed:.2f}s"
    )
    assert not bus.handlers, "All handlers must be cleared after shutdown"


# ---------------------------------------------------------------------------
# attach/detach/finish wrapper functions
# ---------------------------------------------------------------------------


def test_attach_detach_finish_call_bus(monkeypatch):
    def _attach(*, handlers, scopes):
        mock_bus.attached = (handlers, scopes)

    def _detach(n, s):
        mock_bus.detached = (n, s)

    def _shutdown(timeout=None):
        mock_bus.shut = True

    mock_bus = types.SimpleNamespace(
        attach=_attach,
        detach=_detach,
        shutdown=_shutdown,
    )
    monkeypatch.setattr(cast(Any, gg), "get_bus", lambda: mock_bus)

    dummy = DummyHandler()
    gg.attach(dummy, scopes=["run"])
    assert mock_bus.attached[1] == ["run"], (
        "Attach should be called with correct scopes"
    )

    gg.detach("x", "scope")
    assert mock_bus.detached == (
        "x",
        "scope",
    ), "Detach should be called with correct name and scope"

    gg.finish()
    assert getattr(mock_bus, "shut", False) is True, (
        "Shutdown should have been called on finish"
    )


# ---------------------------------------------------------------------------
# get_bus caching + import hook
# ---------------------------------------------------------------------------


def test_get_bus_caches_implementation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure get_bus imports once, caches the callable, and returns its value.

    Args:
        monkeypatch: Fixture used to patch module state and imports.
    """
    # Reset cache
    monkeypatch.setattr(gg, "__impl_get_bus", None, raising=True)

    # Create fake modules to satisfy: from ._core.routing import get_bus
    fake_core = types.ModuleType("goggles._core")
    fake_routing = types.ModuleType("goggles._core.routing")

    calls = {"n": 0}

    def fake_get_bus():
        calls["n"] += 1
        return "client"

    cast(Any, fake_routing).get_bus = fake_get_bus

    # Inject into sys.modules so the relative import resolves to our fake
    sys.modules["goggles._core"] = fake_core
    sys.modules["goggles._core.routing"] = fake_routing

    # First call imports and caches
    result1 = gg.get_bus()
    assert result1 == "client", (
        "get_bus should return the client from the implementation"
    )
    assert gg.__impl_get_bus is fake_get_bus, (
        "Internal cache should store the implementation"
    )
    assert calls["n"] == 1, (
        "Implementation should be called exactly once for the first get_bus()"
    )

    # Second call uses cached callable (no new import) and calls it again
    result2 = gg.get_bus()
    assert result2 == "client", (
        "Subsequent calls to get_bus should still return the client"
    )
    assert calls["n"] == 2, (
        "Implementation itself should be called again, but not re-imported"
    )
