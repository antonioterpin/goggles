"""Smoke tests for the iceoryx2-backed LocalTransport.

The previous transport's chaos / framing / endpoint / shm
fast-path tests were specific to the TCP + pickle-OOB + shared-memory
implementation that has been deleted; iceoryx2 owns those concerns
now. What's left to verify on the goggles side is the contract
between :class:`LocalTransport` and the local :class:`EventBus`:

- emit dispatches synchronously to local handlers and publishes to
  peer processes;
- shutdown is idempotent.
"""

from __future__ import annotations

from typing import ClassVar

import goggles as gg
from goggles import Event, Kind
from goggles._core.transport import LocalTransport, Transport


class _RecordingHandler:
    """Lightweight Handler-shaped recorder; mirrors test_api.DummyHandler."""

    name = "recording"
    capabilities: ClassVar[frozenset[gg.Kind]] = frozenset({"log"})
    handled: ClassVar[list[object]] = []

    def can_handle(self, kind: Kind) -> bool:
        """Accept every kind for simplicity."""
        return True

    def handle(self, event: Event) -> None:
        """Record the event on the class-level buffer."""
        type(self).handled.append(event)

    def open(self) -> None:
        """Reset the recorder buffer when entering a scope."""
        type(self).handled.clear()

    def close(self) -> None:
        """No-op; nothing to flush."""

    def to_dict(self) -> dict:
        """Serialize for cross-bus attach (not used in this transport)."""
        return {"cls": "_RecordingHandler", "data": {}}

    @classmethod
    def from_dict(cls, serialized: dict) -> _RecordingHandler:
        """Reconstruct from a serialized snapshot."""
        return cls()


def test_local_transport_implements_protocol() -> None:
    """LocalTransport satisfies the Transport runtime-checkable protocol."""
    t = LocalTransport()
    try:
        assert isinstance(t, Transport)
        assert t.is_running is True
    finally:
        t.shutdown()


def test_emit_dispatches_to_local_handler() -> None:
    """emit() reaches a handler attached on the same transport."""
    gg.register_handler(_RecordingHandler)
    _RecordingHandler.handled.clear()
    t = LocalTransport()
    try:
        t.attach([_RecordingHandler().to_dict()], ["test"])
        ev = Event(
            kind="log",
            scope="test",
            payload={"msg": "hi"},
            filepath=__file__,
            lineno=0,
            level=gg.INFO,
        )
        t.emit(ev)
        assert any(
            getattr(e, "payload", None) == {"msg": "hi"}
            for e in _RecordingHandler.handled
        ), f"event not dispatched; saw {_RecordingHandler.handled!r}"
    finally:
        t.shutdown()


def test_shutdown_is_idempotent() -> None:
    """Calling shutdown twice does not raise."""
    t = LocalTransport()
    t.shutdown()
    t.shutdown()
    assert t.is_running is False
