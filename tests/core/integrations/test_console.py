import logging
import pytest
from types import SimpleNamespace

from goggles._core.integrations.console import ConsoleHandler


class DummyEvent(SimpleNamespace):
    """Lightweight event for testing without importing full Event class."""


@pytest.fixture
def handler():
    h = ConsoleHandler()
    h.open()
    yield h
    h.close()


def test_can_handle_only_log(handler):
    assert handler.can_handle("log")
    for kind in ["metric", "image", "artifact"]:
        assert not handler.can_handle(kind)


def test_handle_logs_to_console(handler, caplog):
    event = DummyEvent(
        kind="log",
        payload="Hello world",
        level=logging.INFO,
        step=1,
        time=0.0,
        scope="run",
    )
    with caplog.at_level(logging.DEBUG):
        handler.handle(event)
    assert any("Hello world" in msg for msg in caplog.messages)


def test_handle_respects_event_level(handler, caplog):
    event = DummyEvent(
        kind="log",
        payload="debug message",
        level=logging.DEBUG,
        step=0,
        time=0.0,
        scope="run",
    )
    handler._logger.setLevel(logging.NOTSET)
    with caplog.at_level(logging.DEBUG):
        handler.handle(event)

    # Caplog stores messages in `caplog.messages`
    assert any("debug message" in msg for msg in caplog.messages)


def test_handle_raises_on_non_log_kind(handler):
    event = DummyEvent(
        kind="metric", payload="x", level=logging.INFO, step=0, time=0.0, scope="run"
    )
    with pytest.raises(ValueError):
        handler.handle(event)


def test_open_and_close_are_noops(handler):
    # Should not raise or alter logger state
    before_handlers = len(handler._logger.handlers)
    handler.open()
    handler.close()
    after_handlers = len(handler._logger.handlers)
    assert after_handlers == before_handlers


def test_multiple_initializations_do_not_duplicate_handlers():
    h1 = ConsoleHandler()
    h2 = ConsoleHandler()
    # Both use same underlying named logger; handlers should not multiply
    assert len(h1._logger.handlers) == 1
    assert h1._logger.handlers[0] is h2._logger.handlers[0]
