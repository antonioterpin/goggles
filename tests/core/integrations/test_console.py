import logging
import pytest
from types import SimpleNamespace
from pathlib import Path

from goggles._core.integrations.console import ConsoleHandler


class DummyEvent(SimpleNamespace):
    """Lightweight event for testing without importing full Event class."""


@pytest.fixture
def handler(tmp_path):
    """Return a ConsoleHandler with open/close lifecycle."""
    h = ConsoleHandler(project_root=tmp_path)
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
        filepath=str(Path(__file__)),
        lineno=123,
        step=1,
        time=0.0,
        scope="run",
    )
    with caplog.at_level(logging.DEBUG):
        handler.handle(event)
    assert any("Hello world" in msg for msg in caplog.messages)
    # Ensure that filepath and line number are in the log prefix
    assert any(f"{Path(__file__).name}:123" in msg for msg in caplog.messages)


def test_handle_respects_event_level(handler, caplog):
    event = DummyEvent(
        kind="log",
        payload="debug message",
        level=logging.DEBUG,
        filepath=__file__,
        lineno=77,
        step=0,
        time=0.0,
        scope="run",
    )
    handler._logger.setLevel(logging.NOTSET)
    with caplog.at_level(logging.DEBUG):
        handler.handle(event)
    assert any("debug message" in msg for msg in caplog.messages)


def test_handle_raises_on_non_log_kind(handler):
    event = DummyEvent(
        kind="metric",
        payload="x",
        level=logging.INFO,
        filepath=__file__,
        lineno=10,
        step=0,
        time=0.0,
        scope="run",
    )
    with pytest.raises(ValueError):
        handler.handle(event)


def test_open_and_close_are_noops(handler):
    before_handlers = len(handler._logger.handlers)
    handler.open()
    handler.close()
    after_handlers = len(handler._logger.handlers)
    assert after_handlers == before_handlers


def test_multiple_initializations_do_not_duplicate_handlers():
    h1 = ConsoleHandler()
    h1.open()
    h2 = ConsoleHandler()
    h2.open()
    # Both use same named logger
    assert len(h1._logger.handlers) == 1
    assert h1._logger.handlers[0] is h2._logger.handlers[0]


@pytest.mark.parametrize("style", ["absolute", "relative"])
def test_path_style_affects_output(tmp_path, caplog, style):
    """Ensure path_style option changes displayed prefix."""
    project_root = tmp_path
    fake_file = project_root / "src" / "main.py"
    event = DummyEvent(
        kind="log",
        payload="test message",
        level=logging.INFO,
        filepath=str(fake_file),
        lineno=99,
        step=0,
        time=0.0,
        scope="run",
    )
    handler = ConsoleHandler(path_style=style, project_root=project_root)
    handler.open()
    with caplog.at_level(logging.INFO):
        handler.handle(event)
    message = " ".join(caplog.messages)
    if style == "relative":
        assert "src/main.py:99" in message
        assert str(fake_file) not in message
    else:
        assert str(fake_file) in message
        assert "src/main.py:99" in message
    handler.close()


def test_to_from_dict_roundtrip(tmp_path):
    h = ConsoleHandler(
        level=logging.WARNING, path_style="absolute", project_root=tmp_path
    )
    serialized = h.to_dict()
    new = ConsoleHandler.from_dict(serialized)
    assert new.name == h.name
    assert new.level == h.level
    assert new.path_style == h.path_style
    assert Path(new.project_root) == Path(h.project_root)
