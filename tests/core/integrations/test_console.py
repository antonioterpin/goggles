import logging
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Literal, cast

import pytest

from goggles._core.integrations.console import ConsoleHandler

if TYPE_CHECKING:
    from goggles import Event


class DummyEvent(SimpleNamespace):
    """Lightweight event for testing without importing full Event class."""


def _capture_logger_messages(
    logger: logging.Logger,
) -> tuple[list[str], logging.Handler]:
    messages: list[str] = []

    class _MessageCollector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            messages.append(record.getMessage())

    collector = _MessageCollector()
    logger.addHandler(collector)
    return messages, collector


@pytest.fixture
def handler(tmp_path: Path) -> Iterator[ConsoleHandler]:
    """Return a ConsoleHandler with open/close lifecycle.

    Args:
        tmp_path: Temporary project root used by the handler.

    Yields:
        ConsoleHandler: Open console handler instance.
    """
    h = ConsoleHandler(project_root=tmp_path)
    h.open()
    yield h
    h.close()


def test_can_handle_only_log(handler):
    assert handler.can_handle("log"), (
        "ConsoleHandler should handle 'log' events"
    )
    for kind in ["metric", "image", "artifact"]:
        assert not handler.can_handle(kind), (
            f"ConsoleHandler should not handle '{kind}' events"
        )


def test_handle_logs_to_console(handler):
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
    messages, collector = _capture_logger_messages(handler._logger)
    try:
        handler.handle(event)
    finally:
        handler._logger.removeHandler(collector)
    output = " ".join(messages)
    assert "Hello world" in output, (
        "Log message 'Hello world' not found in console output"
    )
    # Ensure that filepath and line number are in the log prefix
    assert f"{Path(__file__).name}:123" in output, (
        "Source file:line prefix not found in console output"
    )


def test_handle_respects_event_level(handler):
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
    handler._logger.setLevel(logging.DEBUG)
    messages, collector = _capture_logger_messages(handler._logger)
    try:
        handler.handle(event)
    finally:
        handler._logger.removeHandler(collector)
    output = " ".join(messages)
    assert "debug message" in output, (
        "Debug message not found after setting proper log level"
    )


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
    assert after_handlers == before_handlers, (
        "open/close should not permanently change the number of handlers"
    )


def test_multiple_initializations_do_not_duplicate_handlers():
    h1 = ConsoleHandler()
    h1.open()
    h2 = ConsoleHandler()
    h2.open()
    # Both use same named logger
    assert len(h1._logger.handlers) == 1, (
        "Should have exactly one console handler "
        "even with multiple handler instances"
    )
    assert h1._logger.handlers[0] is h2._logger.handlers[0], (
        "Multiple handler instances should share the same logger handler"
    )


@pytest.mark.parametrize("style", ["absolute", "relative"])
def test_path_style_affects_output(
    tmp_path: Path,
    style: Literal["absolute", "relative"],
) -> None:
    """Ensure path_style option changes displayed prefix.

    Args:
        tmp_path: Temporary root used to compute relative paths.
        style: Configured path display style.
    """
    project_root = tmp_path
    fake_file = project_root / "src" / "main.py"
    event = cast(
        "Event",
        DummyEvent(
            kind="log",
            payload="test message",
            level=logging.INFO,
            filepath=str(fake_file),
            lineno=99,
            step=0,
            time=0.0,
            scope="run",
        ),
    )
    handler = ConsoleHandler(path_style=style, project_root=project_root)
    handler.open()
    messages, collector = _capture_logger_messages(handler._logger)
    try:
        handler.handle(event)
    finally:
        handler._logger.removeHandler(collector)
    message = " ".join(messages)
    if style == "relative":
        assert "src/main.py:99" in message, (
            "Relative path prefix missing from output"
        )
        assert str(fake_file) not in message, (
            "Absolute path found in relative path style output"
        )
    else:
        assert str(fake_file) in message, (
            "Absolute path prefix missing from output"
        )
        assert "src/main.py:99" in message, (
            "File:line info missing from absolute path output"
        )
    handler.close()


def test_to_from_dict_roundtrip(tmp_path):
    h = ConsoleHandler(
        level=logging.WARNING, path_style="absolute", project_root=tmp_path
    )
    serialized = h.to_dict()
    new = ConsoleHandler.from_dict(serialized)
    assert new.name == h.name, "Deserialized handler name mismatch"
    assert new.level == h.level, "Deserialized handler level mismatch"
    assert new.path_style == h.path_style, (
        "Deserialized handler path_style mismatch"
    )
    assert Path(new.project_root) == Path(h.project_root), (
        "Deserialized handler project_root mismatch"
    )
