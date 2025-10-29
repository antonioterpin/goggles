import io
import json
import logging
import threading
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from goggles.types import Event
from goggles._core.integrations.jsonl import JsonlHandler


@pytest.fixture
def tmp_jsonl_path():
    with TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "log.jsonl"


@pytest.fixture
def handler(tmp_jsonl_path):
    h = JsonlHandler(tmp_jsonl_path)
    h.open()
    yield h
    h.close()


def make_event(payload="hello", level=logging.INFO, step=1, time=1.0, scope="run"):
    return Event(
        kind="log",
        scope=scope,
        payload=payload,
        filepath="test_file.py",
        lineno=10,
        level=level,
        step=step,
        time=time,
    )


def test_can_handle_and_name_attributes(handler):
    assert handler.can_handle("log")
    assert not handler.can_handle("metric")
    assert isinstance(handler.name, str)
    assert handler.capabilities == {"log"}


def test_handle_writes_json_line(handler, tmp_jsonl_path):
    event = make_event("test message")
    handler.handle(event)

    lines = tmp_jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record["payload"] == "test message"
    assert record["kind"] == "log"
    assert record["scope"] == "run"
    assert record["step"] == 1
    assert record["level"] == logging.INFO


def test_handle_raises_if_not_open(tmp_jsonl_path):
    handler = JsonlHandler(tmp_jsonl_path)
    event = make_event()
    with pytest.raises(RuntimeError):
        handler.handle(event)


def test_handle_raises_on_unsupported_kind(handler):
    """Unsupported event kinds should raise ValueError."""
    # Include all required Event fields
    from goggles.types import Event

    event = Event(
        kind="metric",
        scope="run",
        payload="x",
        level=0,
        step=0,
        time=0.0,
        filepath="file.py",
        lineno=10,
        extra=None,
    )
    with pytest.raises(ValueError):
        handler.handle(event)


def test_handle_includes_filepath_and_lineno(handler, tmp_jsonl_path):
    """Ensure filepath and lineno fields are serialized correctly."""
    event = Event(
        kind="log",
        scope="run",
        payload="with path info",
        level=logging.INFO,
        step=1,
        time=1.0,
        filepath="/tmp/example.py",
        lineno=42,
        extra=None,
    )
    handler.handle(event)

    record = json.loads(tmp_jsonl_path.read_text(encoding="utf-8").strip())
    assert record["filepath"] == "/tmp/example.py"
    assert record["lineno"] == 42


def test_handle_includes_extra_field_when_present(handler, tmp_jsonl_path):
    """Ensure extra field is serialized when provided."""
    event = Event(
        kind="log",
        scope="run",
        payload="extra test",
        level=logging.INFO,
        step=1,
        time=1.0,
        filepath="f.py",
        lineno=5,
        extra={"user": "tester", "id": 7},
    )
    handler.handle(event)

    record = json.loads(tmp_jsonl_path.read_text(encoding="utf-8").strip())
    assert "extra" in record
    assert record["extra"] == {"user": "tester", "id": 7}


def test_handle_skips_extra_when_none(handler, tmp_jsonl_path):
    """Ensure 'extra' key is omitted when None."""
    event = Event(
        kind="log",
        scope="run",
        payload="no extra",
        level=logging.INFO,
        step=1,
        time=1.0,
        filepath="f.py",
        lineno=5,
        extra=None,
    )
    handler.handle(event)

    record = json.loads(tmp_jsonl_path.read_text(encoding="utf-8").strip())
    assert "extra" not in record


def test_close_is_idempotent(handler):
    handler.close()
    handler.close()  # second call should not raise
    assert handler._fp.closed


def test_thread_safety(handler, tmp_jsonl_path):
    """Ensure multiple threads can write without corrupting the JSONL file."""
    events = [make_event(payload=f"msg-{i}") for i in range(20)]

    def worker(ev):
        handler.handle(ev)

    threads = [threading.Thread(target=worker, args=(e,)) for e in events]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    lines = tmp_jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(events)
    decoded = [json.loads(line)["payload"] for line in lines]
    assert set(decoded) == {f"msg-{i}" for i in range(20)}


def test_multiple_events_append_correctly(handler, tmp_jsonl_path):
    """Check that multiple events append to file as separate JSON lines."""
    events = [make_event(payload=f"msg-{i}") for i in range(3)]
    for ev in events:
        handler.handle(ev)

    lines = tmp_jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    for i, line in enumerate(lines):
        record = json.loads(line)
        assert record["payload"] == f"msg-{i}"
        assert record["kind"] == "log"
