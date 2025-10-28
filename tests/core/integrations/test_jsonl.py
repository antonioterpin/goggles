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
        kind="log", scope=scope, payload=payload, level=level, step=step, time=time
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
    event = make_event()
    event = event.__class__(
        kind="metric", scope="run", payload="x", level=0, step=0, time=0.0
    )
    with pytest.raises(ValueError):
        handler.handle(event)


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
