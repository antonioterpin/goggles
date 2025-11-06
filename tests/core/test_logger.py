import logging
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from goggles._core.logger import CoreTextLogger, CoreGogglesLogger


@pytest.fixture
def mock_client():
    """Mock EventBus client to capture emitted events."""
    client = MagicMock()
    future = MagicMock()
    future.result = MagicMock(return_value=None)
    client.emit = MagicMock(return_value=future)
    return client


@pytest.fixture
def patch_bus(monkeypatch, mock_client):
    """Patch get_bus() to return a mock client."""
    monkeypatch.setattr("goggles._core.routing.get_bus", lambda: mock_client)
    return mock_client


@pytest.fixture
def text_logger(patch_bus):
    """Return a CoreTextLogger bound to a dummy scope."""
    return CoreTextLogger(name="test", scope="global")


@pytest.fixture
def goggles_logger(patch_bus):
    """Return a CoreGogglesLogger bound to a dummy scope."""
    return CoreGogglesLogger(name="test", scope="global")


# -------------------------------------------------------------------------
# CoreTextLogger tests
# -------------------------------------------------------------------------


def test_bind_creates_new_context(text_logger):
    text_logger._bound = {"old": 1}
    bound_logger = text_logger.bind(scope="run", new=2)
    assert bound_logger.get_bound() == {"old": 1, "new": 2}
    assert bound_logger._scope == "run"


@pytest.mark.parametrize(
    "level,method",
    [
        (logging.DEBUG, "debug"),
        (logging.INFO, "info"),
        (logging.WARNING, "warning"),
        (logging.ERROR, "error"),
        (logging.CRITICAL, "critical"),
    ],
)
def test_log_methods_emit_event(text_logger, patch_bus, level, method):
    msg = f"message-{method}"
    getattr(text_logger, method)(msg, step=1, time=123.0, extra_field="x")
    assert patch_bus.emit.called
    event_dict = patch_bus.emit.call_args[0][0]
    assert event_dict["kind"] == "log"
    assert event_dict["payload"] == msg
    assert event_dict["level"] == level
    assert event_dict["extra"]["extra_field"] == "x"


def test_repr_includes_name_and_bound(text_logger):
    text_logger._bound = {"a": 1}
    rep = repr(text_logger)
    assert "CoreTextLogger" in rep
    assert "a" in rep
    assert "test" in rep


# -------------------------------------------------------------------------
# CoreGogglesLogger tests
# -------------------------------------------------------------------------


def test_push_emits_metric_event(goggles_logger, patch_bus):
    metrics = {"loss": 0.1, "acc": 0.9}
    goggles_logger.push(metrics, step=2)
    event_dict = patch_bus.emit.call_args[0][0]
    assert event_dict["kind"] == "metric"
    assert event_dict["payload"] == metrics
    assert event_dict["step"] == 2


def test_scalar_emits_metric_event(goggles_logger, patch_bus):
    goggles_logger.scalar("loss", 0.42)
    event_dict = patch_bus.emit.call_args[0][0]
    assert event_dict["kind"] == "metric"
    assert event_dict["payload"] == {"loss": 0.42}


@pytest.mark.parametrize(
    "kind,method,arg_key",
    [
        ("image", "image", "image"),
        ("video", "video", "video"),
        ("artifact", "artifact", "data"),
        ("vector_field", "vector_field", "vector_field"),
        ("histogram", "histogram", "histogram"),
    ],
)
def test_artifact_like_methods_emit_event(
    goggles_logger, patch_bus, kind, method, arg_key
):
    fake_payload = SimpleNamespace(dummy=True)
    kwargs = {}
    if method == "video":
        kwargs["fps"] = 60
    getattr(goggles_logger, method)(fake_payload, name="foo", **kwargs)
    event_dict = patch_bus.emit.call_args[0][0]
    assert event_dict["kind"] == kind
    assert "name" in event_dict["extra"]
    assert event_dict["extra"]["name"] == "foo"


def test_histogram_adds_name_and_payload(goggles_logger, patch_bus):
    goggles_logger.histogram([1, 2, 3], name="hist")
    event_dict = patch_bus.emit.call_args[0][0]
    assert event_dict["kind"] == "histogram"
    assert event_dict["extra"]["name"] == "hist"
    assert event_dict["payload"] == [1, 2, 3]


def test_all_emitters_call_future_result(monkeypatch, patch_bus):
    """Ensure synchronous mode calls future.result()."""
    import goggles._core.logger as core_logger

    monkeypatch.setattr(core_logger, "GOGGLES_ASYNC", False)
    g = CoreGogglesLogger(name="sync", scope="run")
    g._client = patch_bus

    g.scalar("metric", 1.0)
    future = patch_bus.emit.return_value
    future.result.assert_called_once()
