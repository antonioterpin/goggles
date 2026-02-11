import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import goggles._core.integrations.wandb as wandb_module
from goggles._core.integrations.wandb import WandBHandler


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
def mock_wandb(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr(wandb_module, "wandb", mock)
    return mock


def make_event(kind="metric", scope="global", payload=None, step=0):
    return SimpleNamespace(kind=kind, scope=scope, payload=payload, step=step)


@pytest.mark.parametrize(
    "reinit", ["finish_previous", "return_previous", "create_new", "default"]
)
def test_open_is_noop(mock_wandb, reinit):
    handler = WandBHandler(
        project="proj", entity="ent", run_name="name", reinit=reinit
    )
    handler.open()
    mock_wandb.init.assert_not_called()


def test_open_idempotent(mock_wandb):
    handler = WandBHandler(project="p")
    handler.open()
    handler.open()
    mock_wandb.init.assert_not_called()


def test_can_handle_supported_kinds():
    h = WandBHandler()
    for kind in ["metric", "image", "video", "artifact"]:
        assert h.can_handle(kind), f"WandBHandler should handle '{kind}' events"
    assert not h.can_handle("log"), (
        "WandBHandler should not handle 'log' events by default"
    )


def test_handle_metric_raises_if_not_mapping(mock_wandb):
    h = WandBHandler()
    event = make_event(kind="metric", payload=[1, 2])
    with pytest.raises(ValueError):
        h.handle(event)


def test_handle_unsupported_kind_warns(mock_wandb):
    h = WandBHandler()
    event = make_event(kind="nonsense", payload={})

    messages, collector = _capture_logger_messages(h._logger)
    try:
        h.handle(event)
    finally:
        h._logger.removeHandler(collector)
    assert any("unsupported" in msg.lower() for msg in messages), (
        "Should log a warning for unsupported event kind"
    )


def test_get_or_create_run_creates_new(mock_wandb):
    h = WandBHandler(project="proj", entity="ent", run_name="base")
    run = h._get_or_create_run("scope1", extra_config={})
    mock_wandb.init.assert_called_once()
    assert h._runs["scope1"] == run, (
        "Run should be cached under the given scope"
    )
