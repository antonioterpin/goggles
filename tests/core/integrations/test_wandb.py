import logging
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import goggles._core.integrations.wandb as wandb_module
from goggles._core.integrations.wandb import WandBHandler


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
def test_open_initializes_run(mock_wandb, reinit):
    handler = WandBHandler(project="proj", entity="ent", run_name="name", reinit=reinit)
    handler.open()
    mock_wandb.init.assert_called_once()
    kwargs = mock_wandb.init.call_args.kwargs
    assert kwargs["project"] == "proj"
    assert kwargs["entity"] == "ent"
    assert kwargs["name"] == "name"


def test_open_idempotent(mock_wandb):
    handler = WandBHandler(project="p")
    handler.open()
    handler._wandb_run = MagicMock()
    handler.open()
    mock_wandb.init.assert_called_once()  # not reopened


def test_can_handle_supported_kinds():
    h = WandBHandler()
    for kind in ["metric", "image", "video", "artifact"]:
        assert h.can_handle(kind)
    assert not h.can_handle("log")


def test_handle_metric_logs_correctly(mock_wandb):
    h = WandBHandler()
    h._wandb_run = MagicMock()
    event = make_event(kind="metric", payload={"a": 1.0, "b": 2.0}, step=5)
    h.handle(event)
    mock_wandb.log.assert_called_once_with({"a": 1.0, "b": 2.0}, step=5)


def test_handle_metric_raises_if_not_mapping(mock_wandb):
    h = WandBHandler()
    h._wandb_run = MagicMock()
    event = make_event(kind="metric", payload=[1, 2])
    with pytest.raises(ValueError):
        h.handle(event)


@pytest.mark.parametrize("kind", ["image", "video"])
def test_handle_image_video_logs(mock_wandb, kind):
    h = WandBHandler()
    fake_run = MagicMock()
    h._wandb_run = fake_run
    payload = {"x": "array"}
    event = make_event(kind=kind, payload=payload, step=3)
    h.handle(event)
    assert fake_run.log.called
    args, kwargs = fake_run.log.call_args
    assert "x" in args[0]
    assert kwargs["step"] == 3


def test_handle_artifact_uploads(mock_wandb):
    h = WandBHandler()
    fake_run = MagicMock()
    h._wandb_run = fake_run
    payload = {"path": "file.txt", "name": "f", "type": "raw"}
    event = make_event(kind="artifact", payload=payload)
    h.handle(event)
    mock_wandb.Artifact.assert_called_once_with(name="f", type="raw")
    artifact = mock_wandb.Artifact.return_value
    artifact.add_file.assert_called_with("file.txt")
    fake_run.log_artifact.assert_called_with(artifact)


def test_handle_unsupported_kind_warns(mock_wandb, caplog):
    h = WandBHandler()
    h._wandb_run = MagicMock()
    event = make_event(kind="nonsense", payload={})

    # Capture logs from the handler's logger ("wandb")
    with caplog.at_level(logging.WARNING, logger=h.name):
        h.handle(event)

    assert any("unsupported" in msg.lower() for msg in caplog.messages)


def test_get_or_create_run_creates_new(mock_wandb):
    h = WandBHandler(project="proj", entity="ent", run_name="base")
    run = h._get_or_create_run("scope1")
    mock_wandb.init.assert_called_once()
    assert h._runs["scope1"] == run


def test_close_finishes_all(mock_wandb):
    h = WandBHandler()
    h._wandb_run = MagicMock()
    h._runs = {"global": MagicMock(), "scope2": MagicMock()}
    h.close()
    assert mock_wandb.finish.call_count >= 2
    assert not h._runs


def test_to_from_dict_roundtrip():
    h = WandBHandler(
        project="p", entity="e", run_name="r", config={"x": 1}, reinit="default"
    )
    d = h.to_dict()
    restored = WandBHandler.from_dict(d)
    assert restored._project == "p"
    assert restored._entity == "e"
    assert restored._base_run_name == "r"
    assert restored._config == {"x": 1}
    assert restored._reinit == "default"
