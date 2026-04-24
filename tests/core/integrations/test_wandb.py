import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
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
    for kind in [
        "metric",
        "image",
        "video",
        "artifact",
        "vector_field",
        "histogram",
    ]:
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


def test_handle_artifact_uploads_file(mock_wandb, tmp_path):
    artifact_file = tmp_path / "random_artifact.npy"
    artifact_file.write_bytes(b"dummy")

    h = WandBHandler(project="proj")
    event = SimpleNamespace(
        kind="artifact",
        scope="global",
        payload={
            "path": str(artifact_file),
            "name": "random_artifact",
            "type": "misc",
        },
        step=1,
        extra={},
    )

    h.handle(event)

    mock_wandb.Artifact.assert_called_once_with(
        name="random_artifact", type="misc", metadata={}
    )
    mock_wandb.Artifact.return_value.add_file.assert_called_once_with(
        str(artifact_file)
    )
    mock_wandb.init.return_value.log_artifact.assert_called_once_with(
        mock_wandb.Artifact.return_value
    )


def test_handle_artifact_non_mapping_warns(mock_wandb):
    h = WandBHandler(project="proj")
    event = SimpleNamespace(
        kind="artifact",
        scope="global",
        payload=np.zeros((4, 4)),
        step=0,
        extra={},
    )

    messages, collector = _capture_logger_messages(h._logger)
    try:
        h.handle(event)
    finally:
        h._logger.removeHandler(collector)

    assert any("must be a mapping" in m.lower() for m in messages), (
        "Should warn when artifact payload is not a mapping"
    )
    mock_wandb.Artifact.assert_not_called()


def test_handle_vector_field_logs_image(mock_wandb, monkeypatch):
    h = WandBHandler(project="proj")
    event = SimpleNamespace(
        kind="vector_field",
        scope="global",
        payload=np.zeros((16, 16, 2), dtype=np.float32),
        step=3,
        extra={
            "name": "flow",
            "mode": "vorticity",
            "add_colorbar": True,
            "tag": "viz",
        },
    )

    mocked_image = np.zeros((32, 32, 3), dtype=np.uint8)
    render_mock = MagicMock(return_value=mocked_image)
    monkeypatch.setattr(
        wandb_module, "create_numpy_vector_field_visualization", render_mock
    )

    h.handle(event)

    run = mock_wandb.init.return_value
    render_mock.assert_called_once_with(
        event.payload,
        mode="vorticity",
        add_colorbar=True,
    )
    mock_wandb.Image.assert_called_once_with(mocked_image)
    run.log.assert_called_once()
    logged_payload = run.log.call_args[0][0]
    assert "flow" in logged_payload, (
        "Logged payload should contain the field name as key"
    )
    assert logged_payload["tag"] == "viz", (
        "Logged payload should include extra fields"
    )
    assert run.log.call_args.kwargs["step"] == 3, (
        "Logged payload should include the event step"
    )


def test_handle_vector_field_unknown_mode_warns_and_skips(mock_wandb):
    h = WandBHandler(project="proj")
    event = SimpleNamespace(
        kind="vector_field",
        scope="global",
        payload=np.zeros((16, 16, 2), dtype=np.float32),
        step=0,
        extra={"mode": "unknown"},
    )

    messages, collector = _capture_logger_messages(h._logger)
    try:
        h.handle(event)
    finally:
        h._logger.removeHandler(collector)

    assert any(
        "unknown vector field visualization mode" in m.lower() for m in messages
    ), "Should log a warning about the unknown visualization mode"
    run = mock_wandb.init.return_value
    run.log.assert_not_called()


@pytest.mark.parametrize(
    "shape, expected_channels",
    [((5, 8, 12, 1), 3), ((5, 8, 12, 3), 3), ((5, 8, 12, 4), 4)],
    ids=["channels_last_gray", "channels_last_rgb", "channels_last_rgba"],
)
def test_prepare_video_channels_last(shape, expected_channels):
    h = WandBHandler(project="proj")
    F, H, W, _ = shape
    value = np.full(shape, 128, dtype=np.uint8)
    out = h._prepare_video_for_wandb(value)
    assert out.shape == (F, expected_channels, H, W), (
        f"Expected (F, {expected_channels}, H, W) for input {shape},"
        f" got {out.shape}"
    )


def test_prepare_video_channels_first_preserved():
    h = WandBHandler(project="proj")
    F, C, H, W = 5, 3, 8, 12
    value = np.full((F, C, H, W), 128, dtype=np.uint8)
    out = h._prepare_video_for_wandb(value)
    assert out.shape == (F, 3, H, W)


def test_prepare_video_channels_first_grayscale_repeated():
    h = WandBHandler(project="proj")
    F, H, W = 5, 8, 12
    value = np.full((F, 1, H, W), 128, dtype=np.uint8)
    out = h._prepare_video_for_wandb(value)
    assert out.shape == (F, 3, H, W)
