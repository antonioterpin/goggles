import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from mlflow.tracking import MlflowClient

import goggles as gg
import goggles._core.integrations as integ
import goggles._core.integrations.mlflow as mlflow_module
from goggles._core.integrations.mlflow import (
    MLflowHandler,
    _video_to_channels_last,
)


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
def mock_client(monkeypatch):
    """Patch ``MlflowClient`` so handlers talk to a MagicMock client."""
    client = MagicMock()
    client.get_experiment_by_name.return_value = SimpleNamespace(
        experiment_id="0"
    )
    client.create_run.return_value = SimpleNamespace(
        info=SimpleNamespace(run_id="run-0")
    )
    factory = MagicMock(return_value=client)
    monkeypatch.setattr(mlflow_module, "MlflowClient", factory)
    client._factory = factory
    return client


def make_event(
    kind="metric",
    scope="global",
    payload=None,
    step: int | None = 0,
    extra=None,
):
    return SimpleNamespace(
        kind=kind,
        scope=scope,
        payload=payload,
        step=step,
        extra=extra or {},
        time=None,
    )


# Capabilities, the monotonic-step guard, serialization round-tripping and
# event.extra non-mutation are the shared Handler contract -- exercised for
# both trackers in test_handler_contract.py. The tests below cover behaviour
# specific to how MLflowHandler forwards events to MlflowClient.


def test_open_resolves_existing_experiment(mock_client):
    h = MLflowHandler(experiment="exp", tracking_uri="http://srv:5000")
    h.open()
    mock_client._factory.assert_called_once_with(tracking_uri="http://srv:5000")
    mock_client.get_experiment_by_name.assert_called_once_with("exp")
    mock_client.create_experiment.assert_not_called()
    assert h._experiment_id == "0"


def test_open_creates_experiment_when_missing(mock_client):
    mock_client.get_experiment_by_name.return_value = None
    mock_client.create_experiment.return_value = "42"
    h = MLflowHandler(experiment="brand-new")
    h.open()
    mock_client.create_experiment.assert_called_once_with(
        "brand-new", artifact_location=None
    )
    assert h._experiment_id == "42"


def test_open_defaults_to_default_experiment(mock_client):
    h = MLflowHandler()
    h.open()
    mock_client.get_experiment_by_name.assert_called_once_with("Default")


def test_get_or_create_run_creates_and_caches(mock_client):
    h = MLflowHandler(experiment="exp", run_name="base", tags={"team": "rl"})
    h.handle(make_event(kind="metric", payload={"loss": 1.0}, step=0))
    h.handle(make_event(kind="metric", payload={"loss": 0.5}, step=1))

    mock_client.create_run.assert_called_once()
    kwargs = mock_client.create_run.call_args.kwargs
    assert kwargs["experiment_id"] == "0"
    assert kwargs["run_name"] == "base"  # GLOBAL_SCOPE uses base name as-is
    assert kwargs["tags"] == {"team": "rl", "goggles.scope": "global"}
    assert h._runs["global"] == "run-0"


def test_run_name_includes_scope_for_non_global(mock_client):
    h = MLflowHandler(run_name="base")
    h.handle(make_event(kind="metric", payload={"x": 1}, scope="train", step=0))
    assert mock_client.create_run.call_args.kwargs["run_name"] == "base-train"


def test_params_logged_once_on_run_creation(mock_client):
    h = MLflowHandler(params={"lr": 0.1, "seed": 7})
    h.handle(make_event(kind="metric", payload={"loss": 1.0}, step=0))

    param_calls = [
        c
        for c in mock_client.log_batch.call_args_list
        if c.kwargs.get("params")
    ]
    assert len(param_calls) == 1, "params should be logged exactly once"
    logged = {p.key: p.value for p in param_calls[0].kwargs["params"]}
    assert logged == {"lr": "0.1", "seed": "7"}, "params stringified for MLflow"


def test_handle_metric_logs_batch(mock_client):
    h = MLflowHandler()
    h.handle(
        make_event(kind="metric", payload={"loss": 0.25, "acc": 0.9}, step=7)
    )
    metric_calls = [
        c
        for c in mock_client.log_batch.call_args_list
        if c.kwargs.get("metrics")
    ]
    assert len(metric_calls) == 1
    metrics = {m.key: m for m in metric_calls[0].kwargs["metrics"]}
    assert set(metrics) == {"loss", "acc"}
    assert metrics["loss"].value == 0.25
    assert metrics["loss"].step == 7
    assert metric_calls[0].args[0] == "run-0"


def test_handle_metric_raises_if_not_mapping(mock_client):
    h = MLflowHandler()
    with pytest.raises(ValueError):
        h.handle(make_event(kind="metric", payload=[1, 2]))


def test_handle_metric_skips_non_numeric(mock_client):
    h = MLflowHandler()
    h.handle(
        make_event(kind="metric", payload={"loss": 1.0, "note": "hi"}, step=0)
    )
    metrics = {
        m.key
        for c in mock_client.log_batch.call_args_list
        if c.kwargs.get("metrics")
        for m in c.kwargs["metrics"]
    }
    assert metrics == {"loss"}, "non-numeric values must be skipped"


def test_handle_metric_numeric_extra_logged(mock_client):
    h = MLflowHandler()
    h.handle(
        make_event(
            kind="metric",
            payload={"loss": 1.0},
            step=0,
            extra={"custom_step": 3, "split": "train"},
        )
    )
    metrics = {
        m.key: m.value
        for c in mock_client.log_batch.call_args_list
        if c.kwargs.get("metrics")
        for m in c.kwargs["metrics"]
    }
    assert metrics == {"loss": 1.0, "custom_step": 3.0}, (
        "numeric extras become metrics; string extras are ignored"
    )


def test_handle_metric_all_non_numeric_warns(mock_client):
    h = MLflowHandler()
    messages, collector = _capture_logger_messages(h._logger)
    try:
        h.handle(make_event(kind="metric", payload={"note": "x"}, step=0))
    finally:
        h._logger.removeHandler(collector)
    assert any("no numeric values" in m.lower() for m in messages)
    assert not any(
        c.kwargs.get("metrics") for c in mock_client.log_batch.call_args_list
    )


def test_handle_image_logs_log_image(mock_client):
    h = MLflowHandler()
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    h.handle(
        SimpleNamespace(
            kind="image",
            scope="global",
            payload=img,
            step=5,
            extra={"name": "camera"},
            time=None,
        )
    )
    mock_client.log_image.assert_called_once()
    args, kwargs = mock_client.log_image.call_args
    assert args[0] == "run-0"
    np.testing.assert_array_equal(args[1], img)
    assert kwargs["key"] == "camera"
    assert kwargs["step"] == 5


def test_handle_artifact_uploads_file(mock_client, tmp_path):
    artifact_file = tmp_path / "weights.npy"
    artifact_file.write_bytes(b"dummy")
    h = MLflowHandler()
    h.handle(
        SimpleNamespace(
            kind="artifact",
            scope="global",
            payload={"path": str(artifact_file), "name": "weights"},
            step=1,
            extra={},
            time=None,
        )
    )
    mock_client.log_artifact.assert_called_once_with(
        "run-0", str(artifact_file), artifact_path="weights"
    )
    mock_client.log_artifacts.assert_not_called()


def test_handle_artifact_uploads_directory(mock_client, tmp_path):
    ckpt = tmp_path / "checkpoint_step_42"
    ckpt.mkdir()
    (ckpt / "params.msgpack").write_bytes(b"weights")
    h = MLflowHandler()
    h.handle(
        SimpleNamespace(
            kind="artifact",
            scope="global",
            payload={"path": str(ckpt), "name": "model"},
            step=42,
            extra={},
            time=None,
        )
    )
    mock_client.log_artifacts.assert_called_once_with(
        "run-0", str(ckpt), artifact_path="model"
    )
    mock_client.log_artifact.assert_not_called()


def test_handle_artifact_missing_path_warns(mock_client, tmp_path):
    h = MLflowHandler()
    messages, collector = _capture_logger_messages(h._logger)
    try:
        h.handle(
            SimpleNamespace(
                kind="artifact",
                scope="global",
                payload={"path": str(tmp_path / "nope"), "name": "x"},
                step=0,
                extra={},
                time=None,
            )
        )
    finally:
        h._logger.removeHandler(collector)
    assert any("does not exist" in m.lower() for m in messages)
    mock_client.log_artifact.assert_not_called()


def test_handle_artifact_non_mapping_warns(mock_client):
    h = MLflowHandler()
    messages, collector = _capture_logger_messages(h._logger)
    try:
        h.handle(
            SimpleNamespace(
                kind="artifact",
                scope="global",
                payload=np.zeros((4, 4)),
                step=0,
                extra={},
                time=None,
            )
        )
    finally:
        h._logger.removeHandler(collector)
    assert any("must be a mapping" in m.lower() for m in messages)
    mock_client.log_artifact.assert_not_called()


def test_handle_vector_field_logs_image(mock_client, monkeypatch):
    rendered = np.zeros((32, 32, 3), dtype=np.uint8)
    render_mock = MagicMock(return_value=rendered)
    monkeypatch.setattr(
        mlflow_module, "create_numpy_vector_field_visualization", render_mock
    )
    h = MLflowHandler()
    h.handle(
        SimpleNamespace(
            kind="vector_field",
            scope="global",
            payload=np.zeros((16, 16, 2), dtype=np.float32),
            step=3,
            extra={"name": "flow", "mode": "vorticity", "add_colorbar": True},
            time=None,
        )
    )
    render_mock.assert_called_once()
    assert render_mock.call_args.kwargs["mode"] == "vorticity"
    assert render_mock.call_args.kwargs["add_colorbar"] is True
    args, kwargs = mock_client.log_image.call_args
    np.testing.assert_array_equal(args[1], rendered)
    assert kwargs["key"] == "flow"
    assert kwargs["step"] == 3


def test_handle_vector_field_unknown_mode_warns_and_skips(mock_client):
    h = MLflowHandler()
    messages, collector = _capture_logger_messages(h._logger)
    try:
        h.handle(
            SimpleNamespace(
                kind="vector_field",
                scope="global",
                payload=np.zeros((16, 16, 2), dtype=np.float32),
                step=0,
                extra={"mode": "nonsense"},
                time=None,
            )
        )
    finally:
        h._logger.removeHandler(collector)
    assert any("unknown vector field" in m.lower() for m in messages)
    mock_client.log_image.assert_not_called()


def test_handle_trajectories_logs_image(mock_client, monkeypatch):
    rendered = np.zeros((32, 32, 3), dtype=np.uint8)
    render_mock = MagicMock(return_value=rendered)
    monkeypatch.setattr(
        mlflow_module, "create_numpy_trajectories_visualization", render_mock
    )
    h = MLflowHandler()
    payload = np.random.randn(4, 8, 2).astype(np.float32)
    h.handle(
        SimpleNamespace(
            kind="trajectories",
            scope="global",
            payload=payload,
            step=2,
            extra={"name": "paths"},
            time=None,
        )
    )
    render_mock.assert_called_once()
    np.testing.assert_array_equal(render_mock.call_args[0][0], payload)
    assert mock_client.log_image.call_args.kwargs["key"] == "paths"


def test_handle_trajectories_bad_payload_warns(mock_client, monkeypatch):
    def _raise(_value):
        raise ValueError("bad shape")

    monkeypatch.setattr(
        mlflow_module, "create_numpy_trajectories_visualization", _raise
    )
    h = MLflowHandler()
    messages, collector = _capture_logger_messages(h._logger)
    try:
        h.handle(
            SimpleNamespace(
                kind="trajectories",
                scope="global",
                payload=np.zeros((3, 4), dtype=np.float32),
                step=0,
                extra={},
                time=None,
            )
        )
    finally:
        h._logger.removeHandler(collector)
    assert any("trajectories" in m.lower() for m in messages)
    mock_client.log_image.assert_not_called()


def test_handle_histogram_logs_figure(mock_client):
    h = MLflowHandler()
    h.handle(
        SimpleNamespace(
            kind="histogram",
            scope="global",
            payload=np.random.randn(256),
            step=4,
            extra={"name": "weights"},
            time=None,
        )
    )
    mock_client.log_figure.assert_called_once()
    assert (
        mock_client.log_figure.call_args.kwargs["artifact_file"]
        == "histograms/weights/step_4.png"
    )


def test_handle_video_logs_artifact(mock_client, monkeypatch):
    gif_mock = MagicMock()
    monkeypatch.setattr(mlflow_module, "save_numpy_gif", gif_mock)
    h = MLflowHandler()
    video = np.zeros((4, 8, 8, 3), dtype=np.uint8)
    h.handle(
        SimpleNamespace(
            kind="video",
            scope="global",
            payload=video,
            step=9,
            extra={"name": "rollout", "fps": 10, "format": "gif"},
            time=None,
        )
    )
    gif_mock.assert_called_once()
    assert gif_mock.call_args.kwargs["fps"] == 10
    args, kwargs = mock_client.log_artifact.call_args
    assert args[0] == "run-0"
    assert kwargs["artifact_path"] == "videos/rollout"


def test_handle_unsupported_kind_warns(mock_client):
    h = MLflowHandler()
    messages, collector = _capture_logger_messages(h._logger)
    try:
        h.handle(make_event(kind="nonsense", payload={}))
    finally:
        h._logger.removeHandler(collector)
    assert any("unsupported" in m.lower() for m in messages)


# -------------------------------------------------------------------------
# Monotonic-step contract
# -------------------------------------------------------------------------


def test_handle_drops_backward_step_with_warning(mock_client):
    h = MLflowHandler()
    messages, collector = _capture_logger_messages(h._logger)
    try:
        h.handle(make_event(kind="metric", payload={"loss": 1.0}, step=10))
        h.handle(make_event(kind="metric", payload={"loss": 0.9}, step=5))
    finally:
        h._logger.removeHandler(collector)
    metric_calls = [
        c
        for c in mock_client.log_batch.call_args_list
        if c.kwargs.get("metrics")
    ]
    assert len(metric_calls) == 1, "backward-step event must not be logged"
    assert any("out-of-order" in m.lower() and "step=5" in m for m in messages)


def test_handle_tracks_step_per_scope(mock_client):
    mock_client.create_run.side_effect = [
        SimpleNamespace(info=SimpleNamespace(run_id="run-train")),
        SimpleNamespace(info=SimpleNamespace(run_id="run-eval")),
    ]
    h = MLflowHandler()
    h.handle(
        make_event(kind="metric", payload={"x": 1}, scope="train", step=10)
    )
    # Lower step on a different scope must still be forwarded.
    h.handle(make_event(kind="metric", payload={"x": 2}, scope="eval", step=1))
    metric_calls = [
        c
        for c in mock_client.log_batch.call_args_list
        if c.kwargs.get("metrics")
    ]
    assert len(metric_calls) == 2


def test_handle_step_none_bypasses_guard(mock_client):
    h = MLflowHandler()
    h.handle(make_event(kind="metric", payload={"a": 1}, step=10))
    h.handle(make_event(kind="metric", payload={"b": 2}, step=None))
    metric_calls = [
        c
        for c in mock_client.log_batch.call_args_list
        if c.kwargs.get("metrics")
    ]
    assert len(metric_calls) == 2


def test_handle_artifact_bypasses_step_guard(mock_client, tmp_path):
    artifact_file = tmp_path / "a.npy"
    artifact_file.write_bytes(b"x")
    h = MLflowHandler()
    h.handle(make_event(kind="metric", payload={"loss": 1.0}, step=100))
    h.handle(
        SimpleNamespace(
            kind="artifact",
            scope="global",
            payload={"path": str(artifact_file), "name": "a"},
            step=0,  # a regression for non-artifact events
            extra={},
            time=None,
        )
    )
    mock_client.log_artifact.assert_called_once()


def test_close_terminates_runs(mock_client):
    mock_client.create_run.side_effect = [
        SimpleNamespace(info=SimpleNamespace(run_id="run-a")),
        SimpleNamespace(info=SimpleNamespace(run_id="run-b")),
    ]
    h = MLflowHandler()
    h.handle(make_event(kind="metric", payload={"x": 1}, scope="a", step=0))
    h.handle(make_event(kind="metric", payload={"x": 1}, scope="b", step=0))
    h.close()
    terminated = {c.args[0] for c in mock_client.set_terminated.call_args_list}
    assert terminated == {"run-a", "run-b"}
    assert h._runs == {}


def test_to_dict_from_dict_roundtrip(mock_client):
    h = MLflowHandler(
        experiment="exp",
        tracking_uri="http://srv:5000",
        run_name="base",
        params={"lr": 0.1},
        tags={"team": "rl"},
        name="mlflow-custom",
    )
    data = h.to_dict()
    assert data["cls"] == "MLflowHandler"
    assert data["data"]["experiment"] == "exp"
    assert data["data"]["tracking_uri"] == "http://srv:5000"

    restored = MLflowHandler.from_dict(data["data"])
    assert restored._experiment == "exp"
    assert restored._tracking_uri == "http://srv:5000"
    assert restored._base_run_name == "base"
    assert restored._params == {"lr": 0.1}
    assert restored._tags == {"team": "rl"}
    assert restored.name == "mlflow-custom"


# -------------------------------------------------------------------------
# Lazy export wiring (goggles.MLflowHandler resolves without eager import)
# -------------------------------------------------------------------------


def test_lazy_export_resolves_to_handler():
    assert gg.MLflowHandler is MLflowHandler
    assert "MLflowHandler" in gg.__all__


def test_get_handler_class_resolves_mlflow_lazily():
    # Deserialization path used by the bus/host.
    assert gg._get_handler_class("MLflowHandler") is MLflowHandler


def test_module_getattr_unknown_attr_raises():
    with pytest.raises(AttributeError):
        gg.__getattr__("DefinitelyNotAHandler")


def test_integrations_getattr_resolves_and_rejects_unknown():
    assert integ.MLflowHandler is MLflowHandler
    with pytest.raises(AttributeError):
        integ.__getattr__("Nope")


# -------------------------------------------------------------------------
# Video format branches
# -------------------------------------------------------------------------


def test_handle_video_mp4_uses_mp4_encoder(mock_client, monkeypatch):
    mp4_mock = MagicMock()
    monkeypatch.setattr(mlflow_module, "save_numpy_mp4", mp4_mock)
    h = MLflowHandler()
    h.handle(
        SimpleNamespace(
            kind="video",
            scope="global",
            payload=np.zeros((4, 8, 8, 3), dtype=np.uint8),
            step=1,
            extra={"name": "clip", "format": "mp4"},
            time=None,
        )
    )
    mp4_mock.assert_called_once()
    mock_client.log_artifact.assert_called_once()


def test_handle_video_unknown_format_warns_and_defaults_to_gif(
    mock_client, monkeypatch
):
    gif_mock = MagicMock()
    monkeypatch.setattr(mlflow_module, "save_numpy_gif", gif_mock)
    h = MLflowHandler()
    messages, collector = _capture_logger_messages(h._logger)
    try:
        h.handle(
            SimpleNamespace(
                kind="video",
                scope="global",
                payload=np.zeros((4, 8, 8, 3), dtype=np.uint8),
                step=1,
                extra={"name": "clip", "format": "avi"},
                time=None,
            )
        )
    finally:
        h._logger.removeHandler(collector)
    assert any("unsupported video format" in m.lower() for m in messages)
    gif_mock.assert_called_once()  # fell back to gif


def test_handle_video_none_payload_warns(mock_client):
    h = MLflowHandler()
    messages, collector = _capture_logger_messages(h._logger)
    try:
        h.handle(
            SimpleNamespace(
                kind="video",
                scope="global",
                payload=None,
                step=1,
                extra={"name": "clip"},
                time=None,
            )
        )
    finally:
        h._logger.removeHandler(collector)
    assert any("none payload" in m.lower() for m in messages)
    mock_client.log_artifact.assert_not_called()


def test_handle_image_none_payload_warns(mock_client):
    h = MLflowHandler()
    messages, collector = _capture_logger_messages(h._logger)
    try:
        h.handle(
            SimpleNamespace(
                kind="image",
                scope="global",
                payload={"cam": None},
                step=1,
                extra={},
                time=None,
            )
        )
    finally:
        h._logger.removeHandler(collector)
    assert any("none payload" in m.lower() for m in messages)
    mock_client.log_image.assert_not_called()


# -------------------------------------------------------------------------
# Video shape normalization (_video_to_channels_last)
# -------------------------------------------------------------------------


def test_video_to_channels_last_3d_passthrough():
    out = _video_to_channels_last(np.zeros((4, 8, 12), dtype=np.uint8))
    assert out.shape == (4, 8, 12)


@pytest.mark.parametrize("c", [1, 3, 4])
def test_video_to_channels_last_channels_first_moveaxis(c):
    # (F, C, H, W) -> (F, H, W, C); H/W not in {1,3,4} so axis -1 is not
    # mistaken for the channel axis.
    F, H, W = 5, 8, 12
    arr = np.arange(F * c * H * W, dtype=np.uint8).reshape(F, c, H, W)
    out = _video_to_channels_last(arr)
    assert out.shape == (F, H, W, c)
    # A real transpose, not a relabeled view.
    assert np.array_equal(out, np.moveaxis(arr, 1, -1))


@pytest.mark.parametrize("c", [1, 3, 4])
def test_video_to_channels_last_channels_last_passthrough(c):
    arr = np.zeros((5, 8, 12, c), dtype=np.uint8)
    out = _video_to_channels_last(arr)
    assert out.shape == (5, 8, 12, c)


def test_video_to_channels_last_5d_collapses():
    # (B, F, C, H, W) -> leading axes collapse, then channels-last.
    out = _video_to_channels_last(np.zeros((2, 4, 3, 8, 12), dtype=np.uint8))
    assert out.shape == (8, 8, 12, 3)


def test_video_to_channels_last_4d_no_channel_axis_raises():
    with pytest.raises(ValueError, match="4D video"):
        _video_to_channels_last(np.zeros((5, 7, 8, 9), dtype=np.uint8))


@pytest.mark.parametrize("shape", [(8, 12), (2, 3, 4, 5, 6, 7)])
def test_video_to_channels_last_bad_rank_raises(shape):
    with pytest.raises(ValueError, match="Video has shape"):
        _video_to_channels_last(np.zeros(shape, dtype=np.uint8))


# -------------------------------------------------------------------------
# Histogram edge cases and close() resilience
# -------------------------------------------------------------------------


def test_handle_histogram_invalid_payload_warns(mock_client):
    h = MLflowHandler()
    messages, collector = _capture_logger_messages(h._logger)
    try:
        h.handle(
            SimpleNamespace(
                kind="histogram",
                scope="global",
                payload=5,  # not a sequence/array
                step=1,
                extra={"name": "w"},
                time=None,
            )
        )
    finally:
        h._logger.removeHandler(collector)
    assert any("must be a sequence or array" in m.lower() for m in messages)
    mock_client.log_figure.assert_not_called()


def test_handle_histogram_unparseable_payload_warns(mock_client):
    # A str passes the isinstance(Sequence) guard but is not numeric, so
    # building the figure raises -> warn + return (mirrors W&B).
    h = MLflowHandler()
    messages, collector = _capture_logger_messages(h._logger)
    try:
        h.handle(
            SimpleNamespace(
                kind="histogram",
                scope="global",
                payload="not-numbers",
                step=1,
                extra={"name": "w"},
                time=None,
            )
        )
    finally:
        h._logger.removeHandler(collector)
    assert any("invalid histogram payload" in m.lower() for m in messages)
    mock_client.log_figure.assert_not_called()


def test_handle_histogram_step_none_naming(mock_client):
    h = MLflowHandler()
    h.handle(
        SimpleNamespace(
            kind="histogram",
            scope="global",
            payload=np.random.randn(64),
            step=None,
            extra={"name": "weights"},
            time=None,
        )
    )
    assert (
        mock_client.log_figure.call_args.kwargs["artifact_file"]
        == "histograms/weights.png"
    )


def test_close_swallows_set_terminated_error(mock_client):
    mock_client.set_terminated.side_effect = RuntimeError("boom")
    h = MLflowHandler()
    h.handle(make_event(kind="metric", payload={"x": 1}, scope="a", step=0))
    messages, collector = _capture_logger_messages(h._logger)
    try:
        h.close()  # must not raise
    finally:
        h._logger.removeHandler(collector)
    assert any("failed to finish" in m.lower() for m in messages)
    assert h._runs == {}, "runs are cleared even when termination fails"


def test_close_before_open_is_noop(mock_client):
    h = MLflowHandler()
    h.close()  # client was never created
    mock_client.set_terminated.assert_not_called()
    assert h._runs == {}


# -------------------------------------------------------------------------
# End-to-end smoke test against a real local MLflow file store.
# -------------------------------------------------------------------------


@pytest.mark.slow
def test_end_to_end_local_store(tmp_path):
    """Drive the real MlflowClient against a local sqlite store.

    Verifies the API contract (run creation, metric history, artifact
    upload) the unit tests mock out. Uses a sqlite backend because
    MLflow 3 rejects the bare file store by default.

    Args:
        tmp_path: Per-test temporary directory used as the tracking store.
    """
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    artifact_location = (tmp_path / "artifacts").as_uri()
    artifact = tmp_path / "weights.npy"
    np.save(artifact, np.zeros((4, 4)))

    h = MLflowHandler(
        experiment="smoke",
        tracking_uri=uri,
        artifact_location=artifact_location,
        run_name="t",
    )
    h.open()
    for step in range(3):
        h.handle(
            make_event(
                kind="metric", payload={"loss": 1.0 / (step + 1)}, step=step
            )
        )
    h.handle(
        SimpleNamespace(
            kind="image",
            scope="global",
            payload=np.zeros((8, 8, 3), dtype=np.uint8),
            step=2,
            extra={"name": "frame"},
            time=None,
        )
    )
    h.handle(
        SimpleNamespace(
            kind="artifact",
            scope="global",
            payload={"path": str(artifact), "name": "ckpt"},
            step=2,
            extra={},
            time=None,
        )
    )
    h.close()

    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name("smoke")
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    run_id = runs[0].info.run_id
    history = client.get_metric_history(run_id, "loss")
    assert len(history) == 3
    artifact_paths = {a.path for a in client.list_artifacts(run_id)}
    assert "ckpt" in artifact_paths
