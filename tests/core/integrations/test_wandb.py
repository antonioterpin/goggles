from pathlib import Path
from unittest.mock import MagicMock, patch
import importlib
import sys
import os
import time
from unittest.mock import MagicMock
import wandb
import pytest


import goggles._core.integrations.wandb as wandb_module
from goggles._core.integrations.wandb import WandBHandler


@pytest.fixture
def mock_wandb(monkeypatch):
    """Patch the wandb module used inside WandBHandler."""
    mock = MagicMock()
    monkeypatch.setattr(wandb_module, "wandb", mock)
    return mock


@pytest.mark.parametrize(
    "reinit", [None, "finish_previous", "return_previous", "create_new"]
)
def test_open_initializes_wandb_run(mock_wandb, reinit):
    handler = WandBHandler(project="proj", entity="ent", run_name="run", reinit=reinit)
    handler.open()
    assert mock_wandb.init.called
    call_args = mock_wandb.init.call_args.kwargs
    assert call_args["project"] == "proj"
    assert call_args["entity"] == "ent"
    assert call_args["name"] == "run"
    if reinit is not None:
        assert call_args["reinit"] == reinit
    else:
        assert "reinit" not in call_args


def test_open_twice_does_not_reinit(mock_wandb):
    handler = WandBHandler()
    handler._wandb_run = MagicMock()  # Simulate already opened
    handler.open()
    mock_wandb.init.assert_not_called()


def test_close_finishes_run(mock_wandb):
    handler = WandBHandler()
    handler._wandb_run = MagicMock()
    handler.close()
    mock_wandb.finish.assert_called_once()
    assert handler._wandb_run is None


def test_close_without_run_does_nothing(mock_wandb):
    handler = WandBHandler()
    handler.close()
    mock_wandb.finish.assert_not_called()


@pytest.mark.parametrize("kind,expected", [("metric", True), ("log", False)])
def test_can_handle_returns_expected(kind, expected):
    handler = WandBHandler()
    assert handler.can_handle(kind) == expected


def test_handle_logs_metrics_with_step(mock_wandb):
    handler = WandBHandler()
    handler._wandb_run = MagicMock()
    event = MagicMock(payload={"loss": 1.23}, step=10)
    handler.handle(event)
    mock_wandb.log.assert_called_once_with({"loss": 1.23}, step=10)


def test_handle_logs_metrics_without_step(mock_wandb):
    handler = WandBHandler()
    handler._wandb_run = MagicMock()
    event = MagicMock(payload={"acc": 0.9}, step=None)
    handler.handle(event)

    # Accept both call styles: with or without explicit step=None
    call_args = mock_wandb.log.call_args
    assert call_args.args[0] == {"acc": 0.9}
    if "step" in call_args.kwargs:
        assert call_args.kwargs["step"] is None


def test_handle_warns_if_not_opened(mock_wandb):
    handler = WandBHandler()
    event = MagicMock(payload={"loss": 1.0})

    with patch.object(handler._logger, "warning") as mock_warn:
        handler.handle(event)

    mock_warn.assert_called_once()
    args, _ = mock_warn.call_args
    assert "not opened" in args[0] or "ignoring" in args[0]
    mock_wandb.log.assert_not_called()


def test_handle_raises_on_invalid_payload(mock_wandb):
    handler = WandBHandler()
    handler._wandb_run = MagicMock()
    event = MagicMock(payload=["not", "mapping"])
    with pytest.raises(ValueError, match="Metric event payload must be a mapping"):
        handler.handle(event)


@pytest.mark.parametrize("invalid", ["yes", "no", 123, object()])
def test_invalid_reinit_raises_valueerror(invalid):
    with pytest.raises(ValueError, match="Invalid reinit value"):
        WandBHandler(reinit=invalid)


def test_handle_artifact_uploads_file(mock_wandb, tmp_path):
    """Ensure artifact events create and upload a wandb.Artifact."""
    handler = WandBHandler()
    handler._wandb_run = MagicMock()
    mock_run = handler._wandb_run

    # Create dummy file
    dummy_file = tmp_path / "artifact.txt"
    dummy_file.write_text("content")

    event = MagicMock(
        kind="artifact",
        payload={"path": str(dummy_file), "name": "artifact_test", "type": "data"},
    )

    handler.handle(event)

    # Verify artifact creation and upload
    mock_wandb.Artifact.assert_called_once_with(name="artifact_test", type="data")
    artifact_obj = mock_wandb.Artifact.return_value
    artifact_obj.add_file.assert_called_once_with(str(dummy_file))
    mock_run.log_artifact.assert_called_once_with(artifact_obj)


def test_handle_artifact_missing_path_warns(mock_wandb):
    """Warn if artifact event payload lacks a valid path."""
    handler = WandBHandler()
    handler._wandb_run = MagicMock()

    bad_payloads = [
        {"name": "test_art", "type": "misc"},  # missing path
        {"path": 123},  # invalid path type
        "not_a_mapping",  # not a dict
    ]

    for payload in bad_payloads:
        event = MagicMock(kind="artifact", payload=payload)
        with patch.object(handler._logger, "warning") as mock_warn:
            handler.handle(event)
        mock_warn.assert_called()
        handler._wandb_run.log_artifact.assert_not_called()


def test_handle_artifact_default_fields(mock_wandb, tmp_path):
    """Use default name and type when not provided."""
    handler = WandBHandler()
    handler._wandb_run = MagicMock()
    mock_run = handler._wandb_run

    dummy_file = tmp_path / "artifact_default.txt"
    dummy_file.write_text("abc")

    event = MagicMock(kind="artifact", payload={"path": str(dummy_file)})

    handler.handle(event)

    mock_wandb.Artifact.assert_called_once_with(name="artifact", type="misc")
    artifact_obj = mock_wandb.Artifact.return_value
    artifact_obj.add_file.assert_called_once_with(str(dummy_file))
    mock_run.log_artifact.assert_called_once_with(artifact_obj)


def test_handle_unsupported_kind_warns(mock_wandb):
    """Emit a warning for unsupported event kinds."""
    handler = WandBHandler()
    handler._wandb_run = MagicMock()
    event = MagicMock(kind="unknown", payload={"something": 1})
    with patch.object(handler._logger, "warning") as mock_warn:
        handler.handle(event)
    mock_warn.assert_called_once()
    args, _ = mock_warn.call_args
    assert "Unsupported" in args[0]


def test_offline_integration_real_wandb(tmp_path):
    # reload real module (unchanged)
    if "goggles._core.integrations.wandb" in sys.modules:
        del sys.modules["goggles._core.integrations.wandb"]
    import goggles._core.integrations.wandb as wandb_module

    importlib.reload(wandb)
    importlib.reload(wandb_module)
    from goggles._core.integrations.wandb import WandBHandler

    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_DIR"] = str(tmp_path)
    os.environ["WANDB_DISABLE_CODE"] = "true"

    handler = WandBHandler(project="integration-test", run_name="offline-real")
    handler.open()
    run = handler._wandb_run
    assert run is not None
    assert run.settings.mode == "offline"

    # Log a few steps
    for step in range(3):
        event = MagicMock(payload={"metric": float(step)}, step=step)
        handler.handle(event)

    # >>> Assert summary BEFORE finishing (summary proxy is valid now)
    # Convert to a plain dict instead of `"metric" in run.summary`
    if hasattr(run.summary, "_as_dict"):
        summary_dict = run.summary._as_dict()
    else:
        summary_dict = dict(run.summary)  # type: ignore[arg-type]

    assert "metric" in summary_dict, "Expected 'metric' in W&B summary (pre-finish)."
    val = float(summary_dict["metric"])
    # W&B may keep the last committed row or a recent row; accept any logged value
    assert val in {0.0, 1.0, 2.0}, f"Unexpected summary value: {val}"

    # Now finish and teardown
    handler.close()
    if hasattr(wandb, "teardown"):
        wandb.teardown()

    # Post-finish: assert directory layout only (avoid reading in-memory summary/file races)
    files_dir = Path(run.dir)  # points to .../offline-run-.../files
    assert files_dir.exists(), f"Run files directory {files_dir} does not exist."
    run_root = files_dir.parent
    assert run_root.name.startswith("offline-run-"), f"Unexpected run root: {run_root}"
    logs_dir = run_root / "logs"
    assert logs_dir.exists(), f"Expected logs/ directory at {logs_dir}"

    # Optional, non-flaky presence checks (no contents assertions)
    # These filenames are stable in offline mode but may flush slightly later across versions.
    # So we only check existence if present; we don't fail the test on absence.
    _maybe_summary = files_dir / "wandb-summary.json"
    _maybe_history = run_root / "logs" / "debug-internal.log"
    # No hard assert here; purely informative if you want:
    # print(_maybe_summary.exists(), _maybe_history.exists())
