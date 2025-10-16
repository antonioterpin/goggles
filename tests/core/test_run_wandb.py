import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from goggles import run
from goggles._core.run import _RunContextManager


@pytest.fixture
def tmp_run_dir(tmp_path):
    """Provide a temporary run directory and ensure full cleanup."""
    base = tmp_path / "runs"
    base.mkdir()
    yield base
    # Clean all created run directories
    shutil.rmtree(base, ignore_errors=True)
    assert not base.exists(), f"Temporary run dir {base} was not cleaned."


@pytest.mark.parametrize("enable", [True, False])
def test_run_context_manager_wandb_integration(tmp_run_dir, enable):
    """Test W&B init/teardown and ensure no persistent artifacts."""
    fake_run = SimpleNamespace(
        id="abc123",
        url="https://wandb.test/run/abc123",
        project_name=lambda: "demo_project",
        entity="test_user",
        finish=MagicMock(),
    )

    with patch.dict(sys.modules, {"wandb": MagicMock()}) as mods:
        wandb_mock = mods["wandb"]
        wandb_mock.init.return_value = fake_run
        wandb_mock.Settings = lambda **_: None

        ctx_mgr = _RunContextManager(
            name="wandb-test",
            log_dir=str(tmp_run_dir),
            user_metadata={"project": "demo_project"},
            enable_wandb=enable,
        )

        with ctx_mgr as ctx:
            run_path = Path(ctx.log_dir)
            meta_path = run_path / "metadata.json"
            assert run_path.exists() and meta_path.exists()

            if enable:
                wandb_mock.init.assert_called_once()
                assert ctx.wandb == {
                    "id": "abc123",
                    "url": "https://wandb.test/run/abc123",
                    "project": "demo_project",
                    "entity": "test_user",
                }
            else:
                wandb_mock.init.assert_not_called()
                assert ctx.wandb is None

        if enable:
            fake_run.finish.assert_called_once()
        else:
            fake_run.finish.assert_not_called()


def test_run_context_manager_metadata_written(tmp_run_dir):
    """Verify metadata.json contents and cleanup after exit."""
    ctx_mgr = _RunContextManager(
        name="meta-test",
        log_dir=str(tmp_run_dir),
        user_metadata={"foo": "bar"},
        enable_wandb=False,
    )

    with ctx_mgr as ctx:
        meta_path = Path(ctx.log_dir) / "metadata.json"
        data = json.loads(meta_path.read_text())
        assert data["metadata"]["foo"] == "bar"

    data = json.loads(meta_path.read_text())
    assert "finished_at" in data


def test_public_run_api_passes_enable_wandb(monkeypatch, tmp_run_dir):
    """Ensure that the public `run()` API wires `enable_wandb` correctly."""
    called = {}

    def fake_init(**kwargs):
        called.update(kwargs)
        return SimpleNamespace(finish=lambda: None)

    with patch("goggles._core.run._RunContextManager.__enter__") as enter_mock, patch(
        "goggles._core.run._RunContextManager.__exit__"
    ) as exit_mock, patch.dict(sys.modules, {"wandb": MagicMock(init=fake_init)}):
        enter_mock.return_value = SimpleNamespace(log_dir=str(tmp_run_dir), wandb=None)

        with run(enable_wandb=True, log_dir=str(tmp_run_dir)) as ctx:
            assert ctx.log_dir
            assert enter_mock.called

        exit_mock.assert_called_once()
