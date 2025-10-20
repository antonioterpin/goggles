import sys
import builtins
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
    """W&B init/teardown via integrations; no persistent artifacts."""
    fake_run = SimpleNamespace(
        id="abc123",
        url="https://wandb.test/run/abc123",
        project="demo_project",
        entity="test_user",
        finish=MagicMock(),
    )

    # Patch wandb inside the integrations module (where it's imported/used)
    with patch.dict(sys.modules, {"wandb": MagicMock()}) as mods:
        wandb_mock = mods["wandb"]
        wandb_mock.init.return_value = fake_run
        wandb_mock.Settings = lambda **_: None

        # (optional) also ensure defaults are off, and pass per-run override
        ctx_mgr = _RunContextManager(
            name="wandb-test",
            log_dir=str(tmp_run_dir),
            user_metadata={"project": "demo_project"},
            enable_wandb=enable,  # per-run override routed to integrations
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


def test_configure_defaults_enable_then_run_override_disable(tmp_run_dir, monkeypatch):
    # enable via configure
    import importlib

    integ = importlib.import_module("goggles._core.config")
    integ.configure(enable_wandb=True)

    fake_run = SimpleNamespace(
        id="abc123",
        url="https://wandb.test/run/abc123",
        project="demo_project",
        entity="test_user",
        finish=MagicMock(),
    )

    with patch.dict(sys.modules, {"wandb": MagicMock()}) as mods:
        wandb_mock = mods["wandb"]
        wandb_mock.init.return_value = fake_run
        wandb_mock.Settings = lambda **_: None

        # Per-run override should win and prevent init
        ctx_mgr = _RunContextManager(
            name="override-disable",
            log_dir=str(tmp_run_dir),
            user_metadata={"project": "demo_project"},
            enable_wandb=False,  # override should disable even if defaults enable
        )
        with ctx_mgr as ctx:
            assert ctx.wandb is None
        wandb_mock.init.assert_not_called()
        fake_run.finish.assert_not_called()


def test_configure_enables_when_run_override_is_none(tmp_run_dir):
    import importlib

    integ = importlib.import_module("goggles._core.config")
    integ.configure(enable_wandb=True)

    fake_run = SimpleNamespace(
        id="abc123",
        url="https://wandb.test/run/abc123",
        project="demo_project",
        entity="test_user",
        finish=MagicMock(),
    )

    with patch.dict(sys.modules, {"wandb": MagicMock()}) as mods:
        wandb_mock = mods["wandb"]
        wandb_mock.init.return_value = fake_run
        wandb_mock.Settings = lambda **_: None

        # Pass enable_wandb=None to simulate "no per-run opinion"
        ctx_mgr = _RunContextManager(
            name="default-enable",
            log_dir=str(tmp_run_dir),
            user_metadata={"project": "demo_project"},
            enable_wandb=None,
        )
        with ctx_mgr as ctx:
            assert ctx.wandb is not None and ctx.wandb["id"] == "abc123"
        wandb_mock.init.assert_called_once()
        fake_run.finish.assert_called_once()


def test_metadata_contains_wandb_when_enabled(tmp_run_dir):
    fake_run = SimpleNamespace(
        id="abc123",
        url="https://wandb.test/run/abc123",
        project="demo_project",
        entity="test_user",
        finish=MagicMock(),
    )

    with patch.dict(sys.modules, {"wandb": MagicMock()}) as mods:
        wandb_mock = mods["wandb"]
        wandb_mock.init.return_value = fake_run
        wandb_mock.Settings = lambda **_: None

        ctx_mgr = _RunContextManager(
            name="meta-wandb",
            log_dir=str(tmp_run_dir),
            user_metadata={"project": "demo_project"},
            enable_wandb=True,
        )
        with ctx_mgr as ctx:
            meta_path = Path(ctx.log_dir) / "metadata.json"
            data = json.loads(meta_path.read_text())
            assert data["wandb"]["id"] == "abc123"


def test_wandb_missing_module_is_graceful(tmp_run_dir, monkeypatch):
    # Ensure it's not already cached
    sys.modules.pop("wandb", None)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "wandb":
            raise ImportError("No module named 'wandb'")
        return real_import(name, *args, **kwargs)

    # Make any attempt to `import wandb` fail inside this test
    monkeypatch.setattr(builtins, "__import__", fake_import)

    ctx_mgr = _RunContextManager(
        name="no-wandb",
        log_dir=str(tmp_run_dir),
        user_metadata={"project": "demo_project"},
        enable_wandb=True,  # requested, but import will fail
    )
    with ctx_mgr as ctx:
        # Should not crash; wandb field should be None
        assert ctx.wandb is None
