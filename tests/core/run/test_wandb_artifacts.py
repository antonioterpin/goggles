import json
import logging
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from goggles import run as public_run
from goggles._core.run import _RunContextManager


@pytest.fixture
def tmp_run_dir(tmp_path):
    base = tmp_path / "runs"
    base.mkdir()
    yield base
    shutil.rmtree(base, ignore_errors=True)
    assert not base.exists(), f"Temporary run dir {base} was not cleaned."


def _flush_handlers():
    root = logging.getLogger()
    for h in root.handlers:
        try:
            h.flush()
        except Exception:
            pass


def _ctx_factory(use_public_api, *, name, log_dir, **kwargs):
    if use_public_api:
        return public_run(name=name, log_dir=str(log_dir), **kwargs)
    return _RunContextManager(
        name=name, log_dir=str(log_dir), user_metadata={}, **kwargs
    )


class FakeArtifact:
    def __init__(self, name, type, metadata=None):
        self.name = name
        self.type = type
        self.metadata = metadata or {}
        self._files = []

    def add_file(self, path):
        self._files.append(str(path))


class FakeWandbRun:
    def __init__(self, project="proj", entity="ent"):
        self._events = []  # record sequence: "log_artifact", "finish"
        self._artifacts = []
        self.project = project
        self._entity = entity

    # mimic public attrs used in your code (not strictly needed here)
    def project_name(self):
        return self.project

    @property
    def entity(self):
        return self._entity

    @property
    def id(self):
        return "fake123"

    @property
    def url(self):
        return "https://wandb.test/run/fake123"

    def log_artifact(self, artifact):
        self._events.append(
            ("log_artifact", artifact.name, artifact.type, list(artifact._files))
        )
        self._artifacts.append(artifact)

    def finish(self):
        self._events.append(("finish", None, None, None))


def _install_fake_wandb(monkeypatch):
    fake = MagicMock()

    def init(**kwargs):
        run = FakeWandbRun(project=kwargs.get("project", "proj"))
        fake._last_run = run  # <- keep a handle for tests
        return run

    fake.init.side_effect = init
    fake.Settings = lambda **_: None
    fake.Artifact.side_effect = lambda name, type, metadata=None: FakeArtifact(
        name, type, metadata
    )

    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake


# ------------------------ Tests ------------------------ #

import sys


@pytest.mark.parametrize("use_public_api", [True, False])
def test_artifacts_upload_with_text_and_jsonl(tmp_run_dir, monkeypatch, use_public_api):
    wandb_mock = _install_fake_wandb(monkeypatch)

    ctx = _ctx_factory(
        use_public_api,
        name="artifacts-all",
        log_dir=tmp_run_dir,
        enable_wandb=True,
        enable_artifacts=True,
        enable_file=True,
        enable_jsonl=True,
        enable_console=False,
        log_level="INFO",
    )
    with ctx as runctx:
        run_path = Path(runctx.log_dir)
        text_path = run_path / "events.log"
        jsonl_path = run_path / "events.jsonl"
        # trigger some logs so sinks create files
        logging.getLogger().info("to-text")
        logging.getLogger().info("to-jsonl")
        _flush_handlers()
        assert text_path.exists()
        assert jsonl_path.exists()

        # metadata exists already at enter; will be finalized on exit
        meta_path = run_path / "metadata.json"
        assert meta_path.exists()

    # Grab the last created FakeWandbRun
    wb_run = getattr(wandb_mock, "_last_run", None)
    assert wb_run is not None, "wandb.init should have been called"
    # Ensure order: log_artifact before finish
    events = wb_run._events
    assert events, "No events recorded on W&B run"
    kinds = [e[0] for e in events]
    assert "log_artifact" in kinds and "finish" in kinds
    assert kinds.index("log_artifact") < kinds.index(
        "finish"
    ), "log_artifact must happen before finish"

    # Inspect files added to artifact
    log_artifact_ev = next(e for e in events if e[0] == "log_artifact")
    _, art_name, art_type, files = log_artifact_ev
    assert art_type == "goggles-run"
    # Should include metadata.json and both sinks
    assert any(f.endswith("metadata.json") for f in files)
    assert any(f.endswith("events.log") for f in files)
    assert any(f.endswith("events.jsonl") for f in files)


@pytest.mark.parametrize("use_public_api", [True, False])
def test_artifacts_name_default_and_override(tmp_run_dir, monkeypatch, use_public_api):
    wandb_mock = _install_fake_wandb(monkeypatch)

    # Case 1: default name
    ctx = _ctx_factory(
        use_public_api,
        name="artifacts-default-name",
        log_dir=tmp_run_dir,
        enable_wandb=True,
        enable_artifacts=True,
        enable_file=True,
        enable_jsonl=False,
    )
    with ctx:
        logging.getLogger().info("x")
        _flush_handlers()

    wb_run = getattr(wandb_mock, "_last_run", None)
    assert wb_run is not None, "wandb.init should have been called"
    for e in wb_run._events:
        print(e)
        print(e[0])
    ev = next(e for e in wb_run._events if e[0] == "log_artifact")
    _, art_name, art_type, _ = ev
    assert art_type == "goggles-run"
    assert art_name == "goggles-artifacts"

    # Case 2: explicit override name & type
    ctx2 = _ctx_factory(
        use_public_api,
        name="artifacts-override-name",
        log_dir=tmp_run_dir,
        enable_wandb=True,
        enable_artifacts=True,
        enable_file=True,
        artifact_name="exp42-logs",
        artifact_type="run",
    )
    with ctx2:
        logging.getLogger().info("y")
        _flush_handlers()

    wb_run2 = getattr(wandb_mock, "_last_run", None)
    assert wb_run2 is not None, "wandb.init should have been called"
    ev2 = next(e for e in wb_run2._events if e[0] == "log_artifact")
    _, art_name2, art_type2, _ = ev2
    assert art_name2 == "exp42-logs"
    assert art_type2 == "run"


@pytest.mark.parametrize("use_public_api", [True, False])
def test_no_artifacts_when_disabled(tmp_run_dir, monkeypatch, use_public_api):
    wandb_mock = _install_fake_wandb(monkeypatch)

    ctx = _ctx_factory(
        use_public_api,
        name="artifacts-disabled",
        log_dir=tmp_run_dir,
        enable_wandb=True,
        enable_artifacts=False,  # disable upload
        enable_file=True,
    )
    with ctx:
        logging.getLogger().info("z")
        _flush_handlers()

    wb_run = getattr(wandb_mock, "_last_run", None)
    assert wb_run is not None, "wandb.init should have been called"
    # Only finish, no log_artifact
    kinds = [e[0] for e in wb_run._events]
    assert "finish" in kinds
    assert "log_artifact" not in kinds


@pytest.mark.parametrize("use_public_api", [True, False])
def test_no_wandb_no_upload(tmp_run_dir, monkeypatch, use_public_api):
    # Install a 'wandb' that is importable but ensure we never call init
    fake = MagicMock()
    fake.init.side_effect = AssertionError("wandb.init should not be called")
    fake.Settings = lambda **_: None
    fake.Artifact.side_effect = lambda *a, **k: AssertionError(
        "Artifact should not be created"
    )
    monkeypatch.setitem(sys.modules, "wandb", fake)

    ctx = _ctx_factory(
        use_public_api,
        name="no-wandb-no-upload",
        log_dir=tmp_run_dir,
        enable_wandb=False,  # W&B off
        enable_artifacts=True,  # even if artifacts are on, there is no wb handle
        enable_file=True,
    )
    with ctx:
        logging.getLogger().info("anything")
        _flush_handlers()

    # We didn't crash; implicit assertion is that wandb.init was never called
    assert True


@pytest.mark.parametrize("use_public_api", [True, False])
def test_uploaded_metadata_has_finished_at_before_upload(
    tmp_run_dir, monkeypatch, use_public_api
):
    # Ensure we upload after metadata is finalized (finished_at present).
    wandb_mock = _install_fake_wandb(monkeypatch)

    ctx = _ctx_factory(
        use_public_api,
        name="meta-finalized-before-upload",
        log_dir=tmp_run_dir,
        enable_wandb=True,
        enable_artifacts=True,
        enable_file=True,
        enable_jsonl=False,
    )
    with ctx as runctx:
        meta_path = Path(runctx.log_dir) / "metadata.json"
        # ensure file exists at enter
        assert meta_path.exists()
        logging.getLogger().info("hello")
        _flush_handlers()

    wb_run = getattr(wandb_mock, "_last_run", None)
    assert wb_run is not None, "wandb.init should have been called"
    ev = next(e for e in wb_run._events if e[0] == "log_artifact")
    _, _, _, files = ev
    meta_files = [f for f in files if f.endswith("metadata.json")]
    assert meta_files, "metadata.json must be part of the artifact"
    # Verify finished_at is in the on-disk metadata that was uploaded
    # (We can't read W&B's upload, but we can assert that the file on disk contains it.)
    meta_path = Path(files[files.index(meta_files[0])])
    data = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    assert "finished_at" in data, "metadata.json should be finalized before upload"
