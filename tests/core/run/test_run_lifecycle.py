import json
import logging
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
import warnings

import pytest

from goggles import run as public_run
from goggles._core.run import _RunContextManager
from goggles._core import config as cfg


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


# --- WandB fakes (used across a couple of tests) ---


class _FakeArtifact:
    def __init__(self, name, type, metadata=None):
        self.name = name
        self.type = type
        self.metadata = metadata or {}
        self._files = []

    def add_file(self, path):
        self._files.append(str(path))


class _FakeWandbRun:
    def __init__(self, project="proj"):
        self._events = []  # list of tuples: ("log_artifact"| "finish", ...)
        self._artifacts = []
        self._project = project

    def project_name(self):
        return self._project

    @property
    def entity(self):  # unused but safe
        return "ent"

    @property
    def id(self):
        return "fake123"

    @property
    def url(self):
        return "https://wandb.test/run/fake123"

    def log_artifact(self, artifact):
        # record files so we can assert later
        self._events.append(
            ("log_artifact", artifact.name, artifact.type, list(artifact._files))
        )
        self._artifacts.append(artifact)

    def finish(self):
        self._events.append(("finish", None, None, None))


def _install_fake_wandb(monkeypatch):
    fake = MagicMock()

    def init(**kwargs):
        run = _FakeWandbRun(project=kwargs.get("project", "proj"))
        fake._last_run = run
        return run

    fake.init.side_effect = init
    fake.Settings = lambda **_: None
    fake.Artifact.side_effect = lambda name, type, metadata=None: _FakeArtifact(
        name, type, metadata
    )
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake


# ------------------------------------------------------------------------------------
# 1) Full lifecycle matrix: defaults via configure + per-run overrides + sinks + wandb
# ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "defaults, overrides, expect_files",
    [
        # A) Defaults prefer JSONL; run overrides add file log too
        (
            {"enable_jsonl": True, "enable_file": False},
            {"enable_file": True, "enable_console": False, "enable_wandb": False},
            {"metadata.json", "events.jsonl", "events.log"},
        ),
        # B) Console only by overrides; ensure no files
        (
            {"enable_file": False, "enable_jsonl": False, "enable_console": True},
            {
                "enable_console": True,
                "enable_file": False,
                "enable_jsonl": False,
                "enable_wandb": False,
            },
            {"metadata.json"},
        ),
        # C) File + JSONL + W&B artifacts on; verify upload happens
        (
            {"enable_file": True, "enable_jsonl": True},
            {"enable_wandb": True, "enable_artifacts": True, "enable_console": False},
            {"metadata.json", "events.jsonl", "events.log"},
        ),
    ],
)
def test_full_lifecycle_configs(
    tmp_run_dir, monkeypatch, defaults, overrides, expect_files
):
    # apply defaults
    cfg.configure(**defaults)

    # optionally install fake wandb if needed
    if overrides.get("enable_wandb"):
        _install_fake_wandb(monkeypatch)

    with public_run(name="matrix", log_dir=str(tmp_run_dir), **overrides) as ctx:
        run_path = Path(ctx.log_dir)
        # emit something so sinks create files
        logging.getLogger().info("hello")
        _flush_handlers()

        meta_path = run_path / "metadata.json"
        assert meta_path.exists()
        for fname in expect_files:
            assert (run_path / fname).exists()

    # if wandb was enabled, ensure artifact upload happened before finish
    if overrides.get("enable_wandb"):
        wb = sys.modules["wandb"]
        wb_run = getattr(wb, "_last_run", None)
        assert wb_run is not None
        kinds = [e[0] for e in wb_run._events]
        assert "log_artifact" in kinds and "finish" in kinds
        assert kinds.index("log_artifact") < kinds.index("finish")


# -----------------------------------------------------
# 2) Fail fast: nested runs should raise RuntimeError
# -----------------------------------------------------


def test_nested_run_fails_fast(tmp_run_dir):
    with public_run(name="outer", log_dir=str(tmp_run_dir)) as _:
        with pytest.raises(RuntimeError):
            with public_run(name="inner", log_dir=str(tmp_run_dir)):
                pass


# -------------------------------------------------------------------
# 3) Fail fast: unknown configure key should raise ValueError (API)
# -------------------------------------------------------------------


def test_configure_unknown_key_raises_value_error():
    with pytest.raises(ValueError):
        cfg.configure(does_not_exist=True)


# ----------------------------------------------------------------------
# 4) Artifacts enabled but W&B disabled -> should no-op (no exceptions)
# ----------------------------------------------------------------------


def test_artifacts_enabled_without_wandb_is_noop(tmp_run_dir):
    with public_run(
        name="artifacts-no-wandb",
        log_dir=str(tmp_run_dir),
        enable_wandb=False,
        enable_artifacts=True,
        enable_file=True,
        enable_jsonl=True,
        enable_console=False,
    ) as ctx:
        run_path = Path(ctx.log_dir)
        logging.getLogger().warning("line")
        _flush_handlers()
        # files still present
        assert (run_path / "metadata.json").exists()
        assert (run_path / "events.log").exists()
        assert (run_path / "events.jsonl").exists()
    # nothing to assert on wandb: absence is the point


# ----------------------------------------------------------------------
# 5) Reset root + capture warnings + propagate cooperate end-to-end
# ----------------------------------------------------------------------


def test_reset_root_capture_propagate_cooperate(tmp_run_dir):
    # pre-install a handler, ensure it's cleared during run and restored after
    root = logging.getLogger()
    saved_records = []

    class _MemHandler(logging.Handler):
        def emit(self, record):
            saved_records.append(record)

    mem = _MemHandler()
    root.addHandler(mem)

    prev_propagate = root.propagate

    with public_run(
        name="flags-cooperate",
        log_dir=str(tmp_run_dir),
        reset_root=True,
        capture_warnings=True,
        propagate=False,  # should override previous state during run
        enable_file=True,
        enable_console=False,
    ) as ctx:
        # dummy handler removed
        assert mem not in logging.getLogger().handlers
        # warnings should be logged to file
        warnings.warn("warn-me", UserWarning)
        _flush_handlers()
        text = (Path(ctx.log_dir) / "events.log").read_text(encoding="utf-8")
        assert "warn-me" in text

    # handler restored after exit + propagate restored
    assert mem in logging.getLogger().handlers
    logging.getLogger().removeHandler(mem)
    assert logging.getLogger().propagate is prev_propagate


# ----------------------------------------------------------------------
# 6) Invalid log_level string shouldn’t crash; INFO fallback effective
# (matches current implementation behavior)
# ----------------------------------------------------------------------


@pytest.mark.parametrize("use_public_api", [True, False])
def test_invalid_log_level_falls_back_to_info(tmp_run_dir, use_public_api):
    # invalid level string; current impl falls back to INFO (does not raise)
    if use_public_api:
        ctx = public_run(
            name="bad-level",
            log_dir=str(tmp_run_dir),
            log_level="NOT_A_LEVEL",
            enable_file=True,
        )
    else:
        ctx = _RunContextManager(
            name="bad-level",
            log_dir=str(tmp_run_dir),
            user_metadata={},
            enable_file=True,
            log_level="NOT_A_LEVEL",
        )

    with ctx as runctx:
        p = Path(runctx.log_dir) / "events.log"
        logging.getLogger().info("visible")
        _flush_handlers()
        assert p.exists()
        assert "visible" in p.read_text(encoding="utf-8")


# ----------------------------------------------------------------------
# 7) Defaults apply when per-run override is None (or not provided)
# ----------------------------------------------------------------------


def test_defaults_apply_when_override_missing(tmp_run_dir):
    # set defaults to enable JSONL and disable file
    cfg.configure(enable_jsonl=True, enable_file=False)

    with public_run(name="defaults-apply", log_dir=str(tmp_run_dir)) as ctx:
        run_path = Path(ctx.log_dir)
        logging.getLogger().info("hello")
        _flush_handlers()
        assert (run_path / "events.jsonl").exists()
        assert not (run_path / "events.log").exists()
