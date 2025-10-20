"""Tests for goggles lifecycle management: run context, logging setup, etc."""

import importlib
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Iterator

import pytest


# ---------------------------- Helpers ---------------------------------


def _flush_handlers() -> None:
    """Flush all handlers on all known loggers (root + children)."""
    # Root
    for h in logging.getLogger().handlers:
        try:
            h.flush()
        except Exception:
            pass
    # Children
    for name in list(logging.Logger.manager.loggerDict.keys()):
        logger = logging.getLogger(name)
        if isinstance(logger, logging.Logger):
            for h in logger.handlers:
                try:
                    h.flush()
                except Exception:
                    pass


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(autouse=True)
def _clean_logging() -> Iterator[None]:
    """Ensure a clean logging environment around each test."""
    root = logging.getLogger()
    prev_handlers = list(root.handlers)
    try:
        yield
    finally:
        # Restore original handlers
        root.handlers = prev_handlers
        logging.captureWarnings(False)
        _flush_handlers()


# ------------------------- Import-time behavior ------------------------


def test_reimport_is_idempotent(monkeypatch):
    """Reloading `goggles` should not add extra handlers (no side effects)."""
    import types

    # Create a fresh module import context by deleting known entries.
    for k in list(sys.modules.keys()):
        if k == "goggles" or k.startswith("goggles."):
            del sys.modules[k]

    before = len(logging.getLogger().handlers)
    gg = importlib.import_module("goggles")
    assert hasattr(gg, "run"), "Package must re-export public API"

    # Re-import (reload) and ensure handler count unchanged.
    importlib.reload(gg)
    after = len(logging.getLogger().handlers)
    assert after == before


# --------------------------- Run context basics ------------------------


def test_run_creates_dir_and_files(tmp_path):
    import goggles as gg

    with gg.run("exp_basic", log_dir=tmp_path, enable_console=False) as ctx:
        run_dir = Path(ctx.log_dir)
        assert run_dir.exists()
        # Metadata must exist during/after run entry
        md = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        assert md["run_id"] == ctx.run_id
        assert md["run_name"] == "exp_basic"
        assert md["pid"] == ctx.pid
        assert md["host"] == ctx.host
        assert "python" in md

        # events.log should be present if enable_file defaults True
        assert (run_dir / "events.log").exists()

    # After exit, handlers should be closed; subsequent logging won’t error
    logging.getLogger("post").info("after-run-ok")


def test_nested_run_raises_runtimeerror(tmp_path):
    import goggles as gg

    with gg.run("outer", log_dir=tmp_path):
        with pytest.raises(RuntimeError):
            with gg.run("inner", log_dir=tmp_path / "inner"):
                pass


def test_active_run_state_transitions(tmp_path):
    import goggles as gg

    assert gg.current_run() is None
    with gg.run("state", log_dir=tmp_path):
        assert gg.current_run() is not None
    assert gg.current_run() is None


# --------------------------- Configure defaults ------------------------


def test_configure_overrides_defaults(tmp_path):
    import goggles as gg

    # Enable JSONL globally, then run without specifying it explicitly.
    gg.configure(enable_jsonl=True, enable_console=False)
    with gg.run("cfg-jsonl", log_dir=tmp_path) as ctx:
        run_dir = Path(ctx.log_dir)
        assert (run_dir / "events.jsonl").exists()
        # Logging something should produce at least one JSONL line.
        gg.get_logger("train").info("hello-jsonl", epoch=0)
        _flush_handlers()
        rows = _read_jsonl(run_dir / "events.jsonl")
        assert any(rec.get("msg") == "hello-jsonl" for rec in rows)


# --------------------------- BoundLogger semantics ---------------------


def test_get_logger_before_and_inside_run(tmp_path):
    import goggles as gg

    # Before run() it should be safe to create and use a logger.
    pre = gg.get_logger("pre", library_mode=True)
    pre.info("pre-boot")  # Should not crash even if no handlers yet.

    with gg.run(
        "bind-sem", log_dir=tmp_path, enable_console=False, enable_jsonl=True
    ) as ctx:
        log = gg.get_logger("worker", a=1).bind(b=2)
        log.info("bound-hello", step=10, c=3)
        _flush_handlers()
        rows = _read_jsonl(Path(ctx.log_dir) / "events.jsonl")
        # Ensure persistent fields (a,b) and per-call extras (c, step) are present.
        rec = next(r for r in rows if r.get("msg") == "bound-hello")
        # Allow either flattened or nested storage; check permissively.
        flat = {**rec, **rec.get("_g_bound", {}), **rec.get("_g_extra", {})}
        assert flat.get("a") == 1
        assert flat.get("b") == 2
        assert flat.get("c") == 3
        assert flat.get("step") == 10


# --------------------------- Warnings capture --------------------------


@pytest.mark.parametrize("use_public_api", [True, False])
def test_capture_warnings_routes_to_logging_when_true(tmp_path, use_public_api):
    """
    Route warnings into logging + ensure they land in events.log or JSONL.

    `use_public_api=True`: trigger warning via `warnings.warn`.
    `use_public_api=False`: trigger via logging's warnings capture as an internal path.
    """
    import goggles as gg

    with gg.run(
        "warns",
        log_dir=tmp_path,
        enable_console=False,
        capture_warnings=True,
        enable_jsonl=False,
    ) as ctx:
        if use_public_api:
            warnings.simplefilter("always", category=UserWarning)
            warnings.warn("hello-warning", UserWarning)
        else:
            logging.captureWarnings(True)
            warnings.warn("hello-warning", UserWarning)

        gg.get_logger("aux").info("after-warning")
        _flush_handlers()

        text = (Path(ctx.log_dir) / "events.log").read_text(encoding="utf-8")
        print(text)
        assert "hello-warning" in text


# --------------------------- Metrics & media ---------------------------


@pytest.mark.skip(reason="scalar() wired to legacy implementation; needs refactor")
def test_scalar_event_emits_jsonl_when_enabled(tmp_path):
    import goggles as gg

    with gg.run(
        "metrics",
        log_dir=tmp_path,
        enable_console=False,
        enable_jsonl=True,
    ) as ctx:
        gg.scalar("train/loss", 0.123, step=5)
        _flush_handlers()
        rows = _read_jsonl(Path(ctx.log_dir) / "events.jsonl")
        # Accept either direct key or nested payload; probe permissively.
        found = False
        for r in rows:
            if r.get("tag") == "train/loss" or r.get("msg") == "scalar":
                found = True
                break
        assert found


@pytest.mark.skip(
    reason="image() and video() wired to legacy implementation; needs refactor"
)
def test_image_and_video_helpers_do_not_embed_binary(tmp_path):
    import numpy as np
    import goggles as gg

    img = (np.ones((16, 16, 3)) * 255).astype("uint8")
    vid = np.zeros((4, 8, 8, 3), dtype="uint8")  # (T,H,W,C)

    with gg.run(
        "media", log_dir=tmp_path, enable_console=False, enable_jsonl=True
    ) as ctx:
        gg.image("samples/img", img, step=1, split="val")
        gg.video("rollout", vid, step=2, fps=4)
        _flush_handlers()
        rows = _read_jsonl(Path(ctx.log_dir) / "events.jsonl")

        # Ensure entries describe payload (shape/dtype/fps), not raw bytes.
        # We only assert the presence of metadata fields, not exact schema.
        descs = [r for r in rows if r.get("tag") in {"samples/img", "rollout"}]
        assert any("shape" in d or "height" in d for d in descs)
        assert all("data" not in d for d in descs)  # no raw binary blobs


# --------------------------- Propagation & levels ----------------------


def test_log_level_and_propagation_controls(tmp_path):
    import goggles as gg

    # Ensure propagation and level are configurable from run().
    parent = logging.getLogger("goggles.test.parent")
    child = logging.getLogger("goggles.test.parent.child")

    seen = []

    class _Spy(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            seen.append((record.name, record.levelno, record.getMessage()))

    spy = _Spy()
    logging.getLogger().addHandler(spy)

    with gg.run(
        "levels",
        log_dir=tmp_path,
        enable_console=False,
        log_level="WARNING",
        propagate=True,
    ):
        gg.get_logger("goggles.test.parent").info("nope")  # filtered by WARNING
        gg.get_logger("goggles.test.parent.child").warning("bubbled")
        _flush_handlers()

    assert ("goggles.test.parent.child", logging.WARNING, "bubbled") in seen
    assert not any(m == "nope" for _, _, m in seen)
