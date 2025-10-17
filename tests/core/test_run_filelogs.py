import json
import logging
from pathlib import Path
import shutil

import pytest

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


def test_text_log_created_and_receives_records(tmp_run_dir):
    msg = "hello-from-text-log"

    ctx_mgr = _RunContextManager(
        name="file-on",
        log_dir=str(tmp_run_dir),
        user_metadata={"project": "demo_project"},
        enable_wandb=False,
        enable_file=True,
    )

    with ctx_mgr as ctx:
        root = logging.getLogger()
        prev_level = root.level
        try:
            root.setLevel(logging.INFO)  # ensure INFO is emitted to handlers
            run_path = Path(ctx.log_dir)
            log_path = run_path / "events.log"

            logging.getLogger().info(msg)
            for h in root.handlers:
                try:
                    h.flush()
                except Exception:
                    pass

            assert (
                log_path.exists()
            ), "events.log should be created when enable_file=True"
            contents = log_path.read_text(encoding="utf-8")
            assert msg in contents, "Logged message should appear in events.log"
        finally:
            root.setLevel(prev_level)  # restore


def test_text_log_not_created_when_disabled(tmp_run_dir):
    ctx_mgr = _RunContextManager(
        name="file-off",
        log_dir=str(tmp_run_dir),
        user_metadata={"project": "demo_project"},
        enable_wandb=False,
        enable_file=False,
    )

    with ctx_mgr as ctx:
        run_path = Path(ctx.log_dir)
        logging.getLogger().info("this-should-not-go-to-file")
        # Flush anyway—there should be no file handler
        for h in logging.getLogger().handlers:
            try:
                h.flush()
            except Exception:
                pass

        assert not (
            run_path / "events.log"
        ).exists(), "events.log must not exist if enable_file=False"


def test_metadata_contains_text_log_path_when_enabled(tmp_run_dir):
    ctx_mgr = _RunContextManager(
        name="meta-file-on",
        log_dir=str(tmp_run_dir),
        user_metadata={"project": "demo_project"},
        enable_wandb=False,
        enable_file=True,
    )

    with ctx_mgr as ctx:
        meta_path = Path(ctx.log_dir) / "metadata.json"
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        # logs section should exist and point to events.log
        assert (
            "logs" in data and "text" in data["logs"]
        ), "metadata should include logs.text path"
        assert Path(data["logs"]["text"]).name == "events.log"
        assert Path(data["logs"]["text"]).exists()


def test_text_log_handler_detaches_on_exit(tmp_run_dir):
    ctx_mgr = _RunContextManager(
        name="detach-file",
        log_dir=str(tmp_run_dir),
        user_metadata={"project": "demo_project"},
        enable_wandb=False,
        enable_file=True,
    )

    with ctx_mgr as ctx:
        log_path = Path(ctx.log_dir) / "events.log"
        logging.getLogger().info("line-before-exit")
        for h in logging.getLogger().handlers:
            try:
                h.flush()
            except Exception:
                pass
        size_before = log_path.stat().st_size

    # After exit, the file handler should be removed. Logging should not grow that file.
    logging.getLogger().info("line-after-exit")
    # Give the logging system a tick
    for h in logging.getLogger().handlers:
        try:
            h.flush()
        except Exception:
            pass

    size_after = Path(log_path).stat().st_size
    assert (
        size_after == size_before
    ), "events.log size should not grow after handler detach"


def test_configure_enable_file_then_run_override_disable(tmp_run_dir):
    from goggles._core import config as cfg

    cfg.configure(enable_file=True)

    ctx_mgr = _RunContextManager(
        name="file-override-off",
        log_dir=str(tmp_run_dir),
        user_metadata={},
        enable_file=False,  # override should win
        enable_wandb=False,
    )
    with ctx_mgr as ctx:
        assert not (Path(ctx.log_dir) / "events.log").exists()


def test_configure_enable_file_then_run_none_uses_default(tmp_run_dir):
    from goggles._core import config as cfg

    cfg.configure(enable_file=True)

    ctx_mgr = _RunContextManager(
        name="file-default-on",
        log_dir=str(tmp_run_dir),
        user_metadata={},
        enable_file=None,  # no opinion → default True
        enable_wandb=False,
    )
    with ctx_mgr as ctx:
        assert (Path(ctx.log_dir) / "events.log").exists()


def test_file_sink_detaches_on_exception(tmp_run_dir):
    ctx_mgr = _RunContextManager(
        name="file-exception",
        log_dir=str(tmp_run_dir),
        user_metadata={},
        enable_file=True,
        enable_wandb=False,
    )
    try:
        with ctx_mgr as ctx:
            log_path = Path(ctx.log_dir) / "events.log"
            assert log_path.exists()
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    size_before = log_path.stat().st_size
    logging.getLogger().info("after-exit-should-not-append")
    for h in logging.getLogger().handlers:
        try:
            h.flush()
        except Exception:
            pass
    assert log_path.stat().st_size == size_before


def test_metadata_does_not_include_text_when_disabled(tmp_run_dir):
    ctx_mgr = _RunContextManager(
        name="meta-no-file",
        log_dir=str(tmp_run_dir),
        user_metadata={},
        enable_file=False,
        enable_wandb=False,
    )
    with ctx_mgr as ctx:
        meta = json.loads((Path(ctx.log_dir) / "metadata.json").read_text())
        assert "logs" not in meta or "text" not in meta.get("logs", {})


def test_no_duplicate_handlers_across_runs(tmp_run_dir):
    def count_file_handlers():
        return sum(
            isinstance(h, logging.FileHandler) for h in logging.getLogger().handlers
        )

    before = count_file_handlers()
    for i in range(2):
        with _RunContextManager(
            name=f"file-{i}",
            log_dir=str(tmp_run_dir),
            user_metadata={},
            enable_file=True,
            enable_wandb=False,
        ):
            pass
    after = count_file_handlers()
    assert after == before


def test_log_level_applied_during_run(tmp_run_dir):
    from goggles._core.run import _RunContextManager
    import logging

    ctx_mng = _RunContextManager(
        name="lvl",
        log_dir=str(tmp_run_dir),
        user_metadata={},
        enable_file=True,
    )
    root = logging.getLogger()
    prev = root.level
    try:
        with ctx_mng as ctx:
            # default is INFO, so DEBUG should be filtered
            logging.getLogger().debug("debug-should-not-appear")
            # force flush then check file
            log_path = Path(ctx.log_dir) / "events.log"
            for h in root.handlers:
                try:
                    h.flush()
                except Exception:
                    pass
            text = log_path.read_text(encoding="utf-8")
            assert "debug-should-not-appear" not in text
    finally:
        root.setLevel(prev)


def test_log_level_restored_after_run(tmp_run_dir):
    from goggles._core.run import _RunContextManager
    import logging

    root = logging.getLogger()
    prev = root.level
    ctx = _RunContextManager(
        name="lvl-restore",
        log_dir=str(tmp_run_dir),
        user_metadata={},
        enable_file=True,
        # override to WARNING explicitly
        # (only set overrides dict entry if not None in your __init__)
    )
    # inject a run-time override:
    ctx._overrides["log_level"] = "WARNING"
    with ctx:
        assert logging.getLogger().level == logging.WARNING
    assert logging.getLogger().level == prev
