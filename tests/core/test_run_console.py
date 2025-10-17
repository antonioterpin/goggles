import logging
import shutil
import sys
from pathlib import Path

import pytest

from goggles._core.run import _RunContextManager


@pytest.fixture
def tmp_run_dir(tmp_path):
    """Provide a temporary run directory and ensure full cleanup."""
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


def test_console_emits_when_enabled(tmp_run_dir, capsys):
    msg = "hello-from-console"

    ctx = _RunContextManager(
        name="console-on",
        log_dir=str(tmp_run_dir),
        user_metadata={},
        enable_wandb=False,
        enable_console=True,
        enable_file=False,
        enable_jsonl=False,
    )

    with ctx:
        root = logging.getLogger()
        prev = root.level
        try:
            root.setLevel(logging.INFO)
            logging.getLogger().info(msg)
            _flush_handlers()
        finally:
            root.setLevel(prev)

    out, err = capsys.readouterr()
    # By default we stream to stdout; formatter includes level and logger name.
    assert msg in out
    assert msg not in err  # sanity: we used stdout, not stderr


def test_console_does_not_emit_when_disabled(tmp_run_dir, capsys):
    msg = "should-not-appear-on-console"

    ctx = _RunContextManager(
        name="console-off",
        log_dir=str(tmp_run_dir),
        user_metadata={},
        enable_wandb=False,
        enable_console=False,  # explicit off
        enable_file=False,
        enable_jsonl=False,
    )

    with ctx:
        root = logging.getLogger()
        prev = root.level
        try:
            root.setLevel(logging.INFO)
            logging.getLogger().info(msg)
            _flush_handlers()
        finally:
            root.setLevel(prev)

    out, err = capsys.readouterr()
    assert msg not in out
    assert msg not in err


def test_console_respects_log_level(tmp_run_dir, capsys):
    msg = "info-should-be-filtered"

    ctx = _RunContextManager(
        name="console-level",
        log_dir=str(tmp_run_dir),
        user_metadata={},
        enable_wandb=False,
        enable_console=True,
        enable_file=False,
        enable_jsonl=False,
    )
    # Inject a per-run override for level if public wiring isn't present yet.
    ctx._overrides["log_level"] = "WARNING"

    with ctx:
        logging.getLogger().info(msg)
        _flush_handlers()

    out, err = capsys.readouterr()
    assert msg not in out
    assert msg not in err


def test_console_handler_detaches_on_exit(tmp_run_dir, capsys):
    msg1 = "before-exit"
    msg2 = "after-exit"

    ctx = _RunContextManager(
        name="console-detach",
        log_dir=str(tmp_run_dir),
        user_metadata={},
        enable_wandb=False,
        enable_console=True,
        enable_file=False,
        enable_jsonl=False,
    )

    with ctx:
        root = logging.getLogger()
        prev = root.level
        try:
            root.setLevel(logging.INFO)
            logging.getLogger().info(msg1)
            _flush_handlers()
        finally:
            root.setLevel(prev)

    out1, err1 = capsys.readouterr()
    assert msg1 in out1
    assert msg1 not in err1

    # After context exit, the console handler should be gone.
    logging.getLogger().info(msg2)
    _flush_handlers()
    out2, err2 = capsys.readouterr()
    assert msg2 not in out2
    assert msg2 not in err2


def test_no_duplicate_console_handlers_across_runs(tmp_run_dir):
    def count_console_handlers():
        root = logging.getLogger()
        return sum(
            isinstance(h, logging.StreamHandler)
            and getattr(h, "stream", None) is sys.stdout
            for h in root.handlers
        )

    before = count_console_handlers()
    for i in range(2):
        with _RunContextManager(
            name=f"console-{i}",
            log_dir=str(tmp_run_dir),
            user_metadata={},
            enable_wandb=False,
            enable_console=True,
            enable_file=False,
            enable_jsonl=False,
        ):
            pass
    after = count_console_handlers()
    assert after == before
