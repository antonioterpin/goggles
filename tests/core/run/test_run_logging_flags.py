import io
import json
import logging
import shutil
import warnings
from pathlib import Path

import pytest

from goggles import run as public_run
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


class _ListHandler(logging.Handler):
    """Simple in-memory handler to inspect which handlers are attached."""

    def __init__(self, store):
        super().__init__()
        self.store = store

    def emit(self, record: logging.LogRecord) -> None:
        self.store.append(record)


def _ctx_factory(use_public_api, *, name, log_dir, **kwargs):
    if use_public_api:
        return public_run(name=name, log_dir=str(log_dir), **kwargs)
    return _RunContextManager(
        name=name, log_dir=str(log_dir), user_metadata={}, **kwargs
    )


# ----------------------------- reset_root ----------------------------- #


@pytest.mark.parametrize("use_public_api", [True, False])
def test_reset_root_clears_and_restores_handlers(tmp_run_dir, use_public_api):
    root = logging.getLogger()

    # Pre-existing handler: should be removed when reset_root=True
    records = []
    dummy = _ListHandler(records)
    root.addHandler(dummy)
    try:
        ctx = _ctx_factory(
            use_public_api,
            name="reset-root",
            log_dir=tmp_run_dir,
            enable_wandb=False,
            enable_file=True,  # ensure at least one sink exists during the run
            enable_jsonl=False,
            enable_console=False,
            reset_root=True,  # <-- feature under test
            log_level="INFO",
        )
        with ctx as runctx:
            # The dummy handler should have been cleared during the run
            assert dummy not in logging.getLogger().handlers
            # And the file sink should be active
            log_path = Path(runctx.log_dir) / "events.log"
            logging.getLogger().info("msg-to-file")
            _flush_handlers()
            assert log_path.exists()
            assert "msg-to-file" in log_path.read_text(encoding="utf-8")
    finally:
        # After exit, the original handler should be restored
        assert dummy in logging.getLogger().handlers
        root.removeHandler(dummy)


# ----------------------------- propagate ----------------------------- #


@pytest.mark.parametrize("use_public_api", [True, False])
@pytest.mark.parametrize("flag", [True, False])
def test_propagate_flag_applied_and_restored(tmp_run_dir, use_public_api, flag):
    root = logging.getLogger()
    prev = root.propagate

    ctx = _ctx_factory(
        use_public_api,
        name="propagate-flag",
        log_dir=tmp_run_dir,
        enable_wandb=False,
        enable_file=False,
        enable_console=False,
        enable_jsonl=False,
        propagate=flag,  # <-- feature under test
    )
    with ctx:
        assert logging.getLogger().propagate is flag

    # Restored after exit
    assert logging.getLogger().propagate is prev


# -------------------------- capture_warnings -------------------------- #


@pytest.mark.parametrize("use_public_api", [True, False])
def test_capture_warnings_routes_to_logging_when_true(tmp_run_dir, use_public_api):
    ctx = _ctx_factory(
        use_public_api,
        name="warnings-on",
        log_dir=tmp_run_dir,
        enable_wandb=False,
        enable_file=True,
        enable_console=False,
        enable_jsonl=False,
        capture_warnings=True,
        log_level="INFO",
    )
    with ctx as runctx:
        log_path = Path(runctx.log_dir) / "events.log"
        # IMPORTANT: don't wrap with pytest.warns here, or logging won't see it
        warnings.warn("hello-warning", UserWarning)
        _flush_handlers()
        text = log_path.read_text(encoding="utf-8")
        assert "hello-warning" in text
        assert "UserWarning" in text

    # After exit, we still expect the warning to be raised, but NOT logged anymore
    with pytest.warns(UserWarning, match="after-exit-warning"):
        warnings.warn("after-exit-warning", UserWarning)
    _flush_handlers()
    size_after = Path(log_path).stat().st_size
    # file should NOT have grown from the after-exit warning
    assert size_after == len(text.encode("utf-8"))


@pytest.mark.parametrize("use_public_api", [True, False])
def test_capture_warnings_false_does_not_log_warnings(tmp_run_dir, use_public_api):
    # Ensure warnings are NOT captured when flag is False
    ctx = _ctx_factory(
        use_public_api,
        name="warnings-off",
        log_dir=tmp_run_dir,
        enable_wandb=False,
        enable_file=True,
        enable_console=False,
        enable_jsonl=False,
        capture_warnings=False,
        log_level="INFO",
    )
    with ctx as runctx:
        log_path = Path(runctx.log_dir) / "events.log"
        # write a "normal" log line to ensure file exists
        logging.getLogger().info("ordinary-line")
        _flush_handlers()
        size_before = log_path.stat().st_size

        # this warning should NOT be routed to logging
        with pytest.warns(UserWarning, match="should-not-appear"):
            warnings.warn("should-not-appear", UserWarning)
        _flush_handlers()
        size_after = log_path.stat().st_size

    assert size_after == size_before
