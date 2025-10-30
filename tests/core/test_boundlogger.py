import json
import logging
from unittest.mock import MagicMock, patch
import pytest
import os
import tempfile
from pathlib import Path


from goggles._core.logger import CoreTextLogger, get_logger as core_get_logger
from goggles import get_logger as api_get_logger, TextLogger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_logger():
    """Create a fake logging.Logger for isolated testing."""
    logger = MagicMock()
    for level in ("debug", "info", "warning", "error", "exception"):
        setattr(logger, level, MagicMock())
    return logger


@pytest.fixture
def core_log(mock_logger):
    """Return a CoreTextLogger bound to a mock logger."""
    return CoreTextLogger(mock_logger)


# ---------------------------------------------------------------------------
# CoreTextLogger: binding behavior
# ---------------------------------------------------------------------------


def test_bind_returns_new_instance(core_log):
    log2 = core_log.bind(a=1)
    assert isinstance(log2, CoreTextLogger)
    assert log2 is not core_log
    assert core_log.get_bound() == {}
    assert log2.get_bound() == {"a": 1}


def test_bind_merges_and_overrides(core_log):
    log1 = core_log.bind(a=1, b=2)
    log2 = log1.bind(b=3, c=4)
    assert log2.get_bound() == {"a": 1, "b": 3, "c": 4}


def test_get_bound_returns_copy(core_log):
    bound = core_log.bind(x=5).get_bound()
    bound["x"] = 99
    assert core_log.get_bound() == {}


# ---------------------------------------------------------------------------
# CoreTextLogger: logging emission
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level", ["debug", "info", "warning", "error", "exception"])
def test_emit_calls_correct_logger_method(core_log, mock_logger, level):
    """Ensure each level calls the correct logging method."""
    getattr(core_log, level)("msg", step=1)
    getattr(mock_logger, level).assert_called_once()
    call_args, call_kwargs = mock_logger.__getattribute__(level).call_args
    assert call_args[0] == "msg"
    extra = call_kwargs["extra"]
    assert "_g_bound" in extra
    assert extra["_g_extra"]["step"] == 1


def test_bind_fields_appear_in_extra(core_log, mock_logger):
    log2 = core_log.bind(task="train")
    log2.info("hello", step=42)
    _, kwargs = mock_logger.info.call_args
    extra = kwargs["extra"]
    assert extra["_g_bound"] == {"task": "train"}

    assert extra["_g_extra"]["step"] == 42


def test_exception_method_calls_logger_exception(core_log, mock_logger):
    core_log.exception("oops", code=500)
    mock_logger.exception.assert_called_once()
    args, kwargs = mock_logger.exception.call_args
    assert args[0] == "oops"
    assert kwargs["extra"]["_g_bound"] == {}


# ---------------------------------------------------------------------------
# Core get_logger behavior
# ---------------------------------------------------------------------------


def test_core_get_logger_creates_coreboundlogger(monkeypatch):
    mock_get_logger = MagicMock(return_value=logging.getLogger("x"))
    with patch("logging.getLogger", mock_get_logger):
        log = core_get_logger("x", run_id="123")
    assert isinstance(log, CoreTextLogger)
    assert log.get_bound() == {"run_id": "123"}


# ---------------------------------------------------------------------------
# API-level integration (wiring)
# ---------------------------------------------------------------------------


def test_api_get_logger_returns_boundlogger_protocol():
    """Ensure api.get_logger returns a TextLogger-conforming adapter."""
    log = api_get_logger("test", exp="unit")
    assert isinstance(log, TextLogger)
    assert hasattr(log, "bind")
    assert hasattr(log, "info")
    # verify binding works
    log2 = log.bind(extra_field=42)
    assert getattr(log2, "get_bound")()["extra_field"] == 42


def test_api_and_core_logger_consistency():
    """Check that api.get_logger and core_get_logger are wired identically."""
    api_logger = api_get_logger("consistency", seed=1)
    core_logger = core_get_logger("consistency", seed=1)
    assert type(api_logger) is type(core_logger)
    assert getattr(api_logger, "get_bound")() == getattr(core_logger, "get_bound")()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_bind_non_str_key_raises_typeerror(core_log):
    with pytest.raises(TypeError):
        # Forcefully simulate bad key type — manually patched behavior
        core_log.bind(**{123: "bad"})  # type: ignore[arg-type]


def test_emit_with_empty_message(core_log, mock_logger):
    core_log.info("")
    mock_logger.info.assert_called_once()


# ---------------------------------------------------------------------------
# Visual test for developer-friendly representation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("to_file", [False, True])
def test_visual_jsonl_boundlogger_demo(to_file):
    """Visual demonstration of CoreTextLogger structured logging.

    This test is *opt-in only*: it is skipped unless the environment variable
    `GOGGLES_SHOW_LOGS=1` is set. It is meant to help developers *see* how bound
    fields persist across log calls while per-call fields remain transient.

    Args:
        to_file (bool): Whether to write logs to a file (in `tests/_logs/`) or
            print to stdout.

    Notes:
        When enabled, this test produces JSONL-style output showing the evolving
        `_g_bound` context of the logger. It does *not* assert anything.
    """
    if os.getenv("GOGGLES_SHOW_LOGS") != "1":
        pytest.skip("Set GOGGLES_SHOW_LOGS=1 to enable visual output demo.")

    # ----------------------------------------------------------------------
    # 1. Prepare output (safe folder inside repo, not /tmp)
    # ----------------------------------------------------------------------
    root = Path(__file__).resolve().parents[2]
    log_dir = root / "tests" / "_logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / "boundlogger_demo.jsonl"

    # Clear old log file only if writing this time
    if to_file and log_path.exists():
        log_path.unlink()

    # ----------------------------------------------------------------------
    # 2. Set up standard logging -> JSONL handler
    # ----------------------------------------------------------------------
    base_logger = logging.getLogger("demo")
    base_logger.setLevel(logging.INFO)
    base_logger.handlers.clear()
    base_logger.propagate = False

    if to_file:
        handler = logging.FileHandler(log_path)
    else:
        handler = logging.StreamHandler()

    class JSONFormatter(logging.Formatter):
        """Formatter that emits clean JSONL with structured context only."""

        def format(self, record):
            payload = {"message": record.getMessage()}

            # Include structured Goggles-specific extras only
            for key, value in record.__dict__.items():
                # Keep _g_bound (persistent context)
                if key == "_g_bound":
                    payload["_g_bound"] = value
                # Keep transient user-provided fields (non-private and non-default)
                elif not key.startswith("_") and key not in {
                    "name",
                    "msg",
                    "args",
                    "levelname",
                    "levelno",
                    "pathname",
                    "filename",
                    "module",
                    "exc_info",
                    "exc_text",
                    "stack_info",
                    "lineno",
                    "funcName",
                    "created",
                    "msecs",
                    "relativeCreated",
                    "thread",
                    "threadName",
                    "processName",
                    "process",
                    "taskName",
                }:
                    payload[key] = value

            return json.dumps(payload)

    handler.setFormatter(JSONFormatter())
    base_logger.addHandler(handler)

    # ----------------------------------------------------------------------
    # 3. Wrap logger with CoreTextLogger and demonstrate binding
    # ----------------------------------------------------------------------
    log = CoreTextLogger(base_logger).bind(app="synthpix", mode="train")
    log.info("start", step=0)

    run_log = log.bind(run_id="exp001")
    run_log.info("running", step=1, lr=1e-3)
    run_log.info("checkpoint", step=5)
    log.info("done", step=10)

    # ----------------------------------------------------------------------
    # 4. Developer feedback
    # ----------------------------------------------------------------------
    if to_file:
        print(f"\n✅ Wrote structured logs to: {log_path}")
        print("Open this file to inspect bound context evolution.\n")
    else:
        print("\n✅ Displayed structured logs above (stdout mode).\n")
