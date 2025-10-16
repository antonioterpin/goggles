import logging
from unittest.mock import MagicMock, patch
import pytest

from goggles._core.logger import CoreBoundLogger, get_logger as core_get_logger
from goggles import get_logger as api_get_logger, BoundLogger


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
    """Return a CoreBoundLogger bound to a mock logger."""
    return CoreBoundLogger(mock_logger)


# ---------------------------------------------------------------------------
# CoreBoundLogger: binding behavior
# ---------------------------------------------------------------------------


def test_bind_returns_new_instance(core_log):
    log2 = core_log.bind(a=1)
    assert isinstance(log2, CoreBoundLogger)
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
# CoreBoundLogger: logging emission
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
    assert extra["step"] == 1


def test_bind_fields_appear_in_extra(core_log, mock_logger):
    log2 = core_log.bind(task="train")
    log2.info("hello", step=42)
    _, kwargs = mock_logger.info.call_args
    extra = kwargs["extra"]
    assert extra["_g_bound"] == {"task": "train"}
    assert extra["step"] == 42


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
    assert isinstance(log, CoreBoundLogger)
    assert log.get_bound() == {"run_id": "123"}


# ---------------------------------------------------------------------------
# API-level integration (wiring)
# ---------------------------------------------------------------------------


def test_api_get_logger_returns_boundlogger_protocol():
    """Ensure api.get_logger returns a BoundLogger-conforming adapter."""
    log = api_get_logger("test", exp="unit")
    assert isinstance(log, BoundLogger)
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
