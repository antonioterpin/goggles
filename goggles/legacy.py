"""Legacy API with deprecation warnings."""

from __future__ import annotations
import warnings
from typing import Any
from .api import get_logger, scalar as _scalar, image as _image, video as _video


def _warn(name: str, repl: str) -> None:
    """Issue a deprecation warning for a legacy function."""
    warnings.warn(
        f"goggles.{name} is deprecated and will be removed in a future release; "
        f"use {repl} instead.",
        DeprecationWarning,
        stacklevel=2,
    )


def info(msg: str, /, **extra: Any) -> None:
    """Log an info message."""
    _warn("info", "get_logger(...).info(...)")
    get_logger("goggles.legacy").info(msg, **extra)


def debug(msg: str, /, **extra: Any) -> None:
    """Log a debug message."""
    _warn("debug", "get_logger(...).debug(...)")
    get_logger("goggles.legacy").debug(msg, **extra)


def warning(msg: str, /, **extra: Any) -> None:
    """Log a warning message."""
    _warn("warning", "get_logger(...).warning(...)")
    get_logger("goggles.legacy").warning(msg, **extra)


def error(msg: str, /, **extra: Any) -> None:
    """Log an error message."""
    _warn("error", "get_logger(...).error(...)")
    get_logger("goggles.legacy").error(msg, **extra)


def scalar(tag: str, value: float, *, step: int | None = None, **kw: Any) -> None:
    """Log a scalar metric."""
    _warn("scalar", "goggles.scalar(tag, value, step=..., **kw)")
    _scalar(tag, value, step=step, **kw)


def image(tag: str, data: Any, *, step: int | None = None, **kw: Any) -> None:
    """Log an image artifact."""
    _warn("image", "goggles.image(tag, data, step=..., **kw)")
    _image(tag, data, step=step, **kw)


def video(tag: str, data: Any, *, step: int | None = None, **kw: Any) -> None:
    """Log a video artifact."""
    _warn("video", "goggles.video(tag, data, step=..., **kw)")
    _video(tag, data, step=step, **kw)


# Stubs for legacy scheduler/cleanup APIs — keep as no-ops with guidance.
def init_scheduler(*_a, **_k):
    """Initialize any background scheduling tasks."""
    _warn("init_scheduler", "handled by goggles.run(...)")


def schedule_log(*_a, **_k):
    """Schedule a log to be sent."""
    _warn("schedule_log", "use get_logger(...).info(...)")


def stop_workers(*_a, **_k):
    """Stop any background workers."""
    _warn("stop_workers", "no longer needed; run() manages lifecycle")


def cleanup(*_a, **_k):
    """Perform any necessary cleanup."""
    _warn("cleanup", "no longer needed; run() manages lifecycle")


def new_wandb_run(*_a, **_k):
    """Create a new Weights & Biases run."""
    _warn("new_wandb_run", "enable_wandb=True in run()")


def ensure_tasks_finished(*_a, **_k):
    """Ensure all tasks are finished before proceeding."""
    _warn("ensure_tasks_finished", "run() teardown handles flushing")
