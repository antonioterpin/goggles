"""Goggles for RL on robotics pipelines."""

from .logger import (
    info,
    debug,
    warning,
    error,
    scalar,
    image,
    vector,
    video,
    schedule_log,
    init_scheduler,
    stop_workers,
    cleanup,
    new_wandb_run,
)
from .config import load_configuration, PrettyConfig
from .shutdown import GracefulShutdown
from .severity import Severity
from .decorators import timeit, trace_on_error

__all__ = [
    "info",
    "debug",
    "warning",
    "error",
    "scalar",
    "image",
    "vector",
    "video",
    "timeit",
    "trace_on_error",
    "load_configuration",
    "PrettyConfig",
    "GracefulShutdown",
    "Severity",
    "init_scheduler",
    "schedule_log",
    "cleanup",
    "stop_workers",
    "new_wandb_run",
]

import atexit

atexit.register(cleanup)
