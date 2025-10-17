"""Goggles: structured logging and experiment tracking for research pipelines.

This package provides a stable public API for logging experiments, metrics,
and media in a consistent and composable way.

Typical usage example:

    import goggles as gg

    with gg.run("experiment_42"):
        log = gg.get_logger("train", seed=0)
        log.info("Training started.")
        gg.scalar("train/loss", 0.123, step=0)

The functions exposed at the package root are documented in `goggles/api.py`.
No import-time side effects occur; logging is configured only when `run()`
is called.

See Also:
    - `goggles.api`: Public API contracts and documentation.
    - Internal implementations live under `goggles/_core/` (not public).

"""

from __future__ import annotations
import warnings
import logging

# ---------------------------------------------------------------------------
# Public API re-exports
# ---------------------------------------------------------------------------

from .api import (
    RunContext,
    BoundLogger,
    configure,
    run,
    get_logger,
    current_run,
    scalar as _scalar_impl,
    image as _image_impl,
    video as _video_impl,
)

__all__ = [
    "RunContext",
    "BoundLogger",
    "configure",
    "run",
    "get_logger",
    "current_run",
    "scalar",
    "image",
    "video",
]

# ---------------------------------------------------------------------------
# Import-time safety
# ---------------------------------------------------------------------------

# Attach a NullHandler so importing goggles never emits logs by default.

_logger = logging.getLogger(__name__)
if not any(isinstance(h, logging.NullHandler) for h in _logger.handlers):
    _logger.addHandler(logging.NullHandler())
# ---------------------------------------------------------------------------
# Legacy entry points (adapters)
# ---------------------------------------------------------------------------

_log = None  # type: ignore


def _ensure_legacy_logger():
    """Create a logger only if get_logger() is implemented."""
    global _log
    if _log is None:
        try:
            from .api import get_logger

            _log = get_logger("goggles.legacy")
        except NotImplementedError:
            # Fall back to plain stdlib logger during contract-testing phase
            import logging

            _log = logging.getLogger("goggles.legacy")
    return _log


def info(msg: str, /, **extra):
    """Legacy alias for get_logger().info()."""
    warnings.warn(
        "Direct calls like goggles.info()/scalar() are deprecated...",
        DeprecationWarning,
        stacklevel=2,
    )

    _ensure_legacy_logger().info(msg, **extra)


def debug(msg: str, /, **extra):
    """Legacy alias for get_logger().debug()."""
    warnings.warn(
        "Direct calls like goggles.info()/scalar() are deprecated...",
        DeprecationWarning,
        stacklevel=2,
    )
    _ensure_legacy_logger().debug(msg, **extra)


def warning(msg: str, /, **extra):
    """Legacy alias for get_logger().warning()."""
    warnings.warn(
        "Direct calls like goggles.info()/scalar() are deprecated...",
        DeprecationWarning,
        stacklevel=2,
    )
    _ensure_legacy_logger().warning(msg, **extra)


def error(msg: str, /, **extra):
    """Legacy alias for get_logger().error()."""
    warnings.warn(
        "Direct calls like goggles.info()/scalar() are deprecated...",
        DeprecationWarning,
        stacklevel=2,
    )
    _ensure_legacy_logger().error(msg, **extra)


def scalar(tag, value, *, step=None, **kw):
    """Log a scalar metric."""
    _scalar_impl(tag, value, step=step, **kw)


def image(tag, data, *, step=None, **kw):
    """Log an image artifact."""
    _image_impl(tag, data, step=step, **kw)


def video(tag, data, *, step=None, **kw):
    """Log a video artifact."""
    _video_impl(tag, data, step=step, **kw)
