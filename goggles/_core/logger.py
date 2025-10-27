"""Internal Core BoundLogger implementation.

WARNING: This module is an internal implementation detail of Goggles'
logging system. It is not part of the public API.

External code should not import from this module. Instead, depend on:
  - `goggles.BoundLogger` (protocol / interface), and
  - `goggles.get_logger()` (factory returning a BoundLogger)

This module adapts the standard `logging.Logger` to support persistent,
structured context ("bound" fields) that are merged into each log call.
Behavioral compatibility is provided via the `BoundLogger` protocol,
not by inheritance.
"""

import logging
from typing import Any, Dict, Mapping, Optional

from goggles import BoundLogger, current_run


class CoreBoundLogger:
    """Internal concrete implementation of the BoundLogger protocol.

    This adapter wraps a `logging.Logger` and maintains a dictionary of
    persistent, structured fields ("bound" context). Each log call merges
    the bound context with per-call extras before delegating to the underlying
    logger.

    Notes:
        * This class is **internal** to Goggles. Do not rely on its presence,
          constructor, or attributes from external code.
        * External users should obtain a `BoundLogger` via
          `goggles.get_logger()` and program against the protocol.

    Attributes:
        _logger: Underlying `logging.Logger` instance. Internal use only.
        _bound: Persistent structured fields merged into each record.
            Internal use only.

    """

    def __init__(
        self, logger: logging.Logger, bound: Optional[Mapping[str, Any]] = None
    ):
        """Initialize the CoreBoundLogger.

        Args:
            logger: Underlying Python logger.
            bound: Optional initial persistent context to bind.

        """
        self._logger = logger
        self._bound: Dict[str, Any] = dict(bound or {})

    def bind(self, **fields: Any) -> CoreBoundLogger:
        """Return a new logger with `fields` merged into persistent context.

        This method does not mutate the current instance. It returns a new
        adapter whose bound context is the shallow merge of the existing bound
        dictionary and `fields`. Keys in `fields` overwrite existing keys.

        Args:
            **fields: Key-value pairs to bind into the new logger's context.

        Returns:
            CoreBoundLogger: A new adapter with the merged persistent context.

        Raises:
            TypeError: If provided keys are not strings (may occur in stricter
                configurations; current implementation assumes string keys).

        Examples:
            >>> log = get_logger("goggles")  # via public API
            >>> run_log = log.bind(run_id="exp42", module="train")
            >>> run_log.info("Initialized")

        """
        merged = {**self._bound, **fields}

        return self._with_bound(merged)

    def _with_bound(self, bound: Mapping[str, Any]) -> "CoreBoundLogger":
        """Clone the logger with new bound fields.

        Args:
            bound: New persistent context to bind.

        Returns:
            CoreBoundLogger: New instance with updated bound context.

        """
        return CoreBoundLogger(self._logger, bound)

    def get_bound(self) -> Dict[str, Any]:
        """Get a copy of the current persistent bound context.

        Returns:
            Dict[str, Any]: A shallow copy of the bound context dictionary.

        """
        return dict(self._bound)

    def debug(self, msg: str, /, **extra: Any) -> None:
        """Log a DEBUG message with optional per-call structured fields.

        Args:
            msg: Human-readable message.
            **extra: Per-call structured fields merged with the bound context.

        """
        self._emit("debug", msg, **extra)

    def info(self, msg: str, /, **extra: Any) -> None:
        """Log an INFO message with optional per-call structured fields.

        Args:
            msg: Human-readable message.
            **extra: Per-call structured fields merged with the bound context.

        """
        self._emit("info", msg, **extra)

    def warning(self, msg: str, /, **extra: Any) -> None:
        """Log a WARNING message with optional per-call structured fields.

        Args:
            msg: Human-readable message.
            **extra: Per-call structured fields merged with the bound context.

        """
        self._emit("warning", msg, **extra)

    def error(self, msg: str, /, **extra: Any) -> None:
        """Log an ERROR message with optional per-call structured fields.

        Args:
            msg: Human-readable message.
            **extra: Per-call structured fields merged with the bound context.

        """
        self._emit("error", msg, **extra)

    def exception(self, msg: str, /, **extra: Any) -> None:
        """Log an ERROR message with exception info and optional per-call fields.

        This method is a convenience for logging exceptions within an except
        block. It behaves like `error()`, but adds exception information from
        `sys.exc_info()` to the log record.

        Args:
            msg: Human-readable message.
            **extra: Per-call structured fields merged with the bound context.

        """
        self._emit("exception", msg, **extra)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _emit(self, level: str, msg: str, **extra: Any) -> None:
        """Dispatch a log record to the underlying logger.

        Internal helper that merges bound context and per-call extras, then calls
        the corresponding method on the underlying `logging.Logger`.

        Note: `_g_bound` is reserved for Goggles' internal use and should not
        be set by users. "stacklevel" is also reserved to control the logging stack level.
        "run" is reserved to attach the current run context. "_g_extra" contains
        user-provided extra fields.

        Args:
            level: Logging level name (e.g., "info", "debug").
            msg: Message to log.
            **extra: Per-call structured fields.

        """
        # Reserve and pop user-provided control kwargs.
        user_stacklevel = int((extra or {}).pop("stacklevel", 3))

        # Strip reserved keys users shouldn't clobber.
        extra = {
            k: v
            for k, v in (extra or {}).items()
            if k not in {"_g_bound", "_g_extra", "run", "stacklevel"}
        }

        record_extra: Dict[str, Any] = {
            "_g_bound": self._bound,
            "_g_extra": dict(extra),
        }

        run_ctx = current_run()
        if run_ctx is not None:
            record_extra["run"] = run_ctx

        # Dispatch to the underlying logger.
        getattr(self._logger, level)(
            msg, extra=record_extra, stacklevel=user_stacklevel
        )

    # -------------------------------------------------------------------------
    # Introspection / representation
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a developer-friendly string representation.

        Returns:
            str: String representation showing the underlying logger and bound context.

        """
        return (
            f"{self.__class__.__name__}(logger={self._logger!r}, bound={self._bound!r})"
        )


def get_logger(name: Optional[str] = None, /, **bound: Any) -> BoundLogger:
    """Core implementation: create a CoreBoundLogger and apply initial binding.

    Safe to call before `run(...)`: we do not attach handlers or mutate
    logging config here; we just wrap whatever logger exists.

    Args:
        name: Optional name of the logger. If None, the root logger is used.
        **bound: Optional initial persistent context to bind.

    Returns:
        BoundLogger: A new CoreBoundLogger instance with the specified name
            and bound context.

    """
    base = logging.getLogger(name) if name else logging.getLogger()
    adapter: BoundLogger = CoreBoundLogger(base)
    return adapter.bind(**bound) if bound else adapter
