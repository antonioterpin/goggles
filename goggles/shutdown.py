"""Simple util for graceful shutdowns in Python applications."""

import signal
import types
from typing_extensions import Self


class GracefulShutdown:
    """A context manager for graceful shutdowns.

    Example:
    >>> with GracefulShutdown(exit_message="Shutting down gracefully...") as gs:
    ...     while not gs.stop:
    ...         # Main application logic here, runs until interrupted
    ...         # by SIGINT or SIGTERM.
    ...         pass
    ...     print("Cleanup and exit.")

    """

    stop = False

    def __init__(
        self,
        exit_message: str | None = None,
    ) -> None:
        """Initialize the GracefulShutdown context manager.

        Args:
            exit_message: The message to log upon shutdown.

        """
        from . import get_logger

        self.logger = get_logger("goggles.shutdown")
        self.exit_message = exit_message
        # placeholders for original handlers
        self._orig_sigint = None
        self._orig_sigterm = None

    def __enter__(self) -> Self:
        """Register the signal handlers."""
        # save existing handlers
        self._orig_sigint = signal.getsignal(signal.SIGINT)
        self._orig_sigterm = signal.getsignal(signal.SIGTERM)

        def handle_signal(signum, frame):
            self.stop = True
            if self.exit_message:
                self.logger.info(self.exit_message)

        # register for both SIGINT and SIGTERM
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_value: Exception | None,
        traceback: types.TracebackType | None,
    ) -> None:
        """Unregister the signal handlers, restoring originals.

        Args:
            exc_type: Exception type if any.
            exc_value: Exception value if any.
            traceback: Traceback if any.

        """
        # restore original handlers
        if self._orig_sigint is not None:
            signal.signal(signal.SIGINT, self._orig_sigint)
        if self._orig_sigterm is not None:
            signal.signal(signal.SIGTERM, self._orig_sigterm)
