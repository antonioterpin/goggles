"""Simple util for graceful shutdowns in Python applications."""

import signal
from typing import Optional
from .logger import Goggles


class GracefulShutdown:
    """A context manager for graceful shutdowns."""

    stop = False

    def __init__(
        self,
        exit_message: Optional[str] = None,
    ):
        """Initializes the GracefulShutdown context manager.

        Args:
            exit_message (str): The message to log upon shutdown.
        """
        self.exit_message = exit_message
        # placeholders for original handlers
        self._orig_sigint = None
        self._orig_sigterm = None

    def __enter__(self):
        """Register the signal handlers."""
        # save existing handlers
        self._orig_sigint = signal.getsignal(signal.SIGINT)
        self._orig_sigterm = signal.getsignal(signal.SIGTERM)

        def handle_signal(signum, frame):
            self.stop = True
            if self.exit_message:
                Goggles.info(self.exit_message)

        # register for both SIGINT and SIGTERM
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Unregister the signal handlers, restoring originals."""
        # restore original handlers
        if self._orig_sigint is not None:
            signal.signal(signal.SIGINT, self._orig_sigint)
        if self._orig_sigterm is not None:
            signal.signal(signal.SIGTERM, self._orig_sigterm)
