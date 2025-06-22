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

    def __enter__(self):
        """Register the signal handler."""

        def handle_signal(signum, frame):
            self.stop = True
            if self.exit_message:
                Goggles.info(self.exit_message)

        signal.signal(signal.SIGINT, handle_signal)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Unregister the signal handler."""
        pass
