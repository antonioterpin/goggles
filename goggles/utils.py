"""Utility functions for Goggles."""

import os
import fcntl
from contextlib import contextmanager


def safe_chmod(path, mode):
    """Change the mode of a file or directory, ignoring errors if the operation fails."""
    try:
        os.chmod(path, mode)
    except PermissionError:
        # can’t change ownership or perms on a file you don’t own: ignore
        pass


class FileRWLock:
    """A simple file-based reader-writer lock."""

    def __init__(self, path: str, flags=os.O_RDWR | os.O_CREAT, mode=0o666):
        """Initialize the FileRWLock.

        Args:
            path (str): The path to the lock file.
            flags (int): Flags for opening the file (default: O_RDWR | O_CREAT).
            mode (int): File mode for the lock file (default: 0o666).
        """
        self.path = path
        self.fd = os.open(path, flags, mode)

    @contextmanager
    def read_lock(self):
        """Acquire a shared (reader) lock."""
        try:
            fcntl.flock(self.fd, fcntl.LOCK_SH)
            yield
        finally:
            fcntl.flock(self.fd, fcntl.LOCK_UN)

    @contextmanager
    def write_lock(self):
        """Acquire an exclusive (writer) lock."""
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(self.fd, fcntl.LOCK_UN)

    def close(self):
        """Close the file descriptor."""
        os.close(self.fd)
