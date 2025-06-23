"""Utility functions for Goggles."""

import os


def safe_chmod(path, mode):
    """Change the mode of a file or directory, ignoring errors if the operation fails."""
    try:
        os.chmod(path, mode)
    except PermissionError:
        # can’t change ownership or perms on a file you don’t own: ignore
        pass
