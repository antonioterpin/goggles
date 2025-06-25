"""Severity Enum for Logging."""

from enum import Enum


class Severity(Enum):
    """Severity levels for logging."""

    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40

    def to_str(self):
        """Convert severity to JSON-compatible string."""
        return self.name

    @classmethod
    def from_str(cls, name):
        """Convert JSON-compatible string to Severity enum."""
        return Severity[name.upper()]

    def to_color(self):
        """Return the ANSI color code for this severity."""
        # ANSI color codes for terminal
        _COLOR_MAP = {
            Severity.DEBUG: "[34m",  # blue
            Severity.INFO: "",  # white "[32m",  # green
            Severity.WARNING: "[33m",  # yellow
            Severity.ERROR: "[31m",  # red
        }
        return _COLOR_MAP[self]

    @classmethod
    def reset_color(cls):
        """Return the ANSI reset code."""
        return "[0m"
