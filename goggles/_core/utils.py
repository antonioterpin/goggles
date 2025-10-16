"""Utility functions for goggles core module."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _now_utc_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _short_id(full_uuid: str, length: int = 8) -> str:
    """Return a shortened run ID string for human-readable directory names.

    Args:
        full_uuid: The full UUID string (hex).
        length: Number of characters to include in the short ID.

    Returns:
        A substring of the full UUID.

    """
    return full_uuid[:length]


def _python_version() -> str:
    """Return the current Python version in `major.minor.micro` format.

    Returns:
        A string representing the Python version.

    Example:
        '3.9.1'

    """
    v = sys.version_info
    return f"{v.major}.{v.minor}.{v.micro}"


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    """Atomically write JSON data to disk.

    Args:
        path: The file path to write the JSON data to.
        data: The JSON data to write.

    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
    tmp_path.replace(path)
