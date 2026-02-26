"""Global configuration for Goggles loggers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import logging


@dataclass
class ConsoleConfig:
    """Configuration for console logging.

    Attributes:
        enabled: Whether console logging is enabled (gets flipped True when a ConsoleHandler is attached).
        name: The name of the console handler.
        level: The logging level for the console handler.
        path_style: Whether to display absolute or relative file paths in log messages.
        project_root: The root directory to use for relative paths (if path_style is "relative").
    """

    enabled: bool = False  # gets flipped True when a ConsoleHandler is attached
    name: str = "goggles.console"
    level: int = logging.NOTSET
    path_style: Literal["absolute", "relative"] = "relative"
    project_root: Path = field(default_factory=lambda: Path.cwd())


@dataclass
class WandBConfig:
    """Configuration for WandB logging.

    Attributes:
        enabled: Whether WandB logging is enabled (gets flipped True when a WandBHandler is attached).
        project: The W&B project name to log to.
        entity: The W&B entity (user or team) to log under.
        run_name: The name of the W&B run.
        group: The W&B group to associate this run with.
        reinit: How to handle existing runs when initializing a new run. Options are:
            - "default": Use W&B's default behavior (which may vary based on context).
            - "return_previous": If an active run exists, return it instead of creating a new one.
            - "finish_previous": If an active run exists, finish it before creating a new one.
            - "create_new": Always create a new run, even if an active run exists (this may lead to multiple active runs).
        config: A dictionary of configuration values to log with the run.
    """

    enabled: bool = False  # gets flipped True when a WandBHandler is attached
    project: str | None = None
    entity: str | None = None
    run_name: str | None = None
    group: str | None = None
    reinit: str = "create_new"
    config: dict[str, Any] = field(default_factory=dict)


# --- Process-global config (per-process, per-python-interpreter) ---
CONSOLE = ConsoleConfig()
WANDB = WandBConfig()
