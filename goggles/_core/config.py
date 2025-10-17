"""Configuration management for goggles.

This module provides functions to configure default settings for the goggles logging system.
These settings can be overridden when starting a new run using the `run()` function.
Settings include enabling/disabling various logging backends (console, file, JSONL, Weights & Biases),
log levels, and warning capture.

Example usage:

    import goggles as gg

    # Set global defaults
    gg.configure(enable_console=True, enable_file=False, log_level="DEBUG")

    with gg.run("experiment_1"):
        log = gg.get_logger("train")
        log.debug("This is a debug message.")
        gg.scalar("train/loss", 0.123, step=0)
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class Defaults:
    """Process-wide default configuration for goggles logging.

    Attributes:
        enable_console (bool): Whether to enable console logging by default.
        enable_file (bool): Whether to enable file logging by default.
        enable_jsonl (bool): Whether to enable JSONL logging by default.
        enable_wandb (bool): Whether to enable Weights & Biases logging by default.
        log_level (str): Default log level (e.g., "DEBUG", "INFO").
        propagate (bool): Whether to propagate logs to ancestor loggers.
        reset_root (bool | None): Whether to reset the root logger on run start.
        capture_warnings (bool): Whether to capture Python warnings.

    """

    enable_wandb: bool = False
    enable_file: bool = True
    log_level: str = "INFO"
    # TODO: The rest of the config options will be added in future iterations


_CONFIG = Defaults()


def configure(**defaults: Any) -> None:
    """Set global default configuration for goggles logging.

    These defaults are used when starting a new run with `run()`, but can be
    overridden by arguments passed directly to `run()`.

    Args:
        **defaults: Keyword arguments corresponding to configuration options.
            Valid keys include:
                - enable_console (bool): Enable/disable console logging.
                - enable_file (bool): Enable/disable file logging.
                - enable_jsonl (bool): Enable/disable JSONL logging.
                - enable_wandb (bool): Enable/disable Weights & Biases logging.
                - log_level (str): Default log level (e.g., "DEBUG", "INFO").
                - propagate (bool): Whether to propagate logs to ancestor loggers.
                - reset_root (bool | None): Whether to reset the root logger.
                - capture_warnings (bool): Whether to capture Python warnings.

    Raises:
        ValueError: If an unknown configuration key is provided.

    """
    for key, value in defaults.items():
        if not hasattr(_CONFIG, key):
            raise ValueError(f"Unknown configuration key: {key}")
        setattr(_CONFIG, key, value)
