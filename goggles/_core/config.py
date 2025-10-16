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

_DEFAULTS = {
    "enable_console": True,
    "enable_file": True,
    "enable_jsonl": False,
    "enable_wandb": False,
    "log_level": "INFO",
    "propagate": False,
    "reset_root": None,
    "capture_warnings": True,
}


def configure(**defaults):
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
    # Validate + merge into _DEFAULTS
    for k, v in defaults.items():
        if k not in _DEFAULTS:
            raise ValueError(f"Unknown key: {k}")
        _DEFAULTS[k] = v
