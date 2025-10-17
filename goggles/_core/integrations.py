"""Core integrations for logging sinks (e.g., Weights & Biases).

This module manages per-run integrations such as Weights & Biases (W&B).
It provides functions to attach and detach these sinks, handling their
lifecycle in coordination with the run context.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from goggles._core.config import _CONFIG

logger = logging.getLogger(__name__)

# Track per run resources to clean up on exit
_WANDB_RUNS: Dict[str, Any] = {}
_FILE_HANDLERS: Dict[str, logging.Handler] = {}
_ROOT_PREV_LEVEL: Dict[str, int] = {}


def attach_sinks(
    *,
    run_id: str,
    run_dir: Path,
    run_name: Optional[str],
    user_metadata: Dict[str, Any],
    overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """Attach per-run integrations.

    Supports:
      - Text log file (events.log)
      - Weights & Biases (wandb)

    Returns:
        Dict[str, Any]: Extra metadata to merge into metadata.json and RunContext.
                        Includes "logs.text" and (optionally) "wandb".

    """
    extra: Dict[str, Any] = {}

    root = logging.getLogger()
    # Store previous root log level to restore on detach
    lvl_name = overrides.get("log_level", _CONFIG.log_level)
    try:
        lvl_value = getattr(logging, str(lvl_name).upper())
    except Exception:
        lvl_value = logging.INFO

    _ROOT_PREV_LEVEL[run_id] = root.level
    root.setLevel(lvl_value)

    # Text .log file
    enable_file = overrides.get("enable_file", getattr(_CONFIG, "enable_file", True))
    if enable_file:
        log_path = Path(run_dir) / "events.log"
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logging.getLogger().addHandler(handler)
        _FILE_HANDLERS[run_id] = handler
        extra.setdefault("logs", {})["text"] = str(log_path)

    # Weights & Biases
    enable_wandb = overrides.get(
        "enable_wandb", getattr(_CONFIG, "enable_wandb", False)
    )
    if enable_wandb:
        try:
            import wandb  # type: ignore[import-not-found]

            wb = wandb.init(
                project=user_metadata.get("project", "goggles"),
                name=run_name,
                dir=str(run_dir),
                config=user_metadata,
                reinit=True,
                settings=wandb.Settings(start_method="thread"),
            )
            _WANDB_RUNS[run_id] = wb
            extra["wandb"] = {
                "id": wb.id,
                "url": wb.url,
                "project": wb.project_name(),
                "entity": getattr(wb, "entity", None),
            }
        except Exception as err:
            logger.warning("Failed to initialize W&B; continuing without it: %s", err)

    return extra


def detach_sinks(run_id: str) -> None:
    """Tear down per-run integrations. W&B only (for now).

    Args:
        run_id (str): The unique identifier of the run to clean up.

    """
    # Close and remove text file handler
    handler = _FILE_HANDLERS.pop(run_id, None)
    if handler is not None:
        try:
            handler.flush()
            handler.close()
        finally:
            try:
                logging.getLogger().removeHandler(handler)
            except Exception:
                pass

    # Finish W&B run if active
    wb = _WANDB_RUNS.pop(run_id, None)
    if wb is not None:
        try:
            wb.finish()
        except Exception as err:
            logger.warning("Failed to close W&B run cleanly: %s", err)

    # Restore previous root log level
    prev_level = _ROOT_PREV_LEVEL.pop(run_id, None)
    if prev_level is not None:
        logging.getLogger().setLevel(prev_level)
