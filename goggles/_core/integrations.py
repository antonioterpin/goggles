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

# Keep the active W&B run handle per run_id (if any).
_WANDB_RUNS: Dict[str, Any] = {}


def attach_sinks(
    *,
    run_id: str,
    run_dir: Path,
    run_name: Optional[str],
    user_metadata: Dict[str, Any],
    overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """Attach per-run integrations. W&B only (for now).

    Returns:
        Dict[str, Any]: Extra metadata to merge into metadata.json and RunContext.
                        Includes a "wandb" entry if W&B was started.

    """
    # Resolve effective enable flag: run override -> configured default
    enable_wandb = overrides.get("enable_wandb", _CONFIG.enable_wandb)
    if not enable_wandb:
        return {}

    try:
        import wandb  # type: ignore[import-not-found]
    except Exception as err:
        logger.warning("W&B not available; continuing without integration: %s", err)
        return {}

    try:
        wb = wandb.init(
            project=user_metadata.get("project", "goggles"),
            name=run_name,
            dir=str(run_dir),
            config=user_metadata,
            reinit=True,
            settings=wandb.Settings(start_method="thread"),
        )
        _WANDB_RUNS[run_id] = wb
        return {
            "wandb": {
                "id": wb.id,
                "url": wb.url,
                "project": wb.project_name(),
                "entity": getattr(wb, "entity", None),
            }
        }
    except Exception as err:
        logger.warning("Failed to initialize W&B; continuing without it: %s", err)
        return {}


def detach_sinks(run_id: str) -> None:
    """Tear down per-run integrations. W&B only (for now)."""
    wb = _WANDB_RUNS.pop(run_id, None)
    if wb is None:
        return
    try:
        wb.finish()
    except Exception as err:
        logger.warning("Failed to close W&B run cleanly: %s", err)
