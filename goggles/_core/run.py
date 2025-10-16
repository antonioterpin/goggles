"""Core implementation of the run(...) context manager.

This module defines the foundational context manager for managing experiment runs.
It handles creating a unique run directory, generating a run ID, and writing
initial metadata. More advanced features like logging handlers and W&B
integration are handled elsewhere.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import uuid
from contextlib import AbstractContextManager
from contextvars import ContextVar
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from goggles import RunContext
from .utils import _now_utc_iso, _python_version, _short_id, _write_json

# The currently active run context, if any. Prevents nested run(...) calls.
_ACTIVE_RUN: ContextVar[RunContext | None] = ContextVar("_g_active_run", default=None)


class _RunContextManager(AbstractContextManager[RunContext]):
    """Core implementation of the run(...) context manager.

    This minimal version performs only the foundational tasks needed to start
    and finish a run:
      - Create a uniquely named run directory.
      - Generate a unique run ID.
      - Write an initial `metadata.json` file with environment details.
      - Record user-provided metadata.
      - On exit, update `metadata.json` with a `finished_at` timestamp.

    No logging handlers, filters, or W&B integrations are initialized here.
    """

    def __init__(
        self,
        name: Optional[str],
        log_dir: Optional[str],
        *,
        user_metadata: Optional[Dict[str, Any]] = None,
        enable_wandb: Optional[bool] = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the context manager with run configuration.

        Args:
            name (Optional[str]): Human-readable run name.
            log_dir (Optional[str]): Base directory for run subdirectory.
            user_metadata (Dict[str, Any]): Arbitrary user metadata.
            enable_wandb (bool): Whether to initialize a Weights & Biases run.
            **kwargs (Any): Reserved for future extensions.

        """
        self._run_name = name
        self._base_dir = Path(log_dir) if log_dir else Path("runs")
        self._user_metadata = dict(user_metadata) if user_metadata is not None else {}
        self._context: Optional[RunContext] = None
        self._run_path: Optional[Path] = None
        self._wandb_run: Optional[Any] = None
        self._enable_wandb = enable_wandb
        # Mark kwargs as used so static checkers don't warn about unused parameters.
        del kwargs

    def __enter__(self) -> RunContext:
        """Create and register a new RunContext.

        Optionally starts a Weights & Biases (W&B) run if enabled.

        Raises:
            RuntimeError: If a run is already active in this process.

        Returns:
            RunContext: Newly created context for the active run.

        """
        if _ACTIVE_RUN.get() is not None:
            raise RuntimeError("A run is already active in this process.")

        # Generate unique identifiers and directories
        run_id = uuid.uuid4().hex
        created_at = _now_utc_iso()
        timestamp = datetime.fromisoformat(created_at).strftime("%Y%m%d_%H%M%S")
        short_id = _short_id(run_id)
        human_prefix = self._run_name or "run"
        run_dir = (self._base_dir / f"{human_prefix}-{timestamp}-{short_id}").resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        self._run_path = run_dir

        # Initialize W&B if requested
        wandb_info: Optional[Dict[str, Any]] = None
        if self._enable_wandb:
            try:
                import wandb

                self._wandb_run = wandb.init(
                    project=self._user_metadata.get("project", "goggles"),
                    name=self._run_name,
                    dir=str(run_dir),
                    config=self._user_metadata,
                    reinit=True,
                    settings=wandb.Settings(start_method="thread"),
                )
                wandb_info = {
                    "id": self._wandb_run.id,
                    "url": self._wandb_run.url,
                    "project": self._wandb_run.project_name(),
                    "entity": getattr(self._wandb_run, "entity", None),
                }
            except ImportError:
                print(
                    "[goggles.run] W&B not installed; continuing without integration.",
                    file=sys.stderr,
                )
                self._enable_wandb = False
            except Exception as err:
                print(f"[goggles.run] Failed to initialize W&B: {err}", file=sys.stderr)
                self._enable_wandb = False

        # Build immutable RunContext
        run_context = RunContext(
            run_id=run_id,
            run_name=self._run_name,
            log_dir=str(run_dir),
            created_at=created_at,
            pid=os.getpid(),
            host=socket.gethostname(),
            python=_python_version(),
            metadata=self._user_metadata,
            wandb=wandb_info,
        )
        self._context = run_context

        # Write initial metadata file
        metadata = {
            **asdict(run_context),
            "goggles_version": os.environ.get("GOGGLES_VERSION", "unknown"),
            "user": os.environ.get("USER") or os.environ.get("USERNAME") or "unknown",
        }
        _write_json(run_dir / "metadata.json", metadata)

        _ACTIVE_RUN.set(run_context)
        return run_context

    def __exit__(self, exc_type, exc, tb) -> None:
        """Finalize the run and close resources.

        Updates `metadata.json` with `finished_at` and closes the W&B run if active.

        Args:
            exc_type: Exception type if raised inside context.
            exc: Exception instance if raised inside context.
            tb: Traceback if raised inside context.

        """
        try:
            if self._context and self._run_path:
                metadata_path = self._run_path / "metadata.json"
                try:
                    with metadata_path.open("r", encoding="utf-8") as file:
                        data = json.load(file)
                except Exception:
                    data = asdict(self._context)

                data["finished_at"] = _now_utc_iso()
                _write_json(metadata_path, data)

            # Safely close W&B if active
            if self._enable_wandb and self._wandb_run is not None:
                try:
                    self._wandb_run.finish()
                except Exception as err:
                    print(
                        f"[goggles.run] Warning: Failed to close W&B run cleanly: {err}",
                        file=sys.stderr,
                    )
        finally:
            _ACTIVE_RUN.set(None)
