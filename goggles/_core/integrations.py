"""Core integrations for logging sinks (e.g., Weights & Biases).

This module manages per-run integrations such as Weights & Biases (W&B).
It provides functions to attach and detach these sinks, handling their
lifecycle in coordination with the run context.
"""

from __future__ import annotations

import logging
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional
from logging import Handler

from goggles._core.config import _CONFIG

logger = logging.getLogger(__name__)

# Track per run resources to clean up on exit
_WANDB_RUNS: Dict[str, Any] = {}
_CONSOLE_HANDLERS: Dict[str, logging.Handler] = {}
_FILE_HANDLERS: Dict[str, logging.Handler] = {}
_ROOT_PREV_LEVEL: Dict[str, int] = {}
_JSONL_HANDLERS: Dict[str, Handler] = {}
_ROOT_PREV_PROPAGATE: Dict[str, bool] = {}
_ROOT_PREV_HANDLERS: Dict[str, list[logging.Handler]] = {}
_WARNINGS_PREV: Dict[str, bool] = {}


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
      - Console logging
      - JSONL log file (events.jsonl)

    Args:
        run_id (str): Unique identifier for the run.
        run_dir (Path): Directory where run artifacts are stored.
        run_name (Optional[str]): Human-readable name of the run.
        user_metadata (Dict[str, Any]): User-provided metadata for the run.
        overrides (Dict[str, Any]): Configuration overrides for this run.

    Returns:
        Dict[str, Any]: Extra metadata to merge into metadata.json and RunContext.
                        Includes "logs.text" and (optionally) "wandb".

    """
    extra: Dict[str, Any] = {}

    root = logging.getLogger()

    # Reset root handlers if requested
    reset_root = overrides.get("reset_root", getattr(_CONFIG, "reset_root", False))
    if reset_root:
        _ROOT_PREV_HANDLERS[run_id] = root.handlers[:]
        for h in list(root.handlers):
            root.removeHandler(h)

    # Propagate setting
    propagate = overrides.get("propagate", getattr(_CONFIG, "propagate", False))
    _ROOT_PREV_PROPAGATE[run_id] = root.propagate
    root.propagate = bool(propagate)

    # Capture warnings (route warnings.warn -> logging)
    # Note: logging.captureWarnings has no getter; we infer "previous" as whether
    # py.warnings logger currently has handlers.
    prev_captured = bool(logging.getLogger("py.warnings").handlers)
    _WARNINGS_PREV[run_id] = prev_captured
    capture = overrides.get(
        "capture_warnings", getattr(_CONFIG, "capture_warnings", True)
    )
    logging.captureWarnings(bool(capture))

    # Store previous root log level to restore on detach
    lvl_name = overrides.get("log_level", getattr(_CONFIG, "log_level", "INFO"))
    try:
        lvl_value = getattr(logging, str(lvl_name).upper())
    except Exception:
        lvl_value = logging.INFO

    _ROOT_PREV_LEVEL[run_id] = root.level
    root.setLevel(lvl_value)

    # Console handler
    enable_console = overrides.get(
        "enable_console", getattr(_CONFIG, "enable_console", True)
    )
    if enable_console:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        logging.getLogger().addHandler(ch)
        _CONSOLE_HANDLERS[run_id] = ch
        extra.setdefault("logs", {})["console"] = "stdout"

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

    # JSONL file
    enable_jsonl = overrides.get(
        "enable_jsonl", getattr(_CONFIG, "enable_jsonl", False)
    )
    if enable_jsonl:
        jsonl_path = Path(run_dir) / "events.jsonl"
        jh = _JsonlHandler(jsonl_path)
        logging.getLogger().addHandler(jh)
        _JSONL_HANDLERS[run_id] = jh
        extra.setdefault("logs", {})["jsonl"] = str(jsonl_path)

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
    # Remove console handler
    ch = _CONSOLE_HANDLERS.pop(run_id, None)
    if ch is not None:
        try:
            ch.flush()
            ch.close()
        finally:
            try:
                logging.getLogger().removeHandler(ch)
            except Exception:
                pass

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

    # Close and remove JSONL handler
    jh = _JSONL_HANDLERS.pop(run_id, None)
    if jh is not None:
        try:
            jh.flush()
            jh.close()
        finally:
            try:
                logging.getLogger().removeHandler(jh)
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

    # Restore root propagate
    prev_prop = _ROOT_PREV_PROPAGATE.pop(run_id, None)
    if prev_prop is not None:
        logging.getLogger().propagate = prev_prop

    # Restore previous root handlers if we cleared them
    prev_handlers = _ROOT_PREV_HANDLERS.pop(run_id, None)
    if prev_handlers:
        root = logging.getLogger()
        for h in prev_handlers:
            try:
                root.addHandler(h)
            except Exception:
                pass

    # Restore warnings capture state
    prev_warn = _WARNINGS_PREV.pop(run_id, None)
    if prev_warn is not None:
        logging.captureWarnings(prev_warn)


class _JsonlHandler(Handler):
    """Write one JSON object per log record (UTF-8, line-delimited).

    Attributes:
        _fp (TextIO): File pointer to the open JSONL file.
        _path (Path): Path to the JSONL file.

    """

    def __init__(self, path: Path) -> None:
        """Initialize the JSONL handler.

        Args:
            path (Path): Path to the JSONL log file.

        """
        super().__init__()
        self._fp = open(path, "a", encoding="utf-8", buffering=1)  # line-buffered
        self._path = path

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record as a JSON object.

        Args:
            record (logging.LogRecord): The log record to emit.

        """
        try:
            payload = {
                "message": record.getMessage(),
                "name": record.name,
                "level": record.levelno,
                "levelname": record.levelname,
                "created": record.created,
                "process": record.process,
                "thread": record.thread,
                "pathname": record.pathname,
                "lineno": record.lineno,
            }
            self._fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        """Close the JSONL handler and its file pointer."""
        try:
            self._fp.close()
        finally:
            super().close()


def upload_artifacts(
    run_id: str,
    files: list[Path],
    *,
    name: Optional[str] = None,
    type: Optional[str] = None,
) -> None:
    """Upload a set of files as a W&B Artifact for the given run.

    - No-ops if W&B isn't active for this run.
    - Silently skips missing files.
    - Deduplicates paths.
    - Uses simple defaults when name/type are not provided.

    Args:
        run_id: Run identifier (used to find the active W&B run).
        files: List of files to include in the artifact.
        name: Optional artifact name; defaults to "goggles-run-<short_id>".
        type: Optional artifact type; defaults to "goggles-run".

    """
    wb = _WANDB_RUNS.get(run_id)
    if wb is None:
        return  # W&B not enabled/initialized for this run

    # Normalize + filter existing files
    uniq: list[Path] = []
    seen: set[str] = set()
    for f in files or []:
        try:
            p = Path(f).resolve()
        except Exception:
            continue
        if not p.exists():
            continue
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)

    if not uniq:
        return

    # Defaults
    art_name = name or f"goggles-run-{run_id[:8]}"
    art_type = type or "goggles-run"

    try:
        import wandb  # type: ignore[import-not-found]
    except Exception as err:
        logger.warning("W&B not importable during artifact upload: %s", err)
        return

    try:
        artifact = wandb.Artifact(
            name=art_name,
            type=art_type,
            metadata={"run_id": run_id},
        )
        for p in uniq:
            try:
                artifact.add_file(str(p))
            except Exception as add_err:
                logger.warning("Failed to add file to artifact: %s (%s)", p, add_err)

        wb.log_artifact(artifact)
    except Exception as err:
        logger.warning("Failed to upload artifacts to W&B: %s", err)
