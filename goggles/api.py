"""Public API contract for Goggles logging (single-process v1).

This module intentionally defines the *outer interface* and behavioral contract for
Goggles' logging system. It is *not* a full implementation: the actual logic lives
in private submodules (e.g., `_core/run.py`, `_core/logger.py`) to separate concerns
and avoid import-time side effects.

NOTE: Metrics and media helpers (`scalar`, `image`, `video`) are included here
as contracts only; their implementations is for now gated behind `NotImplementedError`.

Design principles
-----------------
- No import-time side effects. Importing `goggles` must not attach handlers or
  change the global logging configuration. Only `run(...)` configures handlers.
- Single-process v1. This API targets a single Python process. The interface is
  designed so future multi-process backends can be plugged in without breaking
  call sites (e.g., keep function names, arguments, and return types).
- Structured logging. `get_logger(...)` produces adapters that inject immutable
  `RunContext` and persistent bound fields into each record (e.g., for JSONL).
- Ergonomics. Convenience helpers `scalar/image/video` are provided; they should
  write JSONL-friendly events and mirror to W&B when enabled.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Public RunContext dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunContext:
    """Immutable metadata describing a single logging run.

    This record is created by `run(...)` and is safe to pass across modules.
    It must remain serializable to JSON (e.g., for `metadata.json`).

    Attributes:
        run_id (str): Unique run identifier (UUID4 as canonical string).
        run_name (Optional[str]): Human-friendly name shown in UIs; may be None.
        log_dir (str): Absolute or relative path to the run directory containing
            `events.log`, optional `events.jsonl`, and `metadata.json`.
        created_at (str): ISO8601 UTC timestamp of when the run started.
        pid (int): Process ID that opened the run.
        host (str): Hostname of the machine where the run was created.
        python (str): Python version as `major.minor.micro`.
        metadata (Dict[str, Any]): Arbitrary user-provided metadata captured at
            run creation (experiment args, seeds, git SHA, etc.).
        wandb (Optional[Dict[str, Any]]): Optional W&B info (ids, URL, project).
            This field must be `None` if W&B is not enabled.

    """

    run_id: str
    run_name: Optional[str]
    log_dir: str
    created_at: str
    pid: int
    host: str
    python: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    wandb: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Logger adapter contract
# ---------------------------------------------------------------------------


@runtime_checkable
class BoundLogger(Protocol):
    """Protocol for Goggles' structured logger adapters.

    Implementations typically wrap `logging.Logger` and must:
        - Persist fields provided at construction or via `.bind(...)`.
        - Inject `RunContext` + persistent `bound` fields into each `LogRecord`
            (e.g., via `extra={"_g_bound": bound, "run": ctx, ...}`).
        - Keep per-call keyword arguments (`**extra`) separate from `bound`.

    Methods mirror the standard logging API. Additional methods may be present,
    but these are the minimum required by the public contract.
    """

    # Persistent-field API ----------------------------------------------------

    def bind(self, **fields: Any) -> "BoundLogger":
        """Return a new adapter with `fields` merged into persistent state."""
        raise NotImplementedError(
            "This is an API contract. Provide an implementation in the core layer."
        )

    # Emitters ---------------------------------------------------------------

    def debug(self, msg: str, /, **extra: Any) -> None:
        """Log a DEBUG message with optional structured extras."""

    def info(self, msg: str, /, **extra: Any) -> None:
        """Log an INFO message with optional structured extras."""

    def warning(self, msg: str, /, **extra: Any) -> None:
        """Log a WARNING message with optional structured extras."""

    def error(self, msg: str, /, **extra: Any) -> None:
        """Log an ERROR message with optional structured extras."""

    def exception(self, msg: str, /, **extra: Any) -> None:
        """Log an ERROR with current exception info attached."""


# ---------------------------------------------------------------------------
# Public API functions (contracts only; implementations live in _core)
# ---------------------------------------------------------------------------


def current_run() -> Optional[RunContext]:
    """Return the currently active RunContext for this context (or None).

    This is read-only and reflects the active run as tracked by the backend.
    Implementations should source this from a context-local store (e.g., ContextVar).

    Returns:
        Optional[RunContext]: The active run context, or None if no run is active.

    """
    # Lazy import to avoid import-time side effects / cycles
    from ._core.run import get_active_run

    return get_active_run()


def configure(**defaults: Any) -> None:
    """Override global defaults used by `run(...)`.

    This is an optional convenience to set process-wide defaults *before*
    `run(...)` is called (e.g., enabling JSONL by default). Implementations
    should only mutate internal, in-memory configuration; **do not** attach
    handlers or create directories here.

    Recognized keys (all optional):
      - enable_console (bool): Enable console handler. Default: True.
      - enable_file (bool): Enable text file `events.log`. Default: True.
      - enable_jsonl (bool): Enable `events.jsonl`. Default: False.
      - enable_wandb (bool): Enable W&B integration. Default: False.
      - log_level (str): e.g., "INFO", "DEBUG". Default: "INFO".
      - propagate (bool): Set logger propagation. Default: False.
      - reset_root (bool): Remove existing root handlers at run start.
      - capture_warnings (bool): Route `warnings` to logging. Default: True.

    Raises:
        ValueError: If unknown keys are supplied or values have invalid types.

    Examples:
        >>> configure(enable_jsonl=True, log_level="DEBUG")

    """
    from ._core.config import configure as _configure

    _configure(**defaults)


def run(
    name: Optional[str] = None,
    log_dir: Optional[str] = None,
    *,
    enable_console: Optional[bool] = None,
    enable_file: Optional[bool] = None,
    enable_jsonl: Optional[bool] = None,
    enable_wandb: Optional[bool] = None,
    log_level: Optional[str] = None,
    propagate: Optional[bool] = None,
    reset_root: Optional[bool] = None,
    capture_warnings: Optional[bool] = None,
    enable_artifacts: Optional[bool] = None,
    artifact_name: Optional[str] = None,
    artifact_type: Optional[str] = None,
    **metadata: Any,
) -> AbstractContextManager[RunContext]:
    """Configure logging sinks for the current process and yield a `RunContext`.

    Responsibilities (implementation side):
        1) Create a unique `run_id` (UUID4) and a run directory:
            - `events.log` (plain text)
            - optional `events.jsonl` (structured)
            - `metadata.json` (run context + user metadata)
        2) Install handlers on the root logger according to flags/defaults.
        3) Capture Python warnings if requested (`logging.captureWarnings(True)`).
        4) Optionally initialize W&B (if enabled) and update `RunContext.wandb`.
        5) Yield the immutable `RunContext`.
        6) On context exit: flush/close handlers, finalize metadata (e.g., add
            `finished_at` timestamp), and gracefully teardown W&B (finish + upload).

    Behavior:
        - Exactly-once configuration: if a run is already active, raise
            `RuntimeError` rather than silently stacking handlers.
        - No import-time effects: all side effects occur inside this context.
        - Deterministic defaults may be overridden by `configure(...)` and/or
            keyword arguments passed here.

    Args:
        name (Optional[str]): Human-readable name; may be None.
        log_dir (Optional[str]): Target directory; default `./runs/<run_id>`.
        enable_console (Optional[bool]): Console handler toggle.
        enable_file (Optional[bool]): File handler toggle.
        enable_jsonl (Optional[bool]): JSONL handler toggle.
        enable_wandb (Optional[bool]): W&B integration toggle.
        log_level (Optional[str]): Log level ("INFO", "DEBUG", ...).
        propagate (Optional[bool]): Root logger propagation.
        reset_root (Optional[bool]): Remove existing root handlers first.
        capture_warnings (Optional[bool]): Route `warnings` to logging.
        enable_artifacts (Optional[bool]): Enable artifact logging.
        artifact_name (Optional[str]): Default artifact name.
        artifact_type (Optional[str]): Default artifact type.
        **metadata (Any): User-defined metadata persisted in `metadata.json`.

    Returns:
        AbstractContextManager[RunContext]: A context manager yielding `RunContext`.

    Raises:
        RuntimeError: If a run is already active in this process.
        OSError: If directory creation or file opening fails.
        ValueError: If `log_level` is invalid or incompatible options are set.

    Examples:
        >>> # Application entrypoint
        >>> with run("exp42", enable_jsonl=True) as ctx:
        ...     log = get_logger("train", seed=0)
        ...     log.info("start", step=0)

    """
    # Lazy import to avoid import-time side effects / cycles
    from ._core.run import _RunContextManager

    return _RunContextManager(
        name=name,
        log_dir=log_dir,
        user_metadata=metadata,
        enable_console=enable_console,
        enable_file=enable_file,
        enable_jsonl=enable_jsonl,
        enable_wandb=enable_wandb,
        log_level=log_level,
        propagate=propagate,
        reset_root=reset_root,
        capture_warnings=capture_warnings,
        enable_artifacts=enable_artifacts,
        artifact_name=artifact_name,
        artifact_type=artifact_type,
    )


# Cache the impl after first use to avoid repeated imports
__get_logger_impl: Optional[Callable[[Optional[str]], BoundLogger]] = None


def get_logger(name: Optional[str] = None, **bound: Any) -> BoundLogger:
    """Return a structured logger adapter that injects context and bound fields.

    This function must be safe to call before `run(...)`. In that case, the
    adapter should still function and log to whatever handlers exist (often
    none for libraries), and simply omit run-specific fields until a run is
    active. This enables libraries to depend on Goggles without forcing apps
    to configure Goggles as their logging backend.

    Semantics (implementation side):
        - Construct/return an object conforming to `BoundLogger` (Protocol).
        - Persistent *bound* fields provided here must appear on every record.
        - `.bind(**more)` returns a new adapter whose persistent fields are the
            union of the old and new dictionaries (new values override).
        - Per-call `**extra` must stay separate from the persistent bound fields
            (e.g., `extra={"_g_bound": bound, "_g_extra": extra}`) so sinks can
            distinguish provenance.

    Args:
        name (Optional[str]): Logger name (None for root semantics).
        **bound (Any): Persistent fields injected on every record.

    Returns:
        BoundLogger: Adapter exposing `bind()` and standard logging methods.

    Examples:
        >>> log = get_logger("eval", dataset="CIFAR10")
        >>> log.info("starting")
        >>> log.bind(split="test").info("running", step=1)

    """
    global __get_logger_impl
    if __get_logger_impl is None:
        # Lazy import to avoid import-time side effects / cycles
        from ._core.logger import get_logger as _get_logger

        __get_logger_impl = _get_logger

    return __get_logger_impl(name, **bound)


# ---------------------------------------------------------------------------
# Metrics & media helpers (contracts)
# ---------------------------------------------------------------------------


def scalar(tag: str, value: float, *, step: Optional[int] = None, **kw: Any) -> None:
    """Log a scalar metric.

    Contract:
        - Always emit a JSONL-friendly event (no binary payload).
        - If W&B is enabled, mirror via `wandb.log({tag: value}, step=step)`.
        - Validate `tag` (str) and `value` (number) and raise on invalid input.

    Args:
        tag (str): Metric name (e.g., "train/loss").
        value (float): Numeric value.
        step (Optional[int]): Optional global step.
        **kw (Any): Additional structured fields (stored as "extra" in JSONL).

    Raises:
        TypeError: If types are invalid.
        RuntimeError: If sinks are misconfigured (optional; implementer decision).

    Examples:
        >>> scalar("train/loss", 0.123, step=100)

    """
    raise NotImplementedError(
        "This is an API contract. Provide an implementation in the core layer."
    )


def image(tag: str, data: Any, *, step: Optional[int] = None, **kw: Any) -> None:
    """Log an image artifact.

    Contract:
      - Emit a JSONL-friendly event describing the artifact (shape, dtype),
        but do not write raw binary into JSONL.
      - If W&B is enabled, convert to `wandb.Image` and call `wandb.log`.
      - Implementations should accept common Python image types (NumPy arrays,
        PIL images) and document supported shapes/dtypes.

    Args:
        tag (str): Name (e.g., "samples/input").
        data (Any): Image-like payload acceptable by active backends.
        step (Optional[int]): Optional global step.
        **kw (Any): Additional structured fields.

    Raises:
        TypeError: If `tag` is not str or `data` is unsupported.

    """
    raise NotImplementedError(
        "This is an API contract. Provide an implementation in the core layer."
    )


def video(
    tag: str,
    data: Any,
    *,
    step: Optional[int] = None,
    fps: int = 4,
    **kw: Any,
) -> None:
    """Log a video/sequence artifact.

    Contract:
      - Emit a JSONL-friendly event (shape/fps) without embedding binaries.
      - If W&B is enabled, convert to `wandb.Video` with provided `fps`.
      - Accept common `(T, H, W, C)` arrays or backend-native video types.

    Args:
        tag (str): Name (e.g., "rollout").
        data (Any): Video-like payload acceptable by active backends.
        step (Optional[int]): Optional global step.
        fps (int): Frames per second hint for backends. Default 4.
        **kw (Any): Additional structured fields.

    Raises:
        TypeError: If `tag` is not str or `fps` invalid (<= 0).

    """
    raise NotImplementedError(
        "This is an API contract. Provide an implementation in the core layer."
    )


# ---------------------------------------------------------------------------
# Re-exports and explicit public surface
# ---------------------------------------------------------------------------

__all__ = [
    "RunContext",
    "BoundLogger",
    "configure",
    "run",
    "get_logger",
    "current_run",
    "scalar",
    "image",
    "video",
]

# ---------------------------------------------------------------------------
# Implementation notes for contributors
# ---------------------------------------------------------------------------
# - Keep this file limited to *contracts* (types, docstrings, expectations).
# - Provide real implementations behind a private module hierarchy, e.g.:
#     goggles/_core/run.py        -> actual run() implementation
#     goggles/_core/logger.py     -> BoundLogger implementation
#     goggles/_core/config.py     -> configure() implementation
#     goggles/_core/integrations.py -> W&B and other sink integrations
#
# - In `goggles/__init__.py`, re-export the symbols from this file so users can:
#     import goggles as gg
#     with gg.run(...):
#         gg.get_logger(...).info("...")
#         gg.scalar("train/loss", 0.1, step=1)
# - Tests should assert this module's `__all__` is exported at the package root,
#   and that error modes and lifecycle semantics match the docstrings above.
