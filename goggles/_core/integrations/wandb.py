"""WandB integration handler for Goggles logging framework."""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional, FrozenSet

import wandb

Run = Any  # wandb.sdk.wandb_run.Run


class WandBHandler:
    """Forward Goggles events to separate concurrent W&B runs per event scope.

    Each scope corresponds to a distinct W&B run that remains active until
    explicitly closed. This allows different scopes to log concurrently,
    without finishing previous runs when switching scopes.

    Supported event kinds:
      - ``metric``: ``payload`` is a mapping ``{name: value}``.
      - ``image`` or ``video``: ``payload`` is an array-like or mapping
        ``{"name": array}``.
      - ``artifact``: ``payload`` is a mapping with keys ``path``, ``name``,
        and ``type``.

    Attributes:
        name (str): Stable handler identifier ("wandb").
        capabilities (set[str]): Supported event kinds.
        GLOBAL_SCOPE (str): Default scope name for the global run.

    """

    name: str = "wandb"
    capabilities: FrozenSet[str] = frozenset({"metric", "image", "video", "artifact"})
    GLOBAL_SCOPE = "global"

    def __init__(
        self,
        project: Optional[str] = None,
        entity: Optional[str] = None,
        run_name: Optional[str] = None,
        config: Optional[Mapping[str, Any]] = None,
        reinit: Optional[str] = None,
    ) -> None:
        """Initialize the W&B handler.

        Args:
            project (Optional[str]): W&B project name.
            entity (Optional[str]): W&B entity name.
            run_name (Optional[str]): Base run name used as prefix for scopes.
            config (Optional[Mapping[str, Any]]): Default run configuration.
            reinit (Optional[str]): W&B reinitialization mode, one of
                {"finish_previous", "return_previous", "create_new", None}.

        Raises:
            ValueError: If `reinit` is not a valid mode.

        """
        valid_reinit = {"finish_previous", "return_previous", "create_new", None}
        if reinit not in valid_reinit:
            raise ValueError(
                f"Invalid reinit value '{reinit}'. Must be one of: "
                f"{', '.join(valid_reinit - {None})} or None."
            )

        self._logger = logging.getLogger(self.name)
        self._project = project
        self._entity = entity
        self._base_run_name = run_name
        self._config = dict(config) if config is not None else {}
        self._reinit = reinit or "finish_previous"
        self._runs: Dict[str, Run] = {}

    def _get_or_create_run(self, scope: str) -> Run:
        """Return an existing W&B run for the given scope, or create a new one."""
        if scope in self._runs and self._runs[scope] is not None:
            return self._runs[scope]

        run_name = (
            self._base_run_name
            if scope == self.GLOBAL_SCOPE
            else f"{self._base_run_name}-{scope}" if self._base_run_name else scope
        )
        self._logger.debug("Opening new W&B run for scope '%s' (%s)", scope, run_name)
        kwargs: Dict[str, Any] = {
            "project": self._project,
            "entity": self._entity,
            "name": run_name,
            "config": self._config,
            "reinit": self._reinit,
        }
        run = wandb.init(**kwargs)
        self._runs[scope] = run
        return run

    def handle(self, event: Any) -> None:
        """Handle a Goggles event and forward it to W&B.

        Args:
            event (Event): Structured event containing kind, scope, and payload.

        Raises:
            ValueError: If a metric event's payload is not a mapping.

        """
        scope = getattr(event, "scope", None) or self.GLOBAL_SCOPE
        run = self._get_or_create_run(scope)

        kind = getattr(event, "kind", None) or "metric"
        step = getattr(event, "step", None)
        payload = getattr(event, "payload", None)

        if kind == "metric":
            if not isinstance(payload, Mapping):
                raise ValueError(
                    "Metric event payload must be a mapping of name→value."
                )
            run.log(dict(payload), step=step)
            self._logger.debug(
                "Logged metrics to W&B (scope=%s): %s", scope, list(payload.keys())
            )
            return

        if kind in {"image", "video"}:
            items = (
                payload.items() if isinstance(payload, Mapping) else [("data", payload)]
            )
            logs: Dict[str, Any] = {}
            for name, value in items:
                if value is None:
                    self._logger.warning(
                        "Skipping %s '%s' with None payload (scope=%s).",
                        kind,
                        name,
                        scope,
                    )
                    continue
                if kind == "image":
                    logs[name] = wandb.Image(value)
                else:
                    logs[name] = wandb.Video(value, fps=20, format="mp4")
            run.log(logs, step=step)
            self._logger.debug(
                "Logged %s(s) to W&B (scope=%s): %s", kind, scope, list(logs.keys())
            )
            return

        if kind == "artifact":
            if not isinstance(payload, Mapping):
                self._logger.warning(
                    "Artifact payload must be a mapping; got %r", type(payload)
                )
                return
            path = payload.get("path")
            name = payload.get("name", "artifact")
            art_type = payload.get("type", "misc")
            if not isinstance(path, str):
                self._logger.warning("Artifact missing valid 'path' field; skipping.")
                return
            artifact = wandb.Artifact(name=name, type=art_type)
            artifact.add_file(path)
            run.log_artifact(artifact)
            self._logger.debug(
                "Uploaded artifact to W&B (scope=%s): %s (%s)", scope, name, path
            )
            return

        self._logger.warning("Unsupported event kind: %s", kind)

    def close(self) -> None:
        """Finish all active W&B runs."""
        for scope, run in list(self._runs.items()):
            if run is not None:
                self._logger.debug("Closing W&B run for scope '%s'.", scope)
                wandb.finish()
                self._runs[scope] = None
        self._runs.clear()
