"""WandB integration handler for Goggles logging framework."""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Dict, FrozenSet, Literal, Mapping, Optional, Self, Set

import wandb

Run = Any  # wandb.sdk.wandb_run.Run


class WandBHandler:
    """Forward Goggles events to W&B runs (supports concurrent scopes).

    Each scope corresponds to a distinct W&B run that remains active until
    explicitly closed. Compatible with the `Handler` protocol used by the
    EventBus.
    """

    name: ClassVar[str] = "wandb"
    capabilities: ClassVar[FrozenSet[str]] = frozenset(
        {"metric", "image", "video", "artifact"}
    )
    GLOBAL_SCOPE: ClassVar[str] = "global"

    def __init__(
        self,
        project: Optional[str] = None,
        entity: Optional[str] = None,
        run_name: Optional[str] = None,
        config: Optional[Mapping[str, Any]] = None,
        reinit: Optional[
            Literal["default", "return_previous", "finish_previous", "create_new"]
        ] = "finish_previous",
    ) -> None:
        """Initialize the W&B handler."""
        valid_reinit = {"finish_previous", "return_previous", "create_new", "default"}
        if reinit not in valid_reinit:
            raise ValueError(
                f"Invalid reinit value '{reinit}'. Must be one of: "
                f"{', '.join(valid_reinit)}."
            )

        self._logger = logging.getLogger(self.name)
        self._logger.propagate = True
        self._project = project
        self._entity = entity
        self._base_run_name = run_name
        self._config = dict(config) if config is not None else {}
        self._reinit = reinit or "finish_previous"
        self._runs: dict[str, Run] = {}
        self._wandb_run: Optional[Run] = None
        self._current_scope: Optional[str] = None

    # -------------------------------------------------------------------------
    # Protocol methods
    # -------------------------------------------------------------------------

    def can_handle(self, kind: str) -> bool:
        """Return True if the handler supports this event kind."""
        return kind in self.capabilities

    def open(self) -> None:
        """Initialize the global W&B run."""
        if self._wandb_run is not None:
            self._logger.debug("W&B run already open; skipping reinit.")
            return
        self._wandb_run = wandb.init(
            project=self._project,
            entity=self._entity,
            name=self._base_run_name,
            config=self._config,
            reinit=self._reinit,  # type: ignore
        )
        self._runs[self.GLOBAL_SCOPE] = self._wandb_run
        self._current_scope = self.GLOBAL_SCOPE
        self._logger.debug("Opened W&B run '%s'.", self._base_run_name)

    def handle(self, event: Any) -> None:
        """Handle a Goggles event and forward it to W&B."""
        scope = getattr(event, "scope", None) or self.GLOBAL_SCOPE
        kind = getattr(event, "kind", None) or "metric"
        step = getattr(event, "step", None)
        payload = getattr(event, "payload", None)

        run = self._wandb_run or self._runs.get(scope)
        if run is None:
            if scope == self.GLOBAL_SCOPE:
                self._logger.warning("W&B run not opened; ignoring event.")
                return
            run = self._get_or_create_run(scope)

        if kind == "metric":
            if not isinstance(payload, Mapping):
                raise ValueError(
                    "Metric event payload must be a mapping of name→value."
                )
            wandb.log(dict(payload), step=step)
            self._logger.debug("Logged metrics: %s", list(payload.keys()))
            return

        if kind in {"image", "video"}:
            items = (
                payload.items() if isinstance(payload, Mapping) else [("data", payload)]
            )
            logs = {}
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
            self._logger.debug("Uploaded artifact: %s (%s)", name, path)
            return

        self._logger.warning("Unsupported event kind: %s", kind)

    def close(self) -> None:
        """Finish all active W&B runs."""
        if self._wandb_run is not None:
            wandb.finish()
            self._wandb_run = None
        for scope, run in list(self._runs.items()):
            if run is not None:
                wandb.finish()
                self._runs[scope] = None
        self._runs.clear()
        self._logger.debug("All W&B runs closed.")

    def to_dict(self) -> Dict:
        """Serialize the handler for attachment."""
        return {
            "cls": self.__class__.__name__,
            "data": {
                "project": self._project,
                "entity": self._entity,
                "run_name": self._base_run_name,
                "config": self._config,
                "reinit": self._reinit,
            },
        }

    @classmethod
    def from_dict(cls, serialized: Dict) -> Self:
        """De-serialize the handler from its dictionary representation."""
        data = serialized.get("data", {})
        return cls(
            project=data.get("project"),
            entity=data.get("entity"),
            run_name=data.get("run_name"),
            config=data.get("config"),
            reinit=data.get("reinit"),
        )

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _get_or_create_run(self, scope: str) -> Run:
        if scope in self._runs and self._runs[scope] is not None:
            return self._runs[scope]
        run_name = (
            self._base_run_name
            if scope == self.GLOBAL_SCOPE
            else f"{self._base_run_name}-{scope}" if self._base_run_name else scope
        )
        self._logger.debug("Opening new W&B run for scope '%s' (%s)", scope, run_name)
        run = wandb.init(
            project=self._project,
            entity=self._entity,
            name=run_name,
            config=self._config,
            reinit=self._reinit,  # type: ignore
        )
        self._runs[scope] = run
        return run
