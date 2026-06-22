"""Contract tests shared by the experiment-tracker handlers.

``WandBHandler`` and ``MLflowHandler`` implement the same EventBus
``Handler`` contract: identical capabilities, monotonic per-scope step
guarding (drop out-of-order, bypass for ``step is None`` and artifacts),
serialization round-tripping, and never mutating the shared
``event.extra``. These parametrized tests exercise that common contract
against both handlers so the two stay in lockstep; handler-specific
behaviour lives in ``test_wandb.py`` / ``test_mlflow.py``.
"""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

import goggles._core.integrations.mlflow as mlflow_mod
import goggles._core.integrations.wandb as wandb_mod

# Capabilities both experiment-tracker handlers advertise.
_SHARED_KINDS = [
    "metric",
    "image",
    "video",
    "artifact",
    "histogram",
    "vector_field",
    "trajectories",
]


def _build_wandb(monkeypatch):
    """Build a WandBHandler with the wandb SDK mocked out."""
    monkeypatch.setattr(wandb_mod, "wandb", MagicMock())
    return wandb_mod.WandBHandler(project="p", run_name="r")


def _build_mlflow(monkeypatch):
    """Build an MLflowHandler with the MlflowClient mocked out."""
    client = MagicMock()
    client.get_experiment_by_name.return_value = SimpleNamespace(
        experiment_id="0"
    )
    client.create_run.return_value = SimpleNamespace(
        info=SimpleNamespace(run_id="r")
    )
    monkeypatch.setattr(
        mlflow_mod, "MlflowClient", MagicMock(return_value=client)
    )
    return mlflow_mod.MLflowHandler(run_name="r")


@pytest.fixture(params=[_build_wandb, _build_mlflow], ids=["wandb", "mlflow"])
def handler(request, monkeypatch):
    """A constructed handler with its backend mocked (both trackers)."""
    return request.param(monkeypatch)


def _capture(logger: logging.Logger):
    messages: list[str] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            messages.append(record.getMessage())

    collector = _Collector()
    logger.addHandler(collector)
    return messages, collector


def _event(
    kind="metric",
    scope="global",
    payload=None,
    step: int | None = 0,
    extra=None,
):
    return SimpleNamespace(
        kind=kind,
        scope=scope,
        payload=payload,
        step=step,
        extra=extra or {},
        time=None,
    )


@pytest.mark.parametrize("kind", _SHARED_KINDS)
def test_can_handle_shared_kinds(handler, kind):
    assert handler.can_handle(kind), f"handler should support '{kind}'"


def test_can_handle_rejects_log(handler):
    assert not handler.can_handle("log"), "trackers should not handle 'log'"


def test_capabilities_match_shared_set(handler):
    assert handler.capabilities == frozenset(_SHARED_KINDS)


def test_backward_step_dropped_with_warning(handler):
    messages, collector = _capture(handler._logger)
    try:
        handler.handle(_event(payload={"loss": 1.0}, step=10))
        handler.handle(_event(payload={"loss": 0.9}, step=5))
    finally:
        handler._logger.removeHandler(collector)
    assert any(
        "out-of-order" in m.lower() and "step=5" in m for m in messages
    ), "a backward step must be dropped with an out-of-order warning"


def test_step_none_bypasses_guard(handler):
    messages, collector = _capture(handler._logger)
    try:
        handler.handle(_event(payload={"a": 1}, step=10))
        handler.handle(_event(payload={"b": 2}, step=None))
    finally:
        handler._logger.removeHandler(collector)
    assert not any("out-of-order" in m.lower() for m in messages), (
        "step=None has no ordering contract and must not be dropped"
    )


def test_artifact_bypasses_guard(handler, tmp_path):
    artifact_file = tmp_path / "a.npy"
    artifact_file.write_bytes(b"x")
    messages, collector = _capture(handler._logger)
    try:
        handler.handle(_event(payload={"loss": 1.0}, step=100))
        handler.handle(
            _event(
                kind="artifact",
                payload={"path": str(artifact_file), "name": "a"},
                step=0,  # a regression for non-artifact events
            )
        )
    finally:
        handler._logger.removeHandler(collector)
    assert not any("out-of-order" in m.lower() for m in messages), (
        "artifacts are step-less and must bypass the monotonic-step guard"
    )


def test_does_not_mutate_event_extra(handler):
    extra = {"name": "img", "tag": "x"}
    snapshot = dict(extra)
    handler.handle(
        _event(
            kind="image",
            payload=np.zeros((4, 4, 3), dtype=np.uint8),
            step=0,
            extra=extra,
        )
    )
    assert extra == snapshot, "handler must not mutate the shared event.extra"


def test_to_dict_from_dict_roundtrip(handler):
    data = handler.to_dict()
    assert data["cls"] == type(handler).__name__
    assert "data" in data
    rebuilt = type(handler).from_dict(data["data"])
    assert type(rebuilt) is type(handler)
    assert rebuilt.name == handler.name
