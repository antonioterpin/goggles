import glob
import importlib
import inspect
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import wandb as _real_wandb
from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal.datastore import DataStore

import goggles._core.integrations.wandb as wandb_module
from goggles._core.integrations.wandb import WandBHandler


def _moviepy_video_importable() -> bool:
    """Probe whether the moviepy video pipeline is loadable.

    wandb's video path lazy-imports ``moviepy.video.VideoClip``. moviepy <2
    ships a ``config_defaults.py`` whose Windows path constants raise a
    SyntaxWarning on newer Python versions; under pytest's
    ``filterwarnings = ["error"]`` config the warning is upgraded to a
    SyntaxError at import time. Try-importing here lets the test suite
    skip gracefully on every (Python, moviepy, OS) combination that has
    the issue rather than enumerating versions.

    Returns:
        ``True`` if ``moviepy.video.VideoClip`` imports cleanly,
        ``False`` if any exception (including ``SyntaxError``) is raised.
    """
    try:
        importlib.import_module("moviepy.video.VideoClip")
    except Exception:
        return False
    return True


_MOVIEPY_USABLE = _moviepy_video_importable()


def _capture_logger_messages(
    logger: logging.Logger,
) -> tuple[list[str], logging.Handler]:
    messages: list[str] = []

    class _MessageCollector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            messages.append(record.getMessage())

    collector = _MessageCollector()
    logger.addHandler(collector)
    return messages, collector


@pytest.fixture
def mock_wandb(monkeypatch):
    mock = MagicMock()
    mock.init.__signature__ = inspect.signature(_real_wandb.init)
    monkeypatch.setattr(wandb_module, "wandb", mock)
    return mock


def make_event(
    kind="metric",
    scope="global",
    payload=None,
    step: int | None = 0,
):
    return SimpleNamespace(kind=kind, scope=scope, payload=payload, step=step)


@pytest.mark.parametrize(
    "reinit", ["finish_previous", "return_previous", "create_new", "default"]
)
def test_open_is_noop(mock_wandb, reinit):
    handler = WandBHandler(
        project="proj", entity="ent", run_name="name", reinit=reinit
    )
    handler.open()
    mock_wandb.init.assert_not_called()


def test_open_idempotent(mock_wandb):
    handler = WandBHandler(project="p")
    handler.open()
    handler.open()
    mock_wandb.init.assert_not_called()


def test_can_handle_supported_kinds():
    h = WandBHandler()
    for kind in [
        "metric",
        "image",
        "video",
        "artifact",
        "vector_field",
        "trajectories",
        "histogram",
    ]:
        assert h.can_handle(kind), f"WandBHandler should handle '{kind}' events"
    assert not h.can_handle("log"), (
        "WandBHandler should not handle 'log' events by default"
    )


def test_handle_metric_raises_if_not_mapping(mock_wandb):
    h = WandBHandler()
    event = make_event(kind="metric", payload=[1, 2])
    with pytest.raises(ValueError):
        h.handle(event)


def test_handle_unsupported_kind_warns(mock_wandb):
    h = WandBHandler()
    event = make_event(kind="nonsense", payload={})

    messages, collector = _capture_logger_messages(h._logger)
    try:
        h.handle(event)
    finally:
        h._logger.removeHandler(collector)
    assert any("unsupported" in msg.lower() for msg in messages), (
        "Should log a warning for unsupported event kind"
    )


def test_get_or_create_run_creates_new(mock_wandb):
    h = WandBHandler(project="proj", entity="ent", run_name="base")
    run = h._get_or_create_run("scope1", extra_config={})
    mock_wandb.init.assert_called_once_with(
        project="proj",
        entity="ent",
        name="base-scope1",
        config={"scope": "scope1"},
        group=None,
        tags=None,
        reinit="create_new",
    )
    assert h._runs["scope1"] == run, (
        "Run should be cached under the given scope"
    )


def test_get_or_create_run_forwards_group_and_tags(mock_wandb):
    """``group`` and ``tags`` from __init__ reach the wandb.init call.

    Args:
        mock_wandb: Patched ``wandb`` module fixture.
    """
    h = WandBHandler(
        project="proj",
        run_name="base",
        group="cohort-a",
        tags=("baseline", "ablation"),
    )
    h._get_or_create_run("global", extra_config={})

    init_kwargs = mock_wandb.init.call_args.kwargs
    assert init_kwargs["group"] == "cohort-a", (
        "group passed at init should reach wandb.init"
    )
    assert list(init_kwargs["tags"]) == ["baseline", "ablation"], (
        "tags passed at init should reach wandb.init as a list"
    )


def test_get_or_create_run_omits_tags_when_unset(mock_wandb):
    """``tags`` defaults to None — not an empty list — at the wandb.init call.

    Lets callers/W&B distinguish "no tags configured" from "explicitly empty".

    Args:
        mock_wandb: Patched ``wandb`` module fixture.
    """
    h = WandBHandler(project="proj", run_name="base")
    h._get_or_create_run("global", extra_config={})

    assert mock_wandb.init.call_args.kwargs["tags"] is None


def test_tags_roundtrip_through_to_dict(mock_wandb):
    """``group`` and ``tags`` survive to_dict/from_dict serialization.

    Args:
        mock_wandb: Patched ``wandb`` module fixture.
    """
    h = WandBHandler(
        project="proj",
        run_name="base",
        group="cohort-a",
        tags=["baseline"],
    )
    serialized = h.to_dict()["data"]
    assert serialized["group"] == "cohort-a"
    assert serialized["tags"] == ["baseline"]

    restored = WandBHandler.from_dict(serialized)
    restored._get_or_create_run("global", extra_config={})
    init_kwargs = mock_wandb.init.call_args.kwargs
    assert init_kwargs["group"] == "cohort-a"
    assert list(init_kwargs["tags"]) == ["baseline"]


def test_tags_must_be_sequence_of_strings(mock_wandb):
    """Bare ``str`` and non-string items are rejected at construction.

    A ``str`` matches ``Sequence[str]`` under the type system but would
    iterate character by character at runtime, so we reject it explicitly.

    Args:
        mock_wandb: Patched ``wandb`` module fixture.
    """
    with pytest.raises(TypeError, match="tags"):
        WandBHandler(project="proj", tags="single-tag")
    with pytest.raises(TypeError, match="tags"):
        WandBHandler(project="proj", tags=[1, 2])  # pyright: ignore[reportArgumentType]


def test_wandb_init_kwargs_forwarded_on_first_event(mock_wandb):
    init_kwargs = {"save_code": True, "settings": {"code_dir": "."}}
    h = WandBHandler(
        project="proj",
        entity="ent",
        run_name="base",
        wandb_init_kwargs=init_kwargs,
    )
    event = make_event(payload={"loss": 1.0})

    h.handle(event)

    mock_wandb.init.assert_called_once_with(
        project="proj",
        entity="ent",
        name="base",
        config={"scope": "global"},
        group=None,
        tags=None,
        reinit="create_new",
        save_code=True,
        settings={"code_dir": "."},
    )


def test_wandb_init_kwargs_unknown_key_raises(mock_wandb):
    with pytest.raises(ValueError, match=r"Unknown wandb.init.*unknown_key"):
        WandBHandler(wandb_init_kwargs={"unknown_key": True})


def test_wandb_init_kwargs_reserved_key_raises(mock_wandb):
    with pytest.raises(ValueError, match=r"handler-owned.*project"):
        WandBHandler(wandb_init_kwargs={"project": "override"})
    with pytest.raises(ValueError, match=r"handler-owned.*tags"):
        WandBHandler(wandb_init_kwargs={"tags": ["x"]})


def test_wandb_init_kwargs_roundtrip_serialization(mock_wandb):
    init_kwargs = {"save_code": True, "settings": {"code_dir": "."}}
    h = WandBHandler(project="proj", wandb_init_kwargs=init_kwargs)

    serialized = h.to_dict()
    rebuilt = WandBHandler.from_dict(serialized)

    assert serialized["data"]["wandb_init_kwargs"] == init_kwargs
    assert rebuilt.to_dict()["data"]["wandb_init_kwargs"] == init_kwargs

    event = make_event(payload={"loss": 1.0})
    rebuilt.handle(event)

    mock_wandb.init.assert_called_once_with(
        project="proj",
        entity=None,
        name="run-global",
        config={"scope": "global"},
        group=None,
        tags=None,
        reinit="create_new",
        save_code=True,
        settings={"code_dir": "."},
    )


def test_handle_artifact_uploads_file(mock_wandb, tmp_path):
    artifact_file = tmp_path / "random_artifact.npy"
    artifact_file.write_bytes(b"dummy")

    h = WandBHandler(project="proj")
    event = SimpleNamespace(
        kind="artifact",
        scope="global",
        payload={
            "path": str(artifact_file),
            "name": "random_artifact",
            "type": "misc",
        },
        step=1,
        extra={},
    )

    h.handle(event)

    mock_wandb.Artifact.assert_called_once_with(
        name="random_artifact", type="misc", metadata={}
    )
    mock_wandb.Artifact.return_value.add_file.assert_called_once_with(
        str(artifact_file)
    )
    mock_wandb.init.return_value.log_artifact.assert_called_once_with(
        mock_wandb.Artifact.return_value
    )


def test_handle_artifact_non_mapping_warns(mock_wandb):
    h = WandBHandler(project="proj")
    event = SimpleNamespace(
        kind="artifact",
        scope="global",
        payload=np.zeros((4, 4)),
        step=0,
        extra={},
    )

    messages, collector = _capture_logger_messages(h._logger)
    try:
        h.handle(event)
    finally:
        h._logger.removeHandler(collector)

    assert any("must be a mapping" in m.lower() for m in messages), (
        "Should warn when artifact payload is not a mapping"
    )
    mock_wandb.Artifact.assert_not_called()


def test_handle_artifact_uploads_directory(mock_wandb, tmp_path):
    """Directory paths are uploaded recursively via ``Artifact.add_dir``.

    Multi-file artifacts (such as Orbax model checkpoints) live in a
    directory tree on disk. The WandB integration must dispatch to
    ``add_dir`` instead of ``add_file`` so the full tree is uploaded as
    one artifact.
    """
    ckpt_dir = tmp_path / "checkpoint_step_42"
    ckpt_dir.mkdir()
    (ckpt_dir / "params.msgpack").write_bytes(b"weights")
    (ckpt_dir / "opt_state.msgpack").write_bytes(b"opt")

    h = WandBHandler(project="proj")
    event = SimpleNamespace(
        kind="artifact",
        scope="global",
        payload={
            "path": str(ckpt_dir),
            "name": "model_checkpoint",
            "type": "checkpoint",
        },
        step=42,
        extra={},
    )

    h.handle(event)

    mock_wandb.Artifact.assert_called_once_with(
        name="model_checkpoint", type="checkpoint", metadata={}
    )
    mock_wandb.Artifact.return_value.add_dir.assert_called_once_with(
        str(ckpt_dir)
    )
    mock_wandb.Artifact.return_value.add_file.assert_not_called()
    mock_wandb.init.return_value.log_artifact.assert_called_once()


def test_handle_artifact_forwards_aliases(mock_wandb, tmp_path):
    """Aliases in the payload reach ``run.log_artifact`` verbatim.

    WandB exposes artifact aliases as the mechanism for marking the
    "best" or "latest" version of a checkpoint within an artifact
    collection; without forwarding them, callers cannot tag uploads.
    """
    artifact_file = tmp_path / "ckpt.bin"
    artifact_file.write_bytes(b"x")

    h = WandBHandler(project="proj")
    event = SimpleNamespace(
        kind="artifact",
        scope="global",
        payload={
            "path": str(artifact_file),
            "name": "model_checkpoint",
            "type": "checkpoint",
            "aliases": ["best", "step-42"],
        },
        step=42,
        extra={},
    )

    h.handle(event)

    mock_wandb.init.return_value.log_artifact.assert_called_once_with(
        mock_wandb.Artifact.return_value, aliases=["best", "step-42"]
    )


@pytest.mark.parametrize(
    "bad_aliases",
    [
        "best",  # str: matches Sequence but iterates char-by-char
        b"best",  # bytes: matches Sequence but iterates byte-by-byte
        42,  # scalar: not a Sequence at all
        ["best", 123],  # list with non-string element
    ],
    ids=["bare-str", "bare-bytes", "scalar", "non-string-element"],
)
def test_handle_artifact_invalid_aliases_warn_and_drop(
    mock_wandb, tmp_path, bad_aliases
):
    """Invalid ``aliases`` payloads warn and the upload proceeds unaliased.

    Forwarding a ``str``/``bytes`` would be silently iterated and produce
    nonsense aliases like ``['b','e','s','t']``; a scalar would crash
    ``run.log_artifact``; non-string elements would fail server-side. The
    guard turns each into a noisy no-op so the upload still goes
    through.
    """
    artifact_file = tmp_path / "ckpt.bin"
    artifact_file.write_bytes(b"x")

    h = WandBHandler(project="proj")
    event = SimpleNamespace(
        kind="artifact",
        scope="global",
        payload={
            "path": str(artifact_file),
            "name": "ckpt",
            "type": "checkpoint",
            "aliases": bad_aliases,
        },
        step=0,
        extra={},
    )

    messages, collector = _capture_logger_messages(h._logger)
    try:
        h.handle(event)
    finally:
        h._logger.removeHandler(collector)

    assert any("aliases" in m.lower() for m in messages), (
        "Should warn when aliases payload is invalid"
    )
    mock_wandb.init.return_value.log_artifact.assert_called_once_with(
        mock_wandb.Artifact.return_value
    )


def test_handle_artifact_missing_path_warns(mock_wandb, tmp_path):
    """A non-existent ``path`` skips the upload with a warning.

    Without this guard, ``Artifact.add_file`` raises deep inside the
    handler thread and crashes the dispatch loop. Detecting it early
    surfaces a clear message and keeps the handler running.
    """
    missing = tmp_path / "does_not_exist"
    h = WandBHandler(project="proj")
    event = SimpleNamespace(
        kind="artifact",
        scope="global",
        payload={
            "path": str(missing),
            "name": "ckpt",
            "type": "checkpoint",
        },
        step=0,
        extra={},
    )

    messages, collector = _capture_logger_messages(h._logger)
    try:
        h.handle(event)
    finally:
        h._logger.removeHandler(collector)

    assert any("does not exist" in m.lower() for m in messages), (
        "Should warn when artifact path is missing"
    )
    mock_wandb.Artifact.assert_not_called()


def test_handle_vector_field_logs_image(mock_wandb, monkeypatch):
    h = WandBHandler(project="proj")
    event = SimpleNamespace(
        kind="vector_field",
        scope="global",
        payload=np.zeros((16, 16, 2), dtype=np.float32),
        step=3,
        extra={
            "name": "flow",
            "mode": "vorticity",
            "add_colorbar": True,
            "tag": "viz",
        },
    )

    mocked_image = np.zeros((32, 32, 3), dtype=np.uint8)
    render_mock = MagicMock(return_value=mocked_image)
    monkeypatch.setattr(
        wandb_module, "create_numpy_vector_field_visualization", render_mock
    )

    h.handle(event)
    # Same-step events are buffered until a step change or close(); flush.
    h.close()

    run = mock_wandb.init.return_value
    render_mock.assert_called_once_with(
        event.payload,
        mode="vorticity",
        add_colorbar=True,
    )
    mock_wandb.Image.assert_called_once_with(mocked_image)
    run.log.assert_called_once()
    logged_payload = run.log.call_args[0][0]
    assert "flow" in logged_payload, (
        "Logged payload should contain the field name as key"
    )
    assert logged_payload["tag"] == "viz", (
        "Logged payload should include extra fields"
    )
    assert run.log.call_args.kwargs["step"] == 3, (
        "Logged payload should include the event step"
    )


@pytest.mark.parametrize("dim", [2, 3], ids=["2d", "3d"])
def test_handle_trajectories_logs_plotly(mock_wandb, monkeypatch, dim):
    h = WandBHandler(project="proj")
    payload = np.random.randn(4, 8, dim).astype(np.float32)
    event = SimpleNamespace(
        kind="trajectories",
        scope="global",
        payload=payload,
        step=2,
        extra={"name": "trails", "tag": "viz"},
    )

    mocked_fig = MagicMock(name="plotly_figure")
    render_mock = MagicMock(return_value=mocked_fig)
    monkeypatch.setattr(
        wandb_module, "create_plotly_trajectories_figure", render_mock
    )

    h.handle(event)
    # Same-step events are buffered until a step change or close(); flush.
    h.close()

    render_mock.assert_called_once()
    np.testing.assert_array_equal(render_mock.call_args[0][0], payload)
    mock_wandb.Plotly.assert_called_once_with(mocked_fig)
    run = mock_wandb.init.return_value
    run.log.assert_called_once()
    logged_payload = run.log.call_args[0][0]
    assert "trails" in logged_payload
    assert logged_payload["tag"] == "viz"
    assert run.log.call_args.kwargs["step"] == 2


def test_handle_trajectories_bad_payload_warns(mock_wandb, monkeypatch):
    h = WandBHandler(project="proj")
    # 2D array — not (N, L, dim) — renderer will raise
    payload = np.zeros((3, 4), dtype=np.float32)
    event = SimpleNamespace(
        kind="trajectories",
        scope="global",
        payload=payload,
        step=0,
        extra={},
    )

    messages, collector = _capture_logger_messages(h._logger)
    try:
        h.handle(event)
    finally:
        h._logger.removeHandler(collector)

    assert any("trajectories" in m.lower() for m in messages)
    mock_wandb.Plotly.assert_not_called()


def test_handle_vector_field_unknown_mode_warns_and_skips(mock_wandb):
    h = WandBHandler(project="proj")
    event = SimpleNamespace(
        kind="vector_field",
        scope="global",
        payload=np.zeros((16, 16, 2), dtype=np.float32),
        step=0,
        extra={"mode": "unknown"},
    )

    messages, collector = _capture_logger_messages(h._logger)
    try:
        h.handle(event)
    finally:
        h._logger.removeHandler(collector)

    assert any(
        "unknown vector field visualization mode" in m.lower() for m in messages
    ), "Should log a warning about the unknown visualization mode"
    run = mock_wandb.init.return_value
    run.log.assert_not_called()


@pytest.mark.parametrize(
    "shape, expected_channels",
    [((5, 8, 12, 1), 3), ((5, 8, 12, 3), 3), ((5, 8, 12, 4), 4)],
    ids=["channels_last_gray", "channels_last_rgb", "channels_last_rgba"],
)
def test_prepare_video_channels_last(shape, expected_channels):
    h = WandBHandler(project="proj")
    F, H, W, _ = shape
    value = np.full(shape, 128, dtype=np.uint8)
    out = h._prepare_video_for_wandb(value)
    assert out.shape == (F, expected_channels, H, W), (
        f"Expected (F, {expected_channels}, H, W) for input {shape},"
        f" got {out.shape}"
    )


def test_prepare_video_channels_first_preserved():
    h = WandBHandler(project="proj")
    F, C, H, W = 5, 3, 8, 12
    value = np.full((F, C, H, W), 128, dtype=np.uint8)
    out = h._prepare_video_for_wandb(value)
    assert out.shape == (F, 3, H, W)


@pytest.mark.parametrize("W", [1, 3, 4])
def test_prepare_video_channels_first_ambiguous_width_preserved(W):
    h = WandBHandler(project="proj")
    F, C, H = 5, 3, 8
    value = np.full((F, C, H, W), 128, dtype=np.uint8)
    out = h._prepare_video_for_wandb(value)
    assert out.shape == (F, C, H, W), (
        "Channels-first input with W in {1, 3, 4} must not be "
        "misdetected as channels-last"
    )


def test_prepare_video_invalid_ndim_raises():
    h = WandBHandler(project="proj")
    with pytest.raises(ValueError, match="invalid shape"):
        h._prepare_video_for_wandb(np.zeros((2, 3), dtype=np.uint8))


def test_prepare_video_4d_neither_channel_axis_raises():
    # 4D input with no plausible channel dim must raise rather than guess.
    h = WandBHandler(project="proj")
    # F=5, axis-1 size 7, axis-(-1) size 9 — neither is 1/3/4.
    bad = np.zeros((5, 7, 8, 9), dtype=np.uint8)
    with pytest.raises(ValueError, match="expected channel dim"):
        h._prepare_video_for_wandb(bad)


def test_prepare_video_5d_bad_channel_dim_raises():
    # 5D (F,T,C,H,W) with axis-2 not in {1,3,4} must raise.
    h = WandBHandler(project="proj")
    bad = np.zeros((2, 4, 5, 8, 12), dtype=np.uint8)  # C=5
    with pytest.raises(ValueError, match="expected channel dim"):
        h._prepare_video_for_wandb(bad)


@pytest.mark.parametrize("c", [1, 3, 4])
def test_prepare_video_5d_valid_channel_dim_passes(c):
    # 5D (F,T,C,H,W) with C in {1,3,4} passes through (grayscale upcast).
    h = WandBHandler(project="proj")
    value = np.zeros((2, 4, c, 8, 12), dtype=np.uint8)
    out = h._prepare_video_for_wandb(value)
    expected_c = 3 if c == 1 else c
    assert out.shape == (2, 4, expected_c, 8, 12)


@pytest.mark.parametrize(
    "kind, payload",
    [
        ("image", np.zeros((4, 4, 3), dtype=np.uint8)),
        ("video", np.zeros((2, 4, 4, 3), dtype=np.uint8)),
        ("histogram", np.zeros(32, dtype=np.float32)),
    ],
    ids=["image", "video", "histogram"],
)
def test_handle_does_not_mutate_event_extra(mock_wandb, kind, payload):
    h = WandBHandler(project="proj")
    extra = {
        "name": "thing",
        "format": "png",
        "fps": 10,
        "config_wandb": {"x": 1},
        "tag": "viz",
    }
    snapshot = dict(extra)
    event = SimpleNamespace(
        kind=kind,
        scope="global",
        payload=payload,
        step=0,
        extra=extra,
    )
    h.handle(event)
    assert extra == snapshot, (
        f"Handler mutated event.extra for kind={kind!r}: "
        f"before={snapshot}, after={extra}"
    )


def test_prepare_video_channels_first_grayscale_repeated():
    h = WandBHandler(project="proj")
    F, H, W = 5, 8, 12
    value = np.full((F, 1, H, W), 128, dtype=np.uint8)
    out = h._prepare_video_for_wandb(value)
    assert out.shape == (F, 3, H, W)


# -------------------------------------------------------------------------
# Monotonic-step contract
# -------------------------------------------------------------------------


def test_handle_drops_backward_step_with_warning(mock_wandb):
    """Backward step within a scope is dropped before any wandb call."""
    h = WandBHandler(project="proj")
    e1 = make_event(kind="metric", payload={"loss": 1.0}, step=10)
    e2 = make_event(kind="metric", payload={"loss": 0.9}, step=5)

    messages, collector = _capture_logger_messages(h._logger)
    try:
        h.handle(e1)
        h.handle(e2)
        # Flush the pending step=10 buffer to assert against its single commit.
        h.close()
    finally:
        h._logger.removeHandler(collector)

    run = mock_wandb.init.return_value
    assert run.log.call_count == 1, (
        "Backward-step event must not reach run.log; "
        f"saw {run.log.call_count} calls"
    )
    assert run.log.call_args.kwargs["step"] == 10, (
        "Only the forward-step event should have been logged"
    )
    assert any(
        "out-of-order" in m.lower() and "step=5" in m for m in messages
    ), "Should warn that step=5 regressed"


def test_handle_allows_equal_and_forward_steps(mock_wandb):
    """Same-step events coalesce into one commit; step changes flush.

    Logging at step 3, 3, 4 and then closing yields exactly two commits:
    one merged ``{a, b}`` row at step=3 and one ``{c}`` row at step=4.
    Coalescing same-step writes into a single ``run.log`` is the fix for
    issue #177 (W&B not displaying logged data on same-step keys).

    Args:
        mock_wandb: Patched ``wandb`` module fixture.
    """
    h = WandBHandler(project="proj")
    h.handle(make_event(kind="metric", payload={"a": 1}, step=3))
    h.handle(make_event(kind="metric", payload={"b": 2}, step=3))  # equal
    h.handle(make_event(kind="metric", payload={"c": 3}, step=4))  # forward
    h.close()  # flush pending step=4

    run = mock_wandb.init.return_value
    assert run.log.call_count == 2, (
        f"Expected 2 wandb.log calls (one per step); saw {run.log.call_count}"
    )
    calls = [(c.args[0], c.kwargs.get("step")) for c in run.log.call_args_list]
    assert calls[0] == ({"a": 1, "b": 2}, 3), (
        f"step=3 commit should batch keys 'a' and 'b'; got {calls[0]}"
    )
    assert calls[1] == ({"c": 3}, 4), (
        f"step=4 commit should carry only 'c'; got {calls[1]}"
    )


def test_handle_tracks_step_per_scope(mock_wandb):
    """Step monotonicity is tracked per scope, not globally."""
    h = WandBHandler(project="proj")
    h.handle(
        make_event(kind="metric", payload={"x": 1}, scope="train", step=10)
    )
    # Lower step on a different scope must still be forwarded.
    h.handle(make_event(kind="metric", payload={"x": 2}, scope="eval", step=1))
    h.close()  # flush both per-scope buffers

    # mock_wandb.init returns the same MagicMock for every scope, so both
    # scopes share the same .log mock; assert on total calls instead.
    run = mock_wandb.init.return_value
    assert run.log.call_count == 2, (
        f"both scopes' events should have been forwarded; "
        f"saw {run.log.call_count} log calls"
    )


def test_handle_does_not_guard_when_step_is_none(mock_wandb):
    """Events with step=None never trip the guard, even after a regression."""
    h = WandBHandler(project="proj")
    h.handle(make_event(kind="metric", payload={"a": 1}, step=10))
    # step=None must not be flagged.
    h.handle(make_event(kind="metric", payload={"b": 2}, step=None))

    run = mock_wandb.init.return_value
    assert run.log.call_count == 2, (
        f"step=None must not be dropped; saw {run.log.call_count} calls"
    )


def test_handle_artifact_bypasses_step_guard(mock_wandb, tmp_path):
    """Artifacts are step-less and must bypass the monotonic-step check."""
    artifact_file = tmp_path / "a.npy"
    artifact_file.write_bytes(b"x")
    h = WandBHandler(project="proj")
    # Establish a high-water mark on this scope.
    h.handle(make_event(kind="metric", payload={"loss": 1.0}, step=100))

    artifact_event = SimpleNamespace(
        kind="artifact",
        scope="global",
        payload={"path": str(artifact_file), "name": "a", "type": "misc"},
        step=0,  # would be a regression for non-artifact events
        extra={},
    )
    h.handle(artifact_event)

    # mock.assert_called_once() raises with its own diagnostic on failure;
    # this asserts the artifact was uploaded despite step < scope max.
    mock_wandb.Artifact.assert_called_once()


# ---------------------------------------------------------------------------
# Regression tests for issue #177 — W&B not displaying logged data.
#
# Same-step ``run.log`` calls were unreliable on the W&B cloud backend: a
# panel could register one key but show "No data available" for sibling
# keys logged at the same step. The fix coalesces same-step events into a
# single commit per (scope, step).
# ---------------------------------------------------------------------------


def test_same_step_events_batch_into_single_log_call(
    mock_wandb: MagicMock,
) -> None:
    """Three events at step=100 produce one ``run.log`` once step changes.

    Args:
        mock_wandb: Patched ``wandb`` module fixture.
    """
    h = WandBHandler(project="p")
    img1 = np.zeros((4, 4, 3), dtype=np.uint8)
    img2 = np.zeros((4, 4, 4), dtype=np.uint8)
    h.handle(
        SimpleNamespace(
            kind="image",
            scope="global",
            payload=img1,
            step=100,
            extra={"name": "RGB"},
        )
    )
    h.handle(
        SimpleNamespace(
            kind="image",
            scope="global",
            payload=img2,
            step=100,
            extra={"name": "RGBA"},
        )
    )
    h.handle(make_event(kind="metric", payload={"acc": 0.9}, step=100))

    run = mock_wandb.init.return_value
    run.log.assert_not_called()  # all three are buffered at step=100

    # Step change forces a single batched commit for the previous step.
    h.handle(make_event(kind="metric", payload={"acc": 0.95}, step=101))
    assert run.log.call_count == 1, (
        "Same-step events must batch into ONE run.log call"
    )
    logged, kwargs = run.log.call_args.args[0], run.log.call_args.kwargs
    assert kwargs["step"] == 100
    got = sorted(logged)
    assert set(logged.keys()) == {"RGB", "RGBA", "acc"}, (
        f"All same-step keys must land in the single commit; got {got}"
    )


def test_close_flushes_pending_buffer(mock_wandb: MagicMock) -> None:
    """``close()`` flushes the pending step before finishing the run.

    Args:
        mock_wandb: Patched ``wandb`` module fixture.
    """
    h = WandBHandler(project="p")
    h.handle(make_event(kind="metric", payload={"loss": 0.1}, step=42))
    run = mock_wandb.init.return_value
    run.log.assert_not_called()  # buffered, not yet committed

    h.close()
    run.log.assert_called_once_with({"loss": 0.1}, step=42)
    run.finish.assert_called()


def test_step_none_bypasses_buffer_to_preserve_autoincrement(
    mock_wandb: MagicMock,
) -> None:
    """``step=None`` flushes any pending buffer and commits immediately.

    Wandb auto-increments the internal step for step-less ``run.log`` calls,
    so callers relying on auto-increment must not see those events
    coalesced into a single row.

    Args:
        mock_wandb: Patched ``wandb`` module fixture.
    """
    h = WandBHandler(project="p")
    h.handle(make_event(kind="metric", payload={"a": 1}, step=5))
    h.handle(make_event(kind="metric", payload={"b": 2}, step=None))
    h.handle(make_event(kind="metric", payload={"c": 3}, step=None))

    run = mock_wandb.init.return_value
    # 1 flush of step=5 + 2 immediate step-less commits == 3 calls.
    assert run.log.call_count == 3
    seen = [(c.args[0], c.kwargs.get("step")) for c in run.log.call_args_list]
    assert seen[0] == ({"a": 1}, 5)
    assert (
        seen[1][0] == {"b": 2}
        and "step" not in run.log.call_args_list[1].kwargs
    )
    assert (
        seen[2][0] == {"c": 3}
        and "step" not in run.log.call_args_list[2].kwargs
    )


@pytest.mark.slow
@pytest.mark.skipif(
    not _MOVIEPY_USABLE,
    reason=(
        "moviepy.video.VideoClip is not importable in this environment "
        "(commonly: moviepy<2 SyntaxWarning under Py3.12+ promoted to "
        "SyntaxError by pytest's filterwarnings=['error']). Reachable "
        "from wandb's video path. Tracked in #182."
    ),
)
def test_example_04_logs_all_keys_at_step_100_offline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end smoke against the real wandb client (offline mode).

    Drives the same flow as ``examples/04_wandb.py`` (RGB + RGBA images +
    video at step=100), then decodes the persisted ``.wandb`` proto and
    asserts every same-step key lands in the row at ``_step=100``.

    Args:
        tmp_path: Per-test temporary directory used as ``WANDB_DIR``.
        monkeypatch: pytest fixture for setting wandb env vars.
    """
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_SILENT", "true")

    h = WandBHandler(project="goggles_example_04_test", run_name="t")

    rgb = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
    rgba = np.random.randint(0, 255, (8, 8, 4), dtype=np.uint8)
    vid = np.random.randint(0, 255, (4, 3, 8, 8), dtype=np.uint8)

    for i in range(3):
        h.handle(make_event(kind="metric", payload={"acc": i}, step=i))
    h.handle(
        SimpleNamespace(
            kind="image",
            scope="global",
            payload=rgb,
            step=100,
            extra={"name": "Random image"},
        )
    )
    h.handle(
        SimpleNamespace(
            kind="image",
            scope="global",
            payload=rgba,
            step=100,
            extra={"name": "Random RGBA image"},
        )
    )
    h.handle(
        SimpleNamespace(
            kind="video",
            scope="global",
            payload=vid,
            step=100,
            extra={"name": "Random Video", "fps": 5, "format": "mp4"},
        )
    )
    h.handle(make_event(kind="metric", payload={"next": 1}, step=101))
    h.close()

    matches = [
        p
        for p in glob.glob(
            os.path.join(tmp_path, "**", "*.wandb"), recursive=True
        )
        if "latest-run" not in p
    ]
    assert matches, f"No .wandb proto under {tmp_path}"

    by_step: dict[int, set[str]] = {}
    ds = DataStore()
    ds.open_for_scan(matches[0])
    while True:
        rec_data = ds.scan_record()
        if rec_data is None:
            break
        rec = pb.Record()
        rec.ParseFromString(rec_data[1])
        kind = rec.WhichOneof("record_type")
        if kind not in ("history", "partial_history"):
            continue
        items = getattr(rec, kind).item
        step = None
        keys = set()
        for it in items:
            key = it.key or (it.nested_key[0] if it.nested_key else "")
            if key == "_step":
                step = int(it.value_json)
            elif not key.startswith("_"):
                keys.add(key)
        if step is not None:
            by_step.setdefault(step, set()).update(keys)

    assert 100 in by_step, f"step=100 row missing; rows={by_step}"
    assert {"Random image", "Random RGBA image", "Random Video"} <= by_step[
        100
    ], (
        "All three same-step media keys must land at step=100; "
        f"got {sorted(by_step[100])}"
    )
