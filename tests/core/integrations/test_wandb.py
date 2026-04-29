import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

import goggles._core.integrations.wandb as wandb_module
from goggles._core.integrations.wandb import WandBHandler


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
    mock_wandb.init.assert_called_once()
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
    """Equal and forward steps within the same scope are forwarded to wandb."""
    h = WandBHandler(project="proj")
    h.handle(make_event(kind="metric", payload={"a": 1}, step=3))
    h.handle(make_event(kind="metric", payload={"b": 2}, step=3))  # equal
    h.handle(make_event(kind="metric", payload={"c": 3}, step=4))  # forward

    run = mock_wandb.init.return_value
    assert run.log.call_count == 3, (
        f"Expected 3 wandb.log calls; saw {run.log.call_count}"
    )


def test_handle_tracks_step_per_scope(mock_wandb):
    """Step monotonicity is tracked per scope, not globally."""
    h = WandBHandler(project="proj")
    h.handle(
        make_event(kind="metric", payload={"x": 1}, scope="train", step=10)
    )
    # Lower step on a different scope must still be forwarded.
    h.handle(make_event(kind="metric", payload={"x": 2}, scope="eval", step=1))

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
