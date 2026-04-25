import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from goggles._core.logger import CoreGogglesLogger, CoreTextLogger


@pytest.fixture
def mock_client() -> MagicMock:
    """Mock transport to capture emitted events.

    Returns:
        MagicMock: Transport whose ``emit`` / ``emit_sync`` record calls.
    """
    client = MagicMock()
    client.emit = MagicMock(return_value=None)
    client.emit_sync = MagicMock(return_value=None)
    return client


@pytest.fixture
def patch_bus(
    monkeypatch: pytest.MonkeyPatch, mock_client: MagicMock
) -> MagicMock:
    """Patch get_bus() to return a mock client.

    Args:
        monkeypatch: Fixture used to replace routing `get_bus`.
        mock_client: Mock client fixture returned by `mock_client`.

    Returns:
        MagicMock: Patched mock client.
    """
    monkeypatch.setattr("goggles._core.routing.get_bus", lambda: mock_client)
    return mock_client


@pytest.fixture
def text_logger(patch_bus: MagicMock) -> CoreTextLogger:
    """Return a CoreTextLogger bound to a dummy scope.

    Args:
        patch_bus: Patched mock client fixture.

    Returns:
        CoreTextLogger: Text logger under test.
    """
    return CoreTextLogger(name="test", scope="global")


@pytest.fixture
def goggles_logger(patch_bus: MagicMock) -> CoreGogglesLogger:
    """Return a CoreGogglesLogger bound to a dummy scope.

    Args:
        patch_bus: Patched mock client fixture.

    Returns:
        CoreGogglesLogger: Metrics logger under test.
    """
    return CoreGogglesLogger(name="test", scope="global")


# -------------------------------------------------------------------------
# CoreTextLogger tests
# -------------------------------------------------------------------------


def test_bind_creates_new_context(text_logger: CoreTextLogger) -> None:
    """``bind`` returns a derived adapter with merged persistent fields.

    Args:
        text_logger: ``CoreTextLogger`` fixture.
    """
    text_logger._bound = {"old": 1}
    bound_logger = text_logger.bind(scope="run", new=2)
    assert bound_logger.get_bound() == {
        "old": 1,
        "new": 2,
    }, "Bound context mismatch after bind"
    assert bound_logger._scope == "run", "Bound logger scope mismatch"


@pytest.mark.parametrize(
    "level,method",
    [
        (logging.DEBUG, "debug"),
        (logging.INFO, "info"),
        (logging.WARNING, "warning"),
        (logging.ERROR, "error"),
        (logging.CRITICAL, "critical"),
    ],
)
def test_log_methods_emit_event(
    text_logger: CoreTextLogger,
    patch_bus: MagicMock,
    level: int,
    method: str,
) -> None:
    """Each severity method emits an event whose level matches.

    Args:
        text_logger: ``CoreTextLogger`` fixture.
        patch_bus: Patched mock client fixture.
        level: Standard logging level the method should stamp on the event.
        method: Name of the level method to invoke (``debug``, ``info``, ...).
    """
    msg = f"message-{method}"
    getattr(text_logger, method)(msg, step=1, time=123.0, extra_field="x")
    assert patch_bus.emit.called, "patch_bus.emit should have been called"
    event = patch_bus.emit.call_args[0][0]
    # Check Event object attributes (not dict keys)
    assert event.kind == "log", "Event kind should be 'log'"
    assert event.payload == msg, "Event payload should match logged message"
    assert event.level == level, "Event level mismatch"
    assert event.extra["extra_field"] == "x", "Event extra field mismatch"


def test_repr_includes_name_and_bound(text_logger: CoreTextLogger) -> None:
    """``repr`` carries the class, the logger name, and the bound keys.

    Args:
        text_logger: ``CoreTextLogger`` fixture.
    """
    text_logger._bound = {"a": 1}
    rep = repr(text_logger)
    assert "CoreTextLogger" in rep, "repr should include class name"
    assert "a" in rep, "repr should include bound field names"
    assert "test" in rep, "repr should include logger name"


# -------------------------------------------------------------------------
# CoreGogglesLogger tests
# -------------------------------------------------------------------------


def test_push_emits_metric_event(
    goggles_logger: CoreGogglesLogger, patch_bus: MagicMock
) -> None:
    """``push`` emits a metric event carrying the dict payload + step.

    Args:
        goggles_logger: ``CoreGogglesLogger`` fixture.
        patch_bus: Patched mock client fixture.
    """
    metrics = {"loss": 0.1, "acc": 0.9}
    goggles_logger.push(metrics, step=2)
    event = patch_bus.emit.call_args[0][0]
    assert event.kind == "metric", "Event kind should be 'metric'"
    assert event.payload == metrics, "Event payload should match pushed metrics"
    assert event.step == 2, "Event step mismatch"


@pytest.mark.parametrize(
    "shape",
    [(8, 12), (8, 12, 1), (8, 12, 3), (8, 12, 4)],
    ids=["hw", "hw1", "hw3", "hw4"],
)
def test_push_promotes_image_shaped_ndarrays(
    goggles_logger: CoreGogglesLogger,
    patch_bus: MagicMock,
    shape: tuple[int, ...],
) -> None:
    """Image-shaped ndarrays passed to ``push`` are emitted as image events.

    Args:
        goggles_logger: ``CoreGogglesLogger`` fixture.
        patch_bus: Patched mock client fixture.
        shape: Tensor shape under test (parametrized).
    """
    img = np.zeros(shape, dtype=np.uint8)
    goggles_logger.push({"fig": img}, step=3)
    kinds = [c.args[0].kind for c in patch_bus.emit.call_args_list]
    assert kinds == ["image"]
    event = patch_bus.emit.call_args_list[0].args[0]
    np.testing.assert_array_equal(event.payload, img)
    assert event.extra["name"] == "fig"
    assert event.extra["format"] == "png"
    assert event.step == 3


def test_push_keeps_1d_ndarray_as_metric(
    goggles_logger: CoreGogglesLogger, patch_bus: MagicMock
) -> None:
    """1-D ndarrays stay in the metric payload (not promoted to image).

    Args:
        goggles_logger: ``CoreGogglesLogger`` fixture.
        patch_bus: Patched mock client fixture.
    """
    vec = np.arange(4, dtype=np.float32)
    goggles_logger.push({"v": vec}, step=1)
    event = patch_bus.emit.call_args[0][0]
    assert event.kind == "metric"
    np.testing.assert_array_equal(event.payload["v"], vec)


def test_push_mixes_scalars_and_images(
    goggles_logger: CoreGogglesLogger, patch_bus: MagicMock
) -> None:
    """``push`` splits a mixed dict into one metric event + per-image events.

    Args:
        goggles_logger: ``CoreGogglesLogger`` fixture.
        patch_bus: Patched mock client fixture.
    """
    img1 = np.zeros((4, 4), dtype=np.uint8)
    img2 = np.zeros((4, 4, 3), dtype=np.uint8)
    goggles_logger.push(
        {"loss": 0.1, "acc": 0.9, "fig1": img1, "fig2": img2}, step=5
    )
    events = [c.args[0] for c in patch_bus.emit.call_args_list]
    kinds = [e.kind for e in events]
    assert kinds.count("metric") == 1
    assert kinds.count("image") == 2

    metric_event = next(e for e in events if e.kind == "metric")
    assert metric_event.payload == {"loss": 0.1, "acc": 0.9}
    assert metric_event.step == 5

    image_names = [e.extra["name"] for e in events if e.kind == "image"]
    assert sorted(image_names) == ["fig1", "fig2"]
    for e in events:
        assert e.step == 5


def test_push_image_only_does_not_emit_empty_metric_event(
    goggles_logger: CoreGogglesLogger, patch_bus: MagicMock
) -> None:
    """Image-only ``push`` must not emit an empty metric event alongside.

    Args:
        goggles_logger: ``CoreGogglesLogger`` fixture.
        patch_bus: Patched mock client fixture.
    """
    img = np.zeros((4, 4), dtype=np.uint8)
    goggles_logger.push({"fig": img}, step=2)
    kinds = [c.args[0].kind for c in patch_bus.emit.call_args_list]
    assert "metric" not in kinds


def test_push_forwards_extras_to_image_events(
    goggles_logger: CoreGogglesLogger, patch_bus: MagicMock
) -> None:
    """Extras forwarded to ``push`` reach each promoted image event's extras.

    Args:
        goggles_logger: ``CoreGogglesLogger`` fixture.
        patch_bus: Patched mock client fixture.
    """
    img = np.zeros((4, 4), dtype=np.uint8)
    goggles_logger.push({"fig": img}, step=2, split="train")
    event = patch_bus.emit.call_args[0][0]
    assert event.kind == "image"
    assert event.extra["split"] == "train"


def test_scalar_emits_metric_event(
    goggles_logger: CoreGogglesLogger, patch_bus: MagicMock
) -> None:
    """``scalar`` emits a metric event keyed by the metric name.

    Args:
        goggles_logger: ``CoreGogglesLogger`` fixture.
        patch_bus: Patched mock client fixture.
    """
    goggles_logger.scalar("loss", 0.42)
    event = patch_bus.emit.call_args[0][0]
    assert event.kind == "metric", "Event kind should be 'metric' for scalar"
    assert event.payload == {"loss": 0.42}, (
        "Event payload should match scalar metric"
    )


@pytest.mark.parametrize(
    "kind,method,arg_key",
    [
        ("image", "image", "image"),
        ("video", "video", "video"),
        ("artifact", "artifact", "data"),
        ("vector_field", "vector_field", "vector_field"),
        ("trajectories", "trajectories", "trajectories"),
        ("histogram", "histogram", "histogram"),
    ],
)
def test_artifact_like_methods_emit_event(
    goggles_logger: CoreGogglesLogger,
    patch_bus: MagicMock,
    kind: str,
    method: str,
    arg_key: str,
) -> None:
    """Each artifact-like method emits an event of the matching kind.

    Args:
        goggles_logger: ``CoreGogglesLogger`` fixture.
        patch_bus: Patched mock client fixture.
        kind: Expected ``event.kind`` after the call.
        method: Logger method to invoke.
        arg_key: Unused — present for parametrize symmetry.
    """
    del arg_key
    fake_payload = SimpleNamespace(dummy=True)
    kwargs = {}
    if method == "video":
        kwargs["fps"] = 60
    kwargs["step"] = 1
    getattr(goggles_logger, method)(fake_payload, name="foo", **kwargs)
    event = patch_bus.emit.call_args[0][0]
    assert event.kind == kind, f"Event kind should be '{kind}'"
    assert "name" in event.extra, (
        "Event extra should contain 'name' for artifacts"
    )
    assert event.extra["name"] == "foo", "Event extra 'name' mismatch"


def test_histogram_adds_name_and_payload(
    goggles_logger: CoreGogglesLogger, patch_bus: MagicMock
) -> None:
    """``histogram`` emits an event with ``extra['name']`` and payload set.

    Args:
        goggles_logger: ``CoreGogglesLogger`` fixture.
        patch_bus: Patched mock client fixture.
    """
    goggles_logger.histogram([1, 2, 3], name="hist", step=1)
    event = patch_bus.emit.call_args[0][0]
    assert event.kind == "histogram", "Event kind should be 'histogram'"
    assert event.extra["name"] == "hist", (
        "Event extra 'name' mismatch for histogram"
    )
    assert event.payload == [1, 2, 3], "Event payload mismatch for histogram"


def test_sync_mode_uses_emit_sync(patch_bus: MagicMock) -> None:
    """Ensure ``async_mode=False`` routes through the transport's sync path.

    Args:
        patch_bus: Patched mock client fixture.
    """
    g = CoreGogglesLogger(name="sync", scope="run")

    g.scalar("metric", 1.0, async_mode=False)
    patch_bus.emit_sync.assert_called_once()
    patch_bus.emit.assert_not_called()


def test_async_mode_uses_emit(patch_bus: MagicMock) -> None:
    """Default (async) path should route through ``emit``.

    Args:
        patch_bus: Patched mock client fixture.
    """
    g = CoreGogglesLogger(name="async", scope="run")

    g.scalar("metric", 1.0)
    patch_bus.emit.assert_called_once()
    patch_bus.emit_sync.assert_not_called()


def test_default_level_emits_all_severities(
    text_logger: CoreTextLogger, patch_bus: MagicMock
) -> None:
    """With the default ``NOTSET`` gate every severity reaches the bus.

    Args:
        text_logger: ``CoreTextLogger`` fixture.
        patch_bus: Patched mock client fixture.
    """
    for method in ("debug", "info", "warning", "error", "critical"):
        getattr(text_logger, method)("x")
    assert patch_bus.emit.call_count == 5


def test_set_level_drops_below_threshold(patch_bus: MagicMock) -> None:
    """``set_level`` drops calls below the threshold and forwards the rest.

    Args:
        patch_bus: Patched mock client fixture.
    """
    logger = CoreTextLogger(name="t", scope="global")
    logger.set_level(logging.INFO)

    logger.debug("drop")
    assert patch_bus.emit.call_count == 0

    logger.info("keep")
    logger.warning("keep")
    assert patch_bus.emit.call_count == 2


def test_set_level_warning_drops_info_and_debug(
    patch_bus: MagicMock,
) -> None:
    """At ``WARNING`` only WARNING and above propagate.

    Args:
        patch_bus: Patched mock client fixture.
    """
    logger = CoreTextLogger(name="t", scope="global")
    logger.set_level(logging.WARNING)

    logger.debug("drop")
    logger.info("drop")
    assert patch_bus.emit.call_count == 0

    logger.warning("keep")
    logger.error("keep")
    logger.critical("keep")
    assert patch_bus.emit.call_count == 3


def test_level_via_constructor(patch_bus: MagicMock) -> None:
    """Constructor-time ``level=`` is honoured the same as ``set_level``.

    Args:
        patch_bus: Patched mock client fixture.
    """
    logger = CoreTextLogger(name="t", scope="global", level=logging.WARNING)
    logger.info("drop")
    logger.warning("keep")
    assert patch_bus.emit.call_count == 1


def test_bind_preserves_level(patch_bus: MagicMock) -> None:
    """``bind`` carries the parent level into the derived adapter.

    Args:
        patch_bus: Patched mock client fixture.
    """
    logger = CoreTextLogger(name="t", scope="global", level=logging.WARNING)
    child = logger.bind(scope="run", k=1)
    child.info("drop")
    child.warning("keep")
    assert patch_bus.emit.call_count == 1


def test_set_level_is_logger_local(patch_bus: MagicMock) -> None:
    """``set_level`` only mutates the logger it's called on.

    Args:
        patch_bus: Patched mock client fixture.
    """
    quiet = CoreTextLogger(name="quiet", scope="global")
    quiet.set_level(logging.WARNING)

    loud = CoreTextLogger(name="loud", scope="global")

    quiet.debug("drop")
    loud.debug("keep")
    assert patch_bus.emit.call_count == 1


def test_get_logger_level_kwarg(patch_bus: MagicMock) -> None:
    """``gg.get_logger(..., level=...)`` propagates to the logger gate.

    Args:
        patch_bus: Patched mock client fixture.
    """
    import goggles as gg  # noqa: PLC0415

    logger = gg.get_logger("x", level=logging.WARNING)
    logger.info("drop")
    logger.warning("keep")
    assert patch_bus.emit.call_count == 1
