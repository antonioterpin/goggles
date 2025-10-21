import time
from typing import Mapping
import pytest
import numpy as np
from time import perf_counter

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except Exception:
    JAX_AVAILABLE = False

from goggles._core.metrics.event import MetricEvent


# ---------------------------------------------------------------------------
# Helper constructors
# ---------------------------------------------------------------------------
def make_image(h=8, w=8, c=3):
    return np.ones((h, w, c), dtype=np.float32)


def make_video(t=4, h=4, w=4, c=1):
    return np.ones((t, h, w, c), dtype=np.float32)


def _package_events(arrays: Mapping[str, object]) -> None:
    ts = time.time()
    for k, v in arrays.items():
        MetricEvent(key=k, type="image", step=0, payload=v, ts=ts)


# ---------------------------------------------------------------------------
# Scalar tests
# ---------------------------------------------------------------------------
def test_scalar_event_valid():
    ev = MetricEvent(key="loss", type="scalar", step=10, payload=0.42)
    assert ev.key == "loss"
    assert ev.payload == 0.42
    assert ev.type == "scalar"
    assert isinstance(ev.ts, float)


@pytest.mark.parametrize("value", ["str", None, object()])
def test_scalar_event_invalid_payload_type(value):
    with pytest.raises(TypeError):
        MetricEvent(key="acc", type="scalar", step=1, payload=value)


# ---------------------------------------------------------------------------
# Image tests
# ---------------------------------------------------------------------------
def test_image_event_valid():
    img = make_image()
    ev = MetricEvent(key="frame", type="image", step=0, payload=img)
    assert isinstance(ev.payload, np.ndarray)
    assert ev.payload.shape == (8, 8, 3)
    assert np.allclose(ev.payload, 1.0)


@pytest.mark.parametrize(
    "bad_array",
    [
        np.ones((8,), dtype=np.float32),  # 1D
        np.ones((8, 8, 8, 1), dtype=np.float32),  # 4D
    ],
)
def test_image_event_invalid_shape(bad_array):
    with pytest.raises(ValueError):
        MetricEvent(key="img", type="image", step=0, payload=bad_array)


def test_image_event_invalid_dtype():
    img = np.ones((8, 8, 3), dtype=bool)
    with pytest.raises(TypeError):
        MetricEvent(key="img", type="image", step=0, payload=img)


# ---------------------------------------------------------------------------
# Video tests
# ---------------------------------------------------------------------------
def test_video_event_valid():
    vid = make_video()
    ev = MetricEvent(key="video", type="video", step=5, payload=vid)
    assert isinstance(ev.payload, np.ndarray)
    assert ev.payload.shape[0] == 4


@pytest.mark.parametrize(
    "shape",
    [
        (4, 4),  # 2D
        (4, 4, 4, 4, 1),  # 5D
    ],
)
def test_video_event_invalid_shape(shape):
    arr = np.ones(shape, dtype=np.float32)
    with pytest.raises(ValueError):
        MetricEvent(key="vid", type="video", step=0, payload=arr)


def test_video_event_invalid_dtype():
    arr = np.ones((2, 2, 2), dtype=bool)
    with pytest.raises(TypeError):
        MetricEvent(key="vid", type="video", step=0, payload=arr)


# ---------------------------------------------------------------------------
# Histogram tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "payload",
    [
        np.array([1, 2, 3], dtype=np.float32),
        [1.0, 2.0, 3.0],
        (1, 2, 3),
    ],
)
def test_histogram_event_valid(payload):
    ev = MetricEvent(key="hist", type="histogram", step=99, payload=payload)
    assert ev.key == "hist"
    assert ev.step == 99


@pytest.mark.parametrize(
    "payload",
    [
        np.ones((2, 2), dtype=np.float32),  # 2D
        ["a", "b", "c"],  # non-numeric
    ],
)
def test_histogram_event_invalid(payload):
    with pytest.raises((TypeError, ValueError)):
        MetricEvent(key="hist", type="histogram", step=1, payload=payload)


# ---------------------------------------------------------------------------
# Common field validations
# ---------------------------------------------------------------------------
def test_invalid_key_raises():
    with pytest.raises(ValueError):
        MetricEvent(key="", type="scalar", step=0, payload=0.1)


def test_negative_step_raises():
    with pytest.raises(ValueError):
        MetricEvent(key="k", type="scalar", step=-1, payload=0.1)


def test_tags_and_context_validation():
    img = make_image()
    with pytest.raises(TypeError):
        MetricEvent(key="img", type="image", step=0, payload=img, tags=["ok", 123])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        MetricEvent(key="img", type="image", step=0, payload=img, context="not_a_dict")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Materialized payload tests
# ---------------------------------------------------------------------------
def test_materialized_event_skips_validation():
    """Ensure that events with _state='materialized' accept bytes payloads."""
    payload = b"\x89PNG\r\n\x1a\n..."  # dummy encoded data

    # Should not raise, because _state='materialized' bypasses array validation
    event = MetricEvent(
        key="train/image",
        type="image",
        step=0,
        payload=payload,
        context={"_state": "materialized"},
    )

    assert event.key == "train/image"
    assert event.context["_state"] == "materialized"
    assert isinstance(event.payload, (bytes, bytearray))


@pytest.mark.parametrize("bad_payload", [123, np.zeros((8, 8, 3))])
def test_materialized_event_raises_if_not_bytes(bad_payload):
    """Ensure that invalid materialized payloads raise TypeError."""
    with pytest.raises(TypeError):
        MetricEvent(
            key="train/image",
            type="image",
            step=0,
            payload=bad_payload,
            context={"_state": "materialized"},
        )


# ---------------------------------------------------------------------------
# to_dict utility
# ---------------------------------------------------------------------------
def test_to_dict_produces_dict():
    ev = MetricEvent(key="x", type="scalar", step=0, payload=1.0)
    d = ev.to_dict()
    assert isinstance(d, dict)
    assert d["key"] == "x"
    assert isinstance(d["ts"], float)


# ---------------------------------------------------------------------------
# Device-safety and performance tests
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_jax_array_remains_on_device_during_validation():
    """Ensure that MetricEvent validation does not trigger device→host sync."""
    x = jnp.ones((32, 32, 3))
    start = time.perf_counter()

    ev = MetricEvent(key="img", type="image", step=0, payload=x)
    elapsed = time.perf_counter() - start

    # JAX operations are async; validation should take <5ms
    assert (
        elapsed < 0.005
    ), f"Validation too slow ({elapsed:.6f}s) — possible host sync."
    # Check payload is still a device array
    assert isinstance(ev.payload, jax.Array)
    # Explicit sync still works
    _ = x.block_until_ready()


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_video_jax_array_device_safe():
    """Video payload validation should not copy JAX arrays to host."""
    x = jnp.ones((4, 16, 16, 3))
    ev = MetricEvent(key="vid", type="video", step=1, payload=x)
    assert isinstance(ev.payload, jax.Array)


@pytest.mark.parametrize("backend", ["numpy"] + (["jax"] if JAX_AVAILABLE else []))
@pytest.mark.parametrize("num_metrics", [10, 100, 1000])
def test_metric_packaging_scaling_no_fixture(backend: str, num_metrics: int):
    """Benchmark packaging speed for many metrics using a manual timer.

    Target guardrails (generous, per-event):
      - NumPy host arrays:  <= 200 µs
      - JAX device arrays:  <= 300 µs  (shape introspection can be slightly costlier)
    """
    if backend == "numpy":
        arrays = {f"m{i}": np.random.rand(16, 16, 3) for i in range(num_metrics)}
        per_event_budget = 200e-6
    else:
        arrays = {f"m{i}": jnp.ones((16, 16, 3)) for i in range(num_metrics)}
        per_event_budget = 300e-6

    # Warmup to avoid first-call overheads (imports, dataclass caches, etc.)
    _package_events({k: arrays[k] for k in list(arrays)[: min(5, num_metrics)]})

    start = perf_counter()
    _package_events(arrays)
    elapsed = perf_counter() - start

    per_event = elapsed / num_metrics
    print(
        f"[{backend}] {num_metrics} metrics → {per_event*1e6:.2f} µs/event "
        f"(total {elapsed:.4f}s)"
    )
    assert per_event <= per_event_budget, (
        f"Packaging too slow for {backend}: {per_event*1e6:.1f} µs/event "
        f"(budget {per_event_budget*1e6:.0f} µs)"
    )


def test_payload_identity_preserved():
    """Payload object reference should remain identical after packaging."""
    x_np = np.ones((8, 8, 3))
    ev_np = MetricEvent(key="img", type="image", step=1, payload=x_np)
    assert ev_np.payload is x_np

    if JAX_AVAILABLE:
        x_jax = jnp.ones((8, 8, 3))
        ev_jax = MetricEvent(key="img", type="image", step=1, payload=x_jax)
        assert ev_jax.payload is x_jax
