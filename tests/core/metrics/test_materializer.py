import io
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# imageio is optional in the test environment; import defensively so skipif can refer to it.
try:
    import imageio
except Exception:
    imageio = None

from goggles._core.metrics.queue import MetricsQueue
from goggles._core.metrics.event import MetricEvent
from goggles._core.metrics.materializer import MaterializerWorker


@pytest.fixture
def make_event():
    def _make_event(type_, payload):
        return MetricEvent(
            key="test/key",
            type=type_,
            step=0,
            payload=payload,
            context={},
        )

    return _make_event


@pytest.fixture
def queues():
    in_q = MetricsQueue(maxsize=5)
    out_q = MetricsQueue(maxsize=5)
    stop = threading.Event()
    return in_q, out_q, stop


@pytest.fixture
def worker(queues):
    in_q, out_q, stop = queues
    w = MaterializerWorker(in_q, out_q, stop, image_format="png", video_format="mp4")
    yield w
    stop.set()


def test_process_event_image_basic(worker, make_event):
    arr = np.random.rand(16, 16, 3)
    event = make_event("image", arr)
    out = worker.process_event(event)

    assert isinstance(out, MetricEvent)
    assert isinstance(out.payload, bytes)

    assert out.context.get("_state") == "materialized"
    enc = out.context.get("encoding")
    assert isinstance(enc, dict)
    assert enc["content_type"].startswith("image/")
    assert "shape" in enc


def test_process_event_video_basic(worker, make_event):
    arr = np.random.rand(3, 16, 16, 3)
    event = make_event("video", arr)
    out = worker.process_event(event)

    assert isinstance(out.payload, bytes)
    enc = out.context.get("encoding")
    assert enc["content_type"] == "video/mp4"
    assert enc["fps"] == worker._fps


def test_downscale_applied(worker, make_event):
    worker._downscale = 2
    arr = np.ones((8, 8, 3))
    event = make_event("image", arr)
    out = worker.process_event(event)

    enc = out.context.get("encoding")
    assert enc["shape"][0] == arr.shape[0] // 2
    assert enc["shape"][1] == arr.shape[1] // 2


def test_quantize_float_to_uint8(worker):
    x = np.linspace(0, 1, 10, dtype=np.float32)
    y = worker._quantize(x)
    assert y.dtype == np.uint8
    assert y.min() >= 0 and y.max() <= 255


def test_quantize_custom_policy(worker):
    def custom_quant(x):
        return np.zeros_like(x, dtype=np.uint8)

    worker._quantize_policy = custom_quant
    x = np.random.randn(8, 8)
    y = worker._quantize(x)
    assert np.all(y == 0)


def test_quantize_invalid_policy(worker):
    def bad_quant(x):
        return x.astype(np.float32)

    worker._quantize_policy = bad_quant
    with pytest.raises(ValueError):
        worker._quantize(np.random.rand(4, 4))


def test_encode_image_modes(worker):
    gray = np.random.randint(0, 255, (8, 8), np.uint8)
    img3 = np.random.randint(0, 255, (8, 8, 3), np.uint8)
    img4 = np.random.randint(0, 255, (8, 8, 4), np.uint8)
    for arr in [gray, img3, img4]:
        data, meta = worker._encode_image(arr, "png")
        assert isinstance(data, bytes)
        assert meta["shape"] == arr.shape


def test_encode_image_invalid_shape(worker):
    bad = np.random.randint(0, 255, (2, 2, 5), np.uint8)
    with pytest.raises(ValueError):
        worker._encode_image(bad, "png")


@pytest.mark.skipif(
    "imageio" not in globals() or imageio is None, reason="imageio unavailable"
)
def test_encode_video_basic(worker):
    vid = np.random.randint(0, 255, (2, 8, 8, 3), np.uint8)
    data, meta = worker._encode_video(vid, "mp4", fps=10)
    assert isinstance(data, bytes)
    assert meta["content_type"] == "video/mp4"
    assert meta["fps"] == 10


@pytest.mark.skipif(
    "imageio" not in globals() or imageio is None, reason="imageio unavailable"
)
def test_encode_video_grayscale_promoted(worker):
    vid = np.random.randint(0, 255, (2, 8, 8, 1), np.uint8)
    data, meta = worker._encode_video(vid, "mp4", fps=5)
    assert meta["shape"][-1] == 3


def test_encode_video_invalid_channels(worker):
    vid = np.random.randint(0, 255, (2, 8, 8, 2), np.uint8)
    with pytest.raises(ValueError):
        worker._encode_video(vid, "mp4", fps=10)


def test_encode_video_invalid_format(worker):
    vid = np.random.randint(0, 255, (2, 8, 8, 3), np.uint8)
    with pytest.raises(ValueError):
        worker._encode_video(vid, "avi", fps=10)


def test_to_numpy_accepts_array(worker):
    arr = np.ones((2, 2))
    out = worker._to_numpy(arr)
    assert np.allclose(out, arr)


def test_to_numpy_rejects_invalid(worker):
    with pytest.raises(TypeError):
        worker._to_numpy("not-an-array")


def test_attach_error_adds_field(worker, make_event):
    event = make_event("scalar", 1.0)
    out = worker._attach_error(event, "fail")

    assert isinstance(out, MetricEvent)
    assert out.context.get("_state") == "error"
    assert "materializer_error" in out.context
    assert "fail" in out.context["materializer_error"]


def test_attach_error_rejects_dict_event(worker):
    e = {"key": "x", "payload": 1, "type": "scalar"}
    with pytest.raises(TypeError):
        worker._attach_error(e, "bad")


def test_run_stops_on_event(queues, make_event):
    in_q, out_q, stop = queues
    worker = MaterializerWorker(in_q, out_q, stop)
    arr = np.zeros((4, 4, 3))
    in_q.enqueue(make_event("image", arr))

    t = threading.Thread(target=worker.run, daemon=True)
    t.start()
    time.sleep(0.05)
    stop.set()
    t.join(timeout=1.0)

    out_event = out_q.dequeue()
    assert out_event is not None
    assert out_event.context.get("_state") == "materialized"
