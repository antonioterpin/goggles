# tests/core/metrics/test_metrics_queue.py

import threading
import time
from typing import Optional

import numpy as np
import pytest

from goggles._core.metrics.event import MetricEvent
from goggles._core.metrics.queue import MetricsQueue


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_scalar(step: int, key: str = "train/loss", value: float = 1.0) -> MetricEvent:
    return MetricEvent(key=key, type="scalar", step=step, payload=value)


def make_image(step: int, key: str = "viz/image") -> MetricEvent:
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    return MetricEvent(key=key, type="image", step=step, payload=img)


# ---------------------------------------------------------------------------
# Basic enqueue/dequeue with MetricEvent
# ---------------------------------------------------------------------------


def test_enqueue_dequeue_basic_metric_event() -> None:
    q = MetricsQueue(maxsize=4, drop_policy=MetricsQueue.DROP_OLDEST)
    e1 = make_scalar(0)
    e2 = make_image(1)

    assert q.enqueue(e1) is True
    assert q.enqueue(e2) is True
    assert q.get_depth() == 2

    out1 = q.dequeue()
    out2 = q.dequeue()
    assert isinstance(out1, MetricEvent)
    assert isinstance(out2, MetricEvent)
    assert out1.key == e1.key and out1.type == "scalar"
    assert out2.key == e2.key and out2.type == "image"
    assert q.dequeue() is None
    assert q.get_depth() == 0


# ---------------------------------------------------------------------------
# Overflow policies
# ---------------------------------------------------------------------------


def test_overflow_drop_oldest_does_not_increment_drop_stats() -> None:
    q = MetricsQueue(maxsize=2, drop_policy=MetricsQueue.DROP_OLDEST)
    q.enqueue(make_scalar(0))
    q.enqueue(make_scalar(1))
    q.enqueue(make_scalar(2))  # evicts oldest; should still accept
    assert q.get_depth() == 2
    # By design DROP_OLDEST does not increment drop counters.
    assert q.get_drop_stats() == {}
    objects = [q.dequeue() for _ in range(2)]
    steps = [obj.step for obj in objects if obj is not None]
    # Oldest (step=0) should have been evicted
    assert steps == [1, 2]


def test_overflow_drop_newest_increments_drop_stats_by_type() -> None:
    q = MetricsQueue(maxsize=1, drop_policy=MetricsQueue.DROP_NEWEST)
    assert q.enqueue(make_scalar(0)) is True
    # This one should be dropped (queue full, drop newest)
    assert q.enqueue(make_scalar(1)) is False

    # IMPORTANT: The queue should attribute drops to the MetricEvent.type ("scalar").
    # If the implementation looks for `event.event_type` instead of `event.type`,
    # this test will fail and should be fixed in the queue.
    stats = q.get_drop_stats()
    assert stats.get("scalar", 0) == 1


# ---------------------------------------------------------------------------
# Rate limiting (per type)
# ---------------------------------------------------------------------------


def test_rate_limit_per_type_window() -> None:
    # Allow at most 2 scalar events per 0.25 seconds; images unlimited
    q = MetricsQueue(
        maxsize=10,
        rate_limits={"scalar": (2, 0.25)},
        drop_policy=MetricsQueue.DROP_NEWEST,
    )

    assert q.enqueue(make_scalar(0)) is True
    assert q.enqueue(make_scalar(1)) is True
    # Third within the window should be dropped
    assert q.enqueue(make_scalar(2)) is False

    # Image type should not be rate limited
    assert q.enqueue(make_image(0)) is True
    assert q.enqueue(make_image(1)) is True

    stats = q.get_drop_stats()
    assert stats.get("scalar", 0) == 1
    assert "image" not in stats

    # After the window, scalar should be accepted again
    time.sleep(0.3)
    assert q.enqueue(make_scalar(3)) is True


# ---------------------------------------------------------------------------
# Thread safety smoke test
# ---------------------------------------------------------------------------


def test_thread_safety_concurrent_enqueues() -> None:
    q = MetricsQueue(maxsize=128, drop_policy=MetricsQueue.DROP_OLDEST)

    err: Optional[BaseException] = None

    def worker(start: int) -> None:
        nonlocal err
        try:
            for i in range(64):
                q.enqueue(make_scalar(start + i))
        except BaseException as e:  # pragma: no cover
            err = e

    threads = [threading.Thread(target=worker, args=(k * 64,)) for k in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert err is None
    # Depth should be bounded by maxsize
    assert 0 < q.get_depth() <= 128
    # Dequeue all to ensure queue integrity
    prev_step: Optional[int] = None
    while True:
        ev = q.dequeue()
        if ev is None:
            break
        assert isinstance(ev, MetricEvent)
        if prev_step is not None:
            # Not strictly ordered due to DROP_OLDEST evictions, but still integers.
            assert isinstance(ev.step, int)
        prev_step = ev.step


# ---------------------------------------------------------------------------
# Integration expectation: type attribution
# ---------------------------------------------------------------------------


def test_integration_uses_metricevent_type_for_drop_stats() -> None:
    """Regression guard: ensure MetricEvent.type is used as the 'event type'.

    If the queue implementation uses `getattr(event, "event_type", ...)`, this test will
    fail because MetricEvent exposes the attribute `type` (not `event_type`). The queue
    should read `event.type` to attribute rate limits and drop counters correctly.
    """
    q = MetricsQueue(maxsize=1, drop_policy=MetricsQueue.DROP_NEWEST)
    assert q.enqueue(make_scalar(0)) is True
    assert q.enqueue(make_scalar(1)) is False

    stats = q.get_drop_stats()
    # The drops MUST be attributed to 'scalar'
    assert "scalar" in stats and stats["scalar"] == 1
