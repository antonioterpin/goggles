"""Unit tests for the per-scope monotonic step tracker (issue #70)."""

from __future__ import annotations

import threading

from goggles._core.integrations._step_guard import StepGuard


def test_first_step_is_never_flagged():
    g = StepGuard()
    assert g.check("global", 0) is False
    assert g.check("global", 1000) is False  # different scope's first call


def test_step_none_is_never_flagged():
    g = StepGuard()
    g.check("global", 10)
    assert g.check("global", None) is False
    # None must not affect the tracked max either
    assert g.check("global", 11) is False


def test_strictly_backward_step_is_flagged():
    g = StepGuard()
    g.check("global", 10)
    assert g.check("global", 5) is True


def test_equal_step_is_not_flagged():
    g = StepGuard()
    g.check("global", 10)
    # wandb allows multiple metrics at the same step (same call); mirror that.
    assert g.check("global", 10) is False


def test_max_does_not_regress_after_flagged_step():
    g = StepGuard()
    g.check("global", 10)
    g.check("global", 5)  # flagged, must not lower the max
    assert g.check("global", 6) is True  # still backward relative to 10


def test_per_scope_isolation():
    g = StepGuard()
    g.check("train", 10)
    # Different scope; lower step is fine.
    assert g.check("eval", 1) is False
    # train's max is unaffected.
    assert g.check("train", 5) is True


def test_reset_clears_table():
    g = StepGuard()
    g.check("global", 10)
    g.reset()
    assert g.check("global", 1) is False  # back to first-event semantics


def test_concurrent_check_is_safe():
    """Concurrent .check() calls on the same scope must not corrupt state."""
    g = StepGuard()
    barrier = threading.Barrier(8)
    results: list[bool] = []
    lock = threading.Lock()

    def worker(step: int) -> None:
        barrier.wait()
        flagged = g.check("scope", step)
        with lock:
            results.append(flagged)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Whatever the scheduling, the max ends at 7 and a second call below
    # 7 must be flagged.
    assert g.check("scope", 7) is False
    assert g.check("scope", 0) is True
