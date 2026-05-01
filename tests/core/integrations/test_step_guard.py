"""Unit tests for the per-scope monotonic step tracker."""

from __future__ import annotations

import threading

from goggles._core.integrations._step_guard import StepGuard


def test_first_step_is_never_flagged():
    g = StepGuard()
    assert g.check("global", 0) is False, (
        "First step in a scope must never be flagged as out-of-order"
    )
    assert g.check("train", 1000) is False, (
        "First step in a different scope must never be flagged"
    )


def test_step_none_is_never_flagged():
    g = StepGuard()
    g.check("global", 10)
    assert g.check("global", None) is False, "step=None must never be flagged"
    # None must not affect the tracked max either
    assert g.check("global", 11) is False, (
        "step=11 after a None call should not be flagged (max should still be "
        "10)"
    )


def test_strictly_backward_step_is_flagged():
    g = StepGuard()
    g.check("global", 10)
    assert g.check("global", 5) is True, (
        "step=5 after step=10 must be flagged as out-of-order"
    )


def test_equal_step_is_not_flagged():
    g = StepGuard()
    g.check("global", 10)
    # wandb allows multiple metrics at the same step (same call); mirror that.
    assert g.check("global", 10) is False, (
        "Equal step (10 after 10) must not be flagged "
        "— mirrors wandb same-step semantics"
    )


def test_max_does_not_regress_after_flagged_step():
    g = StepGuard()
    g.check("global", 10)
    g.check("global", 5)  # flagged, must not lower the max
    assert g.check("global", 6) is True, (
        "step=6 must still be flagged (max should remain 10 after the flagged "
        "step=5)"
    )


def test_per_scope_isolation():
    g = StepGuard()
    g.check("train", 10)
    # Different scope; lower step is fine.
    assert g.check("eval", 1) is False, (
        "step=1 in 'eval' scope must not be flagged by 'train' scope's max"
    )
    # train's max is unaffected.
    assert g.check("train", 5) is True, (
        "step=5 in 'train' scope must still be flagged (its max is 10)"
    )


def test_reset_clears_table():
    g = StepGuard()
    g.check("global", 10)
    g.reset()
    assert g.check("global", 1) is False, (
        "After reset(), the next call must be treated as the first step again"
    )


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
    assert g.check("scope", 7) is False, (
        "After 8 concurrent checks (steps 0..7), "
        "step=7 must equal the max and not be flagged"
    )
    assert g.check("scope", 0) is True, (
        "After 8 concurrent checks (max=7), "
        "step=0 must be flagged as out-of-order"
    )
