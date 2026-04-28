"""Per-scope monotonic-step tracking shared by handlers.

A small primitive that tracks the highest ``step`` seen per scope and
reports backward jumps, mirroring wandb's per-run monotonicity contract
(``run.log`` drops/warns when ``step`` goes backwards within a run).

The tracker only answers *"was this step a regression?"* — the policy
decision (drop, warn, both) belongs to each handler.
"""

from __future__ import annotations

import threading


class StepGuard:
    """Track the highest ``step`` seen per scope and detect regressions.

    A backward step is one where ``step < max_step_for_scope``. Events
    with ``step is None`` are never flagged (they have no ordering
    contract). The first event for a scope is never flagged.

    Thread-safe: handlers may call :meth:`check` from multiple emit
    threads concurrently. The internal table is guarded by a lock.
    """

    def __init__(self) -> None:
        """Initialize the per-scope step table."""
        self._max: dict[str, int] = {}
        self._lock = threading.Lock()

    def check(self, scope: str, step: int | None) -> bool:
        """Return whether ``step`` regressed for ``scope``.

        If ``step`` is non-decreasing, the tracked max for ``scope`` is
        bumped to ``step``. If it regresses, the tracked max is left
        unchanged so a subsequent valid step is still measured against
        the original high-water mark.

        Args:
            scope: Logical bucket the step is monotonic within. Mirrors
                wandb's per-run tracking — typically the event's scope.
            step: New step value. ``None`` is ignored (no contract).

        Returns:
            ``True`` if ``step`` is strictly less than the previously
            highest step seen for ``scope``. ``False`` otherwise (first
            event, ``step is None``, or step is non-decreasing).
        """
        if step is None:
            return False
        with self._lock:
            prev = self._max.get(scope)
            if prev is not None and step < prev:
                return True
            self._max[scope] = step
            return False

    def reset(self) -> None:
        """Forget all tracked scopes (useful for tests)."""
        with self._lock:
            self._max.clear()
