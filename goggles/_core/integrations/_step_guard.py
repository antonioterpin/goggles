"""Per-scope monotonic-step tracking shared by handlers.

Different handlers historically dealt with out-of-order ``step`` values
inconsistently:

  - WandB enforced monotonicity (wandb's own ``run.log`` drops/warns
    when ``step`` goes backwards within a run).
  - LocalStorage and the console handler accepted anything.

This module gives both built-in handlers a single, consistent way to
reason about backward step jumps. The tracker itself only answers
*"was this step a regression?"* — the policy decision (drop, warn,
both) belongs to each handler.
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
        """Record ``step`` for ``scope`` and return whether it regressed.

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
