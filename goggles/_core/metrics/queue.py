"""Bounded staging queue for metrics events.

WARNING: This module is an internal implementation detail of Goggles' metrics
pipeline. It is **not** part of the public API.

External code should **not** import from this module. Instead, depend on the
public metrics interfaces (e.g., `goggles.metrics.push(...)` and friends) that
abstract queueing and backpressure behavior.

This module provides a thread-safe, bounded FIFO used to stage `MetricEvent`s
before serialization/export. It enforces per-type rate limits and supports
overflow strategies:
  * DROP_OLDEST — evict the oldest event when full,
  * DROP_NEWEST — reject the incoming event,

For observability, drop counters are maintained per event type, and queue depth
is exposed. The implementation favors predictability and constant-time
operations under load; no blocking backpressure is introduced here.
"""

import threading
import time
from collections import deque, defaultdict
from typing import Any, Deque, Dict, Optional, Tuple

from .event import MetricEvent


class MetricsQueue:
    """Thread-safe bounded queue with rate limits and drop/coalesce policies.

    This class provides a bounded FIFO queue for metrics events, ensuring predictable
    memory use and consistent enqueue/dequeue performance under heavy load. It supports
    per-event-type rate limiting and configurable overflow strategies.

    Notes:
        - Thread-safe through internal locking.
        - Intended for short-lived, in-memory buffering of metric events before
          serialization or export.

    """

    DROP_OLDEST = "DROP_OLDEST"
    DROP_NEWEST = "DROP_NEWEST"

    def __init__(
        self,
        maxsize: int = 1000,
        rate_limits: Optional[Dict[str, Tuple[int, float]]] = None,
        drop_policy: str = DROP_OLDEST,
    ) -> None:
        """Initialize a bounded metrics queue.

        Args:
            maxsize (int): Maximum number of items in the queue.
            rate_limits (dict[str, tuple[int, float]] | None): Optional per-type rate limits
                in the form `{event_type: (max_events, window_seconds)}`. When a limit is
                exceeded, new events of that type are dropped.
            drop_policy (str): Overflow handling policy:
                * `"DROP_OLDEST"` — Discard oldest event when full.
                * `"DROP_NEWEST"` — Drop new event and increment drop counter.

        Raises:
            ValueError: If `maxsize` <= 0 or if `drop_policy` is invalid.

        Examples:
            >>> q = MetricsQueue(maxsize=10, drop_policy=MetricsQueue.DROP_OLDEST)
            >>> q.enqueue(MetricEvent(event_type="scalar", key="loss", value=1.0))
            True

        """
        if maxsize <= 0:
            raise ValueError("maxsize must be > 0")
        if drop_policy not in {self.DROP_OLDEST, self.DROP_NEWEST}:
            raise ValueError(f"Invalid drop_policy: {drop_policy}")

        self._maxsize = maxsize
        self._queue: Deque[Any] = deque(maxlen=maxsize)
        self._lock = threading.Lock()
        self._rate_limits = rate_limits or {}
        self._timestamps: Dict[str, Deque[float]] = defaultdict(deque)
        self._drop_policy = drop_policy
        self._drop_counts: Dict[str, int] = defaultdict(int)

    def _check_rate_limit(self, event_type: str) -> bool:
        """Check whether enqueueing is allowed under the current rate limit.

        Args:
            event_type (str): Type name of the metric event.

        Returns:
            bool: True if event can be accepted, False if rate limit exceeded.

        """
        if event_type not in self._rate_limits:
            return True
        max_events, window = self._rate_limits[event_type]
        now = time.time()
        timestamps = self._timestamps[event_type]

        # Drop timestamps outside the sliding window
        while timestamps and now - timestamps[0] > window:
            timestamps.popleft()

        # Enforce maximum number of events within the window
        if len(timestamps) >= max_events:
            return False

        timestamps.append(now)
        return True

    def enqueue(self, event: MetricEvent) -> bool:
        """Enqueue a metric event, enforcing rate limits and drop policy.

        Args:
            event (MetricEvent): Event to enqueue. Must define an `event_type` attribute.

        Returns:
            bool: True if enqueued successfully, False if dropped due to rate limit or
            overflow.

        """
        event_type = getattr(event, "type", "default")

        with self._lock:
            # Rate limiting check
            if not self._check_rate_limit(event_type):
                self._drop_counts[event_type] += 1
                return False

            # Handle full queue cases
            if len(self._queue) >= self._maxsize:
                if self._drop_policy == self.DROP_OLDEST:
                    self._queue.popleft()
                elif self._drop_policy == self.DROP_NEWEST:
                    self._drop_counts[event_type] += 1
                    return False

            self._queue.append(event)
            return True

    def dequeue(self) -> Optional[MetricEvent]:
        """Remove and return the next event if available.

        Returns:
            Optional[MetricEvent]: The next queued event, or None if queue is empty.

        """
        with self._lock:
            if not self._queue:
                return None
            return self._queue.popleft()

    def get_depth(self) -> int:
        """Return the current number of events in the queue.

        Returns:
            int: Number of events currently stored.

        """
        with self._lock:
            return len(self._queue)

    def get_drop_stats(self) -> Dict[str, int]:
        """Return per-type statistics of dropped events.

        Returns:
            Dict[str, int]: Mapping from event type to drop count.

        """
        with self._lock:
            return dict(self._drop_counts)
