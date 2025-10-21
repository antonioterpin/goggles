# [Sub-Issue] Implement Bounded Staging Queue with Rate Limits and Drop/Coalesce Policies

## Summary

Create the queue system for metrics events, enforcing per-type rate limits and drop/coalesce strategies. The queue should be bounded in size and support policies for handling overflow and event coalescence.

## Motivation

A bounded queue with rate limits and drop policies ensures predictable performance and resource usage, preventing metrics logging from overwhelming the system or blocking producers.

## Implementation Plan

- Design a thread-safe, bounded queue for metrics events.
- Implement per-type rate limiting (e.g., images per N steps).
- Support drop policies: DROP_OLDEST, DROP_NEWEST.
- Integrate queue with metrics API (`push` enqueues events).
- Expose queue depth and drop counters for observability.

## API Sketch

```python
class MetricsQueue:
    def enqueue(self, event: MetricEvent) -> bool: ...
    def dequeue(self) -> MetricEvent: ...
    def get_depth(self) -> int: ...
    def get_drop_stats(self) -> dict: ...
```

## Tests

- Test queue overflow and drop behavior.
- Validate rate limiting per type/key.
- Check thread safety under concurrent access.

## Dependencies

Depends on: [Event Schema and Validation Logic]
