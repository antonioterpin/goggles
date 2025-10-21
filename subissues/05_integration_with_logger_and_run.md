# [Sub-Issue] Integrate Metrics API with BoundLogger and Run Lifecycle

## Summary

Extend BoundLogger and run context to support metrics push and context binding. Ensure metrics events inherit bound fields and run context, and that the metrics system is initialized and managed alongside the run lifecycle.

## Motivation

Seamless integration with the existing logging and run management system ensures consistent context, lifecycle management, and user experience for both text and metrics logging.

## Implementation Plan

- Extend BoundLogger with `push(step, metrics, ...)` method.
- Ensure bound fields apply to metrics events (context, prefix resolution).
- Initialize metrics system in `gg.run` and make available to loggers.
- Support contextvars or similar for run-scoped metrics client.
- Update examples and documentation for new API.

## API Sketch

```python
class BoundLogger:
    def push(self, step: int, metrics: Mapping[str, MetricLike], ...): ...
```

## Tests

- Validate context/prefix inheritance in metrics events.
- Test run lifecycle management of metrics system.
- Check compatibility with existing text logging.

## Dependencies

Depends on: [Sink Handlers]
