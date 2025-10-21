# [Sub-Issue] Add Observability: Queue Depth, Drop Counters, Per-Sink Errors

## Summary

Track and expose metrics system health indicators, including queue depth, drop counters by reason/type, and per-sink errors. Provide interfaces for querying and reporting these metrics.

## Motivation

Observability is critical for debugging, tuning, and ensuring reliability of the metrics system. Exposing health indicators allows users and developers to monitor performance and diagnose issues.

## Implementation Plan

- Instrument queue and handlers to track depth, drops, and errors.
- Provide API for querying health metrics.
- Integrate with logging and/or external monitoring tools.
- Document observability features.

## API Sketch

```python
class MetricsObservability:
    def get_stats(self) -> dict: ...
```

## Tests

- Validate correct tracking of all observability metrics.
- Test reporting and querying interfaces.

## Dependencies

Depends on: [Config Routing and Defaults]
