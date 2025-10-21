# [Sub-Issue] Implement Config-Driven Routing and Defaults

## Summary

Support configuration of queue size, rate limits, and enabled sinks via YAML or settings dict. Ensure the metrics system initializes with sensible defaults and applies config-driven routing.

## Motivation

Configurable routing and defaults allow users to tailor metrics logging to their needs, control resource usage, and enable/disable sinks as required.

## Implementation Plan

- Parse metrics config from YAML/settings dict.
- Apply queue size, rate limits, and sink enablement at initialization.
- Support per-type and per-sink config options.
- Document configuration options and defaults.

## API Sketch

```python
class MetricsConfig(TypedDict):
    queue_size: int
    rate_limits: dict
    sinks: dict
```

## Tests

- Validate config parsing and application.
- Test enabling/disabling sinks and rate limits.
- Check default behavior matches spec.

## Dependencies

Depends on: [Thin Wrappers]
