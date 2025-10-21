# [Sub-Issue] Define Metrics Event Schema and Validation Logic

## Summary

Establish the accepted types, fields, and validation logic for metrics events in Goggles. This includes defining the schema for scalars, images, videos, and histograms, and implementing validation routines to ensure all events conform to the specification.

## Motivation

A well-defined schema is essential for robust metrics logging, preventing malformed or inconsistent data from entering the pipeline. Validation at the API boundary ensures downstream handlers and sinks receive predictable, correctly-typed events.

## Implementation Plan

- Define event schema as Python dataclasses or TypedDicts.
- Specify allowed types, required fields, and payload formats.
- Implement validation functions for each metric type.
- Integrate validation into the metrics API entrypoint (`push`).
- Document schema for contributors and users.

## API Sketch

```python
class MetricEvent(TypedDict):
    key: str
    type: Literal['scalar', 'image', 'video', 'histogram']
    step: int
    ts: float
    payload: Any
    tags: list[str]
    context: dict
    policy: Literal['DROP_OLDEST', 'DROP_NEWEST', 'MERGE']
```

## Tests

- Validate correct event construction for all types.
- Reject events with missing/invalid fields.
- Test edge cases (e.g., wrong dtype, shape).

## Dependencies

Depends on: None (foundational for all other sub-issues)
