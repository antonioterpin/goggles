# [Sub-Issue] Add Thin Wrapper Functions (`scalar`, `image`, `video`) for Ergonomic Usage

## Summary

Provide helper functions (`scalar`, `image`, `video`) that build typed payloads and delegate to the metrics API, improving ergonomics and backward compatibility for users.

## Motivation

Thin wrappers make metrics logging more convenient and familiar, supporting legacy usage patterns while ensuring all events go through the unified metrics API.

## Implementation Plan

- Implement `gg.scalar`, `gg.image`, `gg.video` functions.
- Build typed payloads and call the current logger’s `push()`.
- Mark direct W&B calls in wrappers as deprecated.
- Update documentation and examples.

## API Sketch

```python
def scalar(key, value, step, **opts): ...
def image(key, array, step, **opts): ...
def video(key, array, step, **opts): ...
```

## Tests

- Validate wrapper output for all metric types.
- Test backward compatibility with legacy usage.

## Dependencies

Depends on: [Integration with Logger and Run Lifecycle]
