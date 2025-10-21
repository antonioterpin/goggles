# [Sub-Issue] Implement Sink Handlers for W&B, JSONL, and TensorBoard

## Summary

Create handlers to route processed metrics events to configured sinks: W&B (primary), JSONL, and TensorBoard (optional). Each handler should batch, retry, and record per-sink status/errors.

## Motivation

Supporting multiple sinks with consistent semantics enables flexible metrics logging and integration with external tools. Robust handlers ensure reliability and observability.

## Implementation Plan

- Implement W&B handler: log flat dicts, manage run context.
- Implement JSONL handler: write structured records to file.
- Implement TensorBoard handler: log events to TB format.
- Support batching, retry/backoff, and error tracking.
- Integrate handlers with materializer output.
- Make sink configuration driven by YAML or settings dict.

## API Sketch

```python
class SinkHandler:
    def handle(self, event: MetricEvent): ...
class WandbHandler(SinkHandler): ...
class JsonlHandler(SinkHandler): ...
class TensorBoardHandler(SinkHandler): ...
```

## Tests

- Validate correct output for each sink.
- Test error handling and retry logic.
- Check configuration toggling of sinks.

## Dependencies

Depends on: [Materializer Worker]
