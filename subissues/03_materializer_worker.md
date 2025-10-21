# [Sub-Issue] Develop Materializer Worker for Device to Host Transfer and Encoding

## Summary

Build the materializer worker that processes array payloads, performs device-to-host transfer, downscaling, quantization, and encoding (e.g., PNG/JPEG/MP4), preparing metrics events for downstream sinks.

## Motivation

Offloading device transfer and encoding to a background worker keeps the metrics API non-blocking and ensures efficient handling of large or complex payloads, especially for images and videos.

## Implementation Plan

- Implement a background thread or process for materialization.
- Integrate with the staging queue: worker dequeues events, processes payloads.
- Support device→host transfer for JAX/NumPy arrays.
- Apply downscale/quantize policies as needed.
- Encode images/videos to host-resident formats.
- Pass processed events to sink handlers.

## API Sketch

```python
class MaterializerWorker(Thread):
    def run(self): ...
    def process_event(self, event: MetricEvent) -> MetricEvent: ...
```

## Tests

- Test device→host transfer for various array types.
- Validate encoding correctness and performance.
- Check worker thread/process lifecycle and error handling.

## Dependencies

Depends on: [Staging Queue and Policies]
