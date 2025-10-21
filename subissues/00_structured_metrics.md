# [Feature] Goggles Metrics v1.0 — Unified Structured Metrics API

## Summary

Implement a unified, non-blocking metrics API for Goggles, supporting structured logging of scalars, images, videos, and histograms. The new system will decouple metric producers from sinks using a queue and worker model, enforce a well-defined schema, and provide robust integration with existing text logging and run lifecycle.

## Motivation

Current metrics logging in Goggles is tightly coupled to specific sinks (e.g., W&B) and lacks a unified schema, queueing, and rate-limiting. This limits extensibility, performance, and reliability. A new metrics API will enable multi-sink fan-out, predictable performance, and seamless integration with the existing logging system, laying the foundation for future features like advanced routing and multi-process support.

## Acceptance Criteria

- Unified `logger.push(step, metrics)` API with schema validation and non-blocking enqueue.
- Bounded staging queue with per-type rate limits and drop/coalesce policies.
- Materializer worker for device→host transfer and encoding.
- Config-driven sink routing (W&B primary, JSONL/TensorBoard optional).
- Deterministic event attribution (step, timestamp, run_id, context).
- Observability: queue depth, drop counters, per-sink errors.
- Integration with run lifecycle and BoundLogger context.
- Thin wrappers (`scalar`, `image`, `video`) delegate to `push`.
- Full unit tests and example usage.

## Sub-Issues

- [ ] Define metrics event schema and validation logic.
      Establish the accepted types, fields, and validation for metrics events.
- [ ] Implement bounded staging queue with rate limits and drop/coalesce policies.
      Create the queue system for metrics events, enforcing limits and drop strategies.
- [ ] Develop materializer worker for device→host transfer and encoding.
      Build the worker that processes array payloads and prepares them for sinks.
- [ ] Implement sink handlers for W&B (primary), JSONL, and TensorBoard (optional).
      Create handlers to route metrics events to configured sinks.
- [ ] Integrate metrics API with BoundLogger and run lifecycle.
      Extend BoundLogger and run context to support metrics push and context binding.
- [ ] Add thin wrapper functions (`scalar`, `image`, `video`) for ergonomic usage.
      Provide helper functions that build typed payloads and call the metrics API.
- [ ] Implement config-driven routing and defaults.
      Support configuration of queue size, rate limits, and enabled sinks.
- [ ] Add observability: queue depth, drop counters, per-sink errors.
      Track and expose metrics system health and errors.
- [ ] Write unit tests and integration examples.
      Ensure correctness and provide usage demonstrations.
