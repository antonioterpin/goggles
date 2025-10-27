# Copilot Instructions for Goggles

## Project Overview
Goggles is a Python logging and monitoring library tailored for robotics and research experiments. It supports multi-process logging, terminal/file/W&B outputs, performance profiling, error tracing, and device-resident histories for JAX pipelines.

## Architecture & Key Components
- **Transitional Structure**: The project is currently migrating to a new API.
	- `goggles/__init__.py` is the official, up-to-date API entrypoint. All new features and integrations are exposed here.
	- `goggles/_core/` and `goggles/history/` contain the main implementations for logging, device history, and core logic.
	- **Legacy code**: Everything in the top-level `goggles/` folder except `__init__.py` is considered legacy. Avoid extending or modifying these unless porting features to the new API.
- **Device History**: See `goggles/history/` and `goggles/history/README.md` for GPU-resident buffers and JAX integration.
- **Examples**: `examples/` scripts show usage patterns and API features. No side effects on import; only `run(...)` attaches handlers.
- **Configuration**: The legacy configuration uses `.goggles-default.yaml` (see root and examples/). New approaches may supersede this; treat it as legacy unless otherwise specified.

## Developer Workflows
- **Install**: `pip install "goggles @ git+ssh://git@github.com/antonioterpin/goggles.git"`
- **Run Examples**: `python examples/01_basic_run.py` (see `examples/README.md`)
- **Testing**: Tests are in `tests/`. Use `pytest` for test runs.
- **JAX Integration**: For device history, install JAX (see main README for CUDA instructions).


## Patterns & Conventions
- **API Entry Points**: Use `run(...)` to start a logging run and configure sinks. Use `configure(...)` to set process-wide defaults before starting a run.
- **Logger Usage**: Get a structured logger via `get_logger(name, **bound)`. Use `.debug`, `.info`, `.warning`, `.error`, `.exception` methods. Persistent bound fields are supported.
- **Metrics/Media Logging**: The contracts for `scalar`, `image`, and `video` exist, but are not yet implemented in the new API.
- **Configuration**: Set options via `configure(...)` or arguments to `run(...)`. The legacy `.goggles-default.yaml` is not used in the new API.
- **No Decorators**: Profiling and error-tracing decorators (`@timeit`, `@trace_on_error`) are not yet available in the new API.
- **Single-process**: The current API is single-process; multi-process support is planned but not yet present.

## Integration Points
- **Weights & Biases**: Direct logging of metrics, images, videos.
- **JAX**: Optional dependency for GPU-resident histories. See README for installation and usage.

## Examples
- See `examples/` for scripts covering config loading, decorators, W&B logging, async scheduling, and shutdown.
- See `goggles/history/README.md` for JAX device history details.


## Tips for AI Agents
- Use only `__init__.py` for new features and integrations; legacy code is for reference/porting only.
- Configure logging via `configure(...)` or `run(...)` arguments, not `.goggles-default.yaml`.
- Use `get_logger(name, **bound)` for structured logging; bind persistent fields as needed.
- Metrics/media logging (`scalar`, `image`, `video`) are contracts only; not yet implemented.
- For JAX device history, require explicit shape/dtype in history specs (see `goggles/history/README.md`).
- Reference `examples/` for canonical usage of the new API.


# Mode Guidelines for GitHub Issue Generation

This section outlines the modes available for generating GitHub issues from design documents using AI agents. Each mode has specific input requirements, output formats, and instructions to ensure clarity and consistency.

## /issue_plan — Umbrella Issue Planner Mode

Convert a design document into a single **high-level GitHub issue** that outlines the overall goal and plans its sub-issues.

**Input:** A design or architecture document (Markdown).
**Output:** A single Markdown issue ready to post on GitHub **as a Markdown file**.

**Instructions:**
- Extract context, motivation, and goals from the design doc.
- Write a concise **Summary**, **Motivation**, and **Acceptance Criteria** section.
- Add a **Sub-Issues** section that lists all planned sub-issues as GitHub checklists, with:
	- Short but descriptive titles.
	- One-sentence purpose per sub-issue.
	- Logical execution order (e.g., outer → inner → integration).
	- Each checklist item should correspond to a sub-issue that will later be expanded by /issue_split.

**Formatting Example:**
```markdown
# [Feature] GPU-Resident History Module

## Summary

Introduce a GPU-first temporal memory module for Goggles, generalizing FlowGym’s EstimatorState.

## Motivation

Currently, temporal history is hardcoded in FlowGym. This limits modularity and reuse. Moving it
to Goggles aligns with its role as a GPU data handling framework.

## Acceptance Criteria

- Shared history buffers usable across FlowGym and other JAX projects.
- Batched initialization and sharding support.
- Full unit tests and integration demo in FlowGym.

## Sub-Issues

- [ ] Define outer contract and module API.
- [ ] Implement GPU-resident circular buffer backend.
- [ ] Integrate initialization and batching semantics.
- [ ] Add tests and FlowGym integration examples.
```

---

## /issue_split — Sub-Issue Expansion Mode

Expand a high-level umbrella issue into multiple **fully detailed sub-issues**, each ready for another agent or contributor to implement.

**Input:** The umbrella issue produced by /issue_plan.
**Output:** A list of GitHub-ready Markdown issues, one per sub-issue, **each rendered in its own Markdown file in a dedicated directory called subissues/**.

**Instructions:**
- For each checklist item in the umbrella issue:
	- Write a complete issue body with:
		- ## Summary — What the issue implements or changes.
		- ## Motivation — Why this step is necessary.
		- ## Implementation Plan — Key classes/functions, expected behavior, constraints.
		- ## API Sketch — Pseudocode or interface outline (if relevant).
		- ## Tests — Bullet list of test goals (unit, integration, edge cases).
		- ## Dependencies — Links to related or prerequisite issues.
- Ensure each sub-issue is self-contained and actionable.
- Maintain consistent tone and structure.

**Formatting Example:**
```markdown
# [Sub-Issue] Implement GPU-Resident Circular Buffer

## Summary

Create a circular buffer that stores temporal data directly on GPU for efficient reuse.

## Motivation

Temporal memory in FlowGym’s estimators currently lives on host memory, causing transfer overheads.
A GPU-resident buffer allows low-latency updates and sharded access.

## Implementation Plan

- Define `CircularBuffer` class in `goggles/history/buffer.py`.
- Support push/pop semantics with overwrite policy.
- Implement `.to_device()` and `.from_device()` utilities.

## API Sketch

```python
class CircularBuffer:
		def __init__(self, shape, dtype, device): ...
		def push(self, x): ...
		def get(self, n): ...
```

## Tests

- Test push/pop correctness.
- Test wrap-around overwrite behavior.
- Test device transfer consistency (`CPU↔GPU`).
- Benchmark latency vs NumPy baseline.

## Dependencies

Depends on: [Outer Contract Definition Issue]
