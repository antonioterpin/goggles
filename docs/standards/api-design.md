# API design & over-engineering

## Principles

- **Do not over-optimize or over-generalize early.**
- Prefer **simple, explicit APIs** with a single clear usage.
- Goggles is a *library*. Every new public symbol becomes part of the
  supported surface; prefer keeping helpers private until a second caller
  justifies exposure.

## Timing

Generalization should come **after real use cases**, not before.

## Public surface

- The public API is re-exported from [goggles/__init__.py](../../goggles/__init__.py).
- Anything under `goggles/_core/` is implementation detail. Do not import
  from `_core` in examples, docstrings, or user-facing docs.
- Adding or removing a symbol in `goggles.__all__` is a *breaking change*
  for downstream projects (fluidscontrol, flowbench, pinet, ...). Flag it
  clearly in the PR and bump the version per SemVer.
