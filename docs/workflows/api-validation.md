---
description: External API validation and integration
---

Use when interacting with a new external library API or uncertain behavior (for example a new W&B endpoint, a new transport for the routing layer, or a new JAX feature).

1. **Validate first in a scratch script** (not committed):
   - Write a small script to confirm the API, edge cases, and expected behavior.
   - Capture outputs, performance notes, and pitfalls.
2. Summarize findings in 1-5 bullets (for yourself) and decide the minimal integration.
3. **Write failing unit tests** in the test suite that define the desired API and expected behavior (based on your scratch script findings).
4. Run tests to confirm they fail:
   ```bash
   uv run pytest
   ```
5. Implement the **minimal core change** in `goggles/...` to make the tests pass.
   - Do not over-generalize or add multiple input modes prematurely.
6. Run the commit-stage gate:
   ```bash
   uv run pre-commit run --all-files
   ```
7. Run all tests (including the new ones):
   ```bash
   uv run pytest
   ```
8. Run the pre-push gate (basedpyright + full test suite):
   ```bash
   uv run pre-commit run --all-files --hook-stage pre-push
   ```
9. Delete the scratch script. Encode conclusions:
   - small local rationale -> comment (WHY)
   - API behavior -> docstring
   - important design decision -> `docs/` entry (if applicable)

**Done criteria:**
- `uv run pre-commit run --all-files` passes
- `uv run pytest` passes
- `uv run pre-commit run --all-files --hook-stage pre-push` passes
- No temporary scripts committed
- Public APIs are documented (without docstring type hints)
