---
description: Standard feature implementation
---

Use for ordinary feature work that doesn't require external API exploration.

1. Define the desired API and behavior by **writing failing unit tests** (and integration tests if the feature spans handlers or processes).
2. Run tests to confirm they fail:
   ```bash
   uv run pytest
   ```
3. Identify the ownership module under `goggles/` and implement the **minimal change** to make the tests pass.
   - Keep new helpers private (leading underscore, `_core/`) unless there is an immediate user need.
4. Run the commit-stage gate:
   ```bash
   uv run pre-commit run --all-files
   ```
5. Run tests to confirm they pass:
   ```bash
   uv run pytest
   ```
6. Run the pre-push gate (basedpyright + full test suite):
   ```bash
   uv run pre-commit run --all-files --hook-stage pre-push
   ```
7. Add/adjust docstrings and documentation as needed.
   - Docstrings describe behavior; type info stays in signatures.
   - If the feature adds a public symbol, update `goggles/__init__.py` and the relevant README/docs section.

**Done criteria:**
- `uv run pre-commit run --all-files` passes
- `uv run pre-commit run --all-files --hook-stage pre-push` passes
- Docs are updated for public surface changes
