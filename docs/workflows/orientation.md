---
description: First-time orientation before making changes
---

Use when working on the project for the first time, or returning after a long absence. Goal: build a mental model of the system before touching code.

1. **Read the architecture overview.**
   [`docs/guides/architecture.md`](../guides/architecture.md) covers the package layout, public surface, and the main subsystems (core routing, integrations, history, config, filters).

2. **Skim the public API.**
   Read [`goggles/__init__.py`](../../goggles/__init__.py) -- every symbol re-exported from there is part of the supported surface and will affect downstream projects if changed.

3. **Review the standards.**
   Skim [`docs/standards/`](../standards/) before writing code -- in particular
   [`code-organization.md`](../standards/code-organization.md) (public vs `_core`),
   [`testing.md`](../standards/testing.md) (multi-process port discipline), and
   [`linting-formatting.md`](../standards/linting-formatting.md) (no guarded imports).

4. **Confirm the environment works.**
   Run the test suite:
   ```bash
   uv run pytest
   ```
   All tests should pass before you write a single line.

5. **Pick the right workflow for your task.**

   | Task | Workflow |
   |---|---|
   | New feature | [`feature.md`](feature.md) |
   | Bug fix | [`bugfix.md`](bugfix.md) |
   | Refactor | [`refactor.md`](refactor.md) |
   | External API exploration | [`api-validation.md`](api-validation.md) |
   | Documentation only | [`docs.md`](docs.md) |

**Done criteria:**
- You can describe the role of `goggles/_core/routing.py`, `goggles/_core/logger.py`, and `goggles/_core/integrations/` without looking them up.
- You know whether the code you intend to change is public or internal.
- `uv run pytest` passes.
