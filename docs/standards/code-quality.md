# Code quality gates

All modified code must pass the following quality gates before a task is considered complete.

## Required checks

- **Ruff and Ruff-format must pass** on all modified files (lint + format).
- **Pydoclint must pass** (Google-style docstring consistency).
- **BasedPyright must pass** (basic mode).
- **Pytest must pass** (the full test suite, including benchmarks).

## Running quality gates

Pre-commit splits checks into two stages:

- **`pre-commit` (runs on every `git commit`)**: ruff, ruff-format,
  pydoclint, generic hygiene. Fast.
- **`pre-push` (runs on every `git push`)**: basedpyright + pytest on
  top of the commit-stage checks. Slower, comprehensive.

Before finishing a task, **run both stages**:

```bash
uv run pre-commit run --all-files
uv run pre-commit run --all-files --hook-stage pre-push
```

**A task is NOT complete unless both invocations pass.**

All linting and formatting rules live in `pyproject.toml`. Do not
override rule selection via CLI flags in pre-commit or scripts.
Pre-commit controls *when* tools run, not *what* rules apply.
