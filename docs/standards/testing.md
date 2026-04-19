# Testing policy

## Test framework

- **Always use `pytest`** for testing.

## Test structure

- Test directories **mirror `goggles/`** so you can find tests by module path:
  - `tests/core/` for `goggles/_core/`
  - `tests/core/integrations/` for `goggles/_core/integrations/`
  - `tests/history/` for `goggles/history/`
  - `tests/benchmark/` for performance/regression benchmarks
- **Keep test files small and focused.** One file per concern or layer --
  for example `test_api.py`, `test_routing.py`, `test_logger.py`,
  `test_hang_resilience.py`.
- Each test file should have a **module-level docstring** explaining
  what behavior it verifies.

## Test design

- **Functionality-driven tests only.** Test behavior and contracts, not
  implementation details. Drop tests that just verify a constructor sets
  an attribute or that a type check fires.
- Use **multiple small unit tests** rather than one large test.
- Add **integration tests** (`tests/core/integrations/`) when behavior
  spans multiple handlers, processes, or the W&B backend.

## Testing best practices

- Prefer `pytest.mark.parametrize` over repeated tests.
- **Always check `tests/conftest.py`** and subdirectory `conftest.py`
  files for existing fixtures before adding new ones.
- **Always include a descriptive message in `assert` statements** to
  explain what exactly failed and why (for example,
  `assert x == y, f"Expected {y}, got {x}"`).
- Use `pytest.mark.skip(reason=...)` for tests that need adaptation,
  with a clear reason explaining what changed.
- Resilience / hang-recovery tests use the `resilience` marker. They
  are slower and may be disabled in fast iterations:
  `uv run pytest -m "not resilience"`.

## Multi-process tests

- Goggles runs across processes. Tests that spawn subprocesses must use
  a **unique `GOGGLES_PORT` per test run** (see `conftest.py` fixtures).
  Never hardcode a port in a test module.
- Use the `xdist_group` marker to force serial execution inside a group
  when tests share a port or global state.

## Running tests

```bash
# Full suite (matches CI and pre-push)
uv run pytest

# A subsystem
uv run pytest tests/core/integrations/

# Skip slow/disruptive resilience tests
uv run pytest -m "not resilience"

# Benchmarks only
uv run pytest tests/benchmark/
```
