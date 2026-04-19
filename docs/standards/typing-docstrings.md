# Typing & docstrings

## Type information placement

- **Type information belongs in function signatures**, not in docstrings.
- **Do not include type hints in `Args:` or `Returns:` sections.**
- Use **Google-style docstrings** with descriptions only.
- Pydoclint enforces this rule: `arg-type-hints-in-docstring = false`.

## Example

```python
def save_configuration(config: dict[str, Any], path: str) -> None:
    """Save a configuration dict to a YAML file.

    Args:
        config: Configuration to serialize.
        path: Destination path. Parent directories must exist.

    Raises:
        FileNotFoundError: If the parent directory does not exist.
    """
```

## Type ignore & lint suppressions

- **Never use `type: ignore` comments** without explicitly telling the
  user first. `pyproject.toml` has `enableTypeIgnoreComments = false` on
  pyright; adding one will also require changing policy.
- **Never use `# noqa` for specific rules** without explicitly telling
  the user first.
- Both are code debt and should only be used with awareness and
  acknowledgment.

## Documentation requirements

- All **public functions and classes must be documented**.
- Private helpers (leading underscore) may omit docstrings if their
  intent is obvious.
- Module-level docstrings are encouraged on all files (user-facing
  modules and tests alike).
- Docstring ignores configured in ruff:
  - `D100` (missing module docstring) -- allowed
  - `D104` (missing package `__init__` docstring) -- allowed
  - `D105` (missing docstring in magic method) -- allowed
  - `D107` (missing docstring in `__init__`) -- document at class level instead
  - All other `D*` rules apply.

## Shell command examples in docstrings

- **Never use doctest syntax (`>>>` / `...`) for shell commands.** `>>>`
  denotes a Python REPL prompt; using it for shell commands is
  semantically wrong, breaks copy-paste, and misleads tooling.
- Use a Sphinx **`.. code-block:: console`** directive with `$` as the
  shell prompt. This renders correctly in Sphinx and keeps commands
  copyable as a contiguous block.
- Use `r"""..."""` (raw docstring) when the block contains backslash
  line continuations, otherwise `\<newline>` is interpreted as a
  string-literal line continuation and collapses the lines.

### Example

```python
r"""Run the benchmark logger against a dedicated port.

Usage:

.. code-block:: console

    $ GOGGLES_PORT=8374 uv run python \
        -m tests.benchmark.benchmark_logger \
        --log-type scalar --num-logs 10000
"""
```
