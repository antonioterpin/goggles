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

## Docstrings stand alone

Docstrings describe **what the symbol is and how to use it** for a
reader who has no idea about the surrounding repo history, the PR that
introduced it, or what the code looked like before. They are read
out-of-context (IDE hovers, generated API docs, `help()`), so anything
that depends on context the reader does not have is noise.

**Do not include in docstrings or doc-comments:**

- **History or narrative.** No "historically", "previously",
  "used to", "before this fix", "the old behavior was…". The git log
  and PR description carry that. A docstring describing what code
  *used to* do ages instantly: the next reader has no way to verify
  the comparison and may not even realize the "old" thing is gone.
- **Issue / PR / commit references.** No `See #70.`, `fixes #123`,
  `(issue #N)`, `as discussed in PR #M`. Issue links belong in the PR
  body and the commit message. In a docstring they rot — issues get
  closed, renumbered, or lose context — and they push the reader off
  to GitHub for information that should be on the page.
- **Cross-symbol comparisons that require knowing the other symbol.**
  Avoid framings like *"Same monotonic-step tracker as
  ``LocalStorage``"*, *"Like the old ``Foo`` but with X"*. Describe
  what *this* symbol does. If two implementations genuinely share a
  contract, describe the contract; do not point at a sibling.
- **Justifications phrased as "why we changed it".** Inside a
  docstring, the reader sees the current code as the only code that
  exists. Frame intent as a present-tense invariant
  ("Console output is for humans reading in arrival order"), not as
  a delta from a previous version.

**OK to include:**

- Hidden constraints, invariants, thread-safety guarantees,
  performance notes ("called on the producer hot path"), platform
  caveats. Anything a reader needs to use the symbol *correctly*.
- A brief mention of an external standard the symbol mirrors
  ("mirrors wandb's per-run monotonicity contract") — this is naming
  the contract, not pointing at our own history.

These rules apply equally to **inline `#` comments at definition
sites** (e.g. comments above a field initializer), since they are
read in the same out-of-context way.

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

    $ GOGGLES_SOCKET=/tmp/goggles-bench.sock uv run python \
        examples/105_benchmark.py \
        log_type=scalar num_logs=10000
"""
```
