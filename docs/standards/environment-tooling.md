# Environment & tooling

## Using uv

- **Always use `uv run`** to execute tools and scripts.
  - Never assume global installs.
  - Examples:
    ```bash
    uv run python
    uv run pytest
    uv run ruff
    ```

## Managing dependencies

- **Add dependencies with uv (don't edit `pyproject.toml` by hand unless necessary).**
  - Runtime dependency:
    ```bash
    uv add <package>
    ```
  - Dev dependency (linters/formatters/type checkers/test tooling):
    ```bash
    uv add --dev <package>
    ```

- **If you add a new external dependency (new import), you must:**
  1. Add it with `uv add` / `uv add --dev`.
  2. Place it in the correct extras group if it is optional: `jax` for
     JAX-backed features, `wandb` for the W&B handler, `dev` for dev
     tooling only. Goggles keeps the base install light on purpose.
  3. Ensure the lockfile is updated (`uv lock` if needed).
  4. Commit both `pyproject.toml` and `uv.lock`.

## Socket isolation for multi-process tests

- Goggles uses a Unix domain socket (`GOGGLES_SOCKET`) for inter-process
  log routing. When starting processes by hand (benchmarks, reproduction
  scripts, manual experiments), pick a unique path rather than reusing
  the default -- otherwise a lingering host from a previous run (or an
  unrelated project) will capture the new connection.
  ```bash
  GOGGLES_SOCKET="/tmp/goggles-$$.sock" \
      uv run python examples/105_benchmark.py log_type=scalar
  ```
