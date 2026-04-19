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

## Port isolation for multi-process tests

- Goggles uses a shared TCP port (`GOGGLES_PORT`) for inter-process log
  routing. When starting processes by hand (benchmarks, reproduction
  scripts, manual experiments), pick a free port rather than reusing a
  default -- otherwise a lingering server from a previous run will
  capture the new connection.
  ```bash
  GOGGLES_PORT="$(uv run python -c 'import socket; s=socket.socket(); s.bind((\"\", 0)); print(s.getsockname()[1]); s.close()')" \
      uv run python -m tests.benchmark.benchmark_logger --log-type scalar
  ```
