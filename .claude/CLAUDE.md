# Goggles development

All documentation is centralized in `docs/`. Claude will discover what it needs based on your task.

## Essential commands

```bash
# Test
uv run pytest

# Lint and format (commit-stage checks)
uv run pre-commit run --all-files

# Type check + full test suite (pre-push stage)
uv run pre-commit run --all-files --hook-stage pre-push

# Add dependency
uv add <package>              # Runtime
uv add --dev <package>        # Dev dependency
uv add --optional jax <pkg>   # Optional JAX extra
uv add --optional wandb <pkg> # Optional W&B extra
```

## Workflows

When starting a task, refer to the appropriate workflow from `docs/workflows/`:

- **Feature**: see `docs/workflows/feature.md`
- **Bugfix**: see `docs/workflows/bugfix.md`
- **Refactor**: see `docs/workflows/refactor.md`
- **API validation**: see `docs/workflows/api-validation.md`
- **Documentation**: see `docs/workflows/docs.md`
- **Orientation (first time)**: see `docs/workflows/orientation.md`

## Standards

All code must follow standards in `docs/standards/`:

- Always use `uv run` for commands (never assume global installs)
- Run all quality gates before finishing (pre-commit + pre-push hooks must pass)
- Write tests first (TDD approach)
- Google-style docstrings, types in signatures only
- Keep base install light: JAX/W&B are optional extras
- Never import from `goggles/_core/` outside the package itself

## Architecture

See `docs/guides/architecture.md` for codebase structure and subsystem responsibilities.

## Key project-specific rules

- **Do not push to remote without explicit user authorization.**
- **Commit messages**: follow the `.gitmessage` template (Conventional Commits).
- **Allowed uppercase local vars**: `B`, `T`, `H`, `W`, `N` (tensor dimensions).
- **No guarded imports** (`try/except ImportError`): express optionality via extras in `pyproject.toml` instead.
- **Public vs private**: symbols not in `goggles.__all__` are internal and can be moved/renamed freely.

## Documentation

For comprehensive documentation, see `docs/index.md`.
