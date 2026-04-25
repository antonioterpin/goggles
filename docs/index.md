# Goggles documentation hub

This is the single source of truth for all Goggles development documentation. Start here.

User-facing usage lives in the top-level [README](../README.md). This hub is aimed at contributors and agents working on the code.

---

## Guides

| Document | What it covers |
|---|---|
| [contributing.md](guides/contributing.md) | Development setup, branch workflow, PR preparation, commit style |
| [agent-development.md](guides/agent-development.md) | Using Goggles with Claude Code, GitHub Copilot, and custom agents |
| [architecture.md](guides/architecture.md) | Package layout, public surface, and subsystem responsibilities |
| [environment-variables.md](guides/environment-variables.md) | Runtime knobs (`GOGGLES_ASYNC`, `GOGGLES_SOCKET`, `GOGGLES_SHM_THRESHOLD`, ...) |

---

## Standards

One file per topic -- all code must satisfy these:

| Standard | Topic |
|---|---|
| [environment-tooling.md](standards/environment-tooling.md) | Always use `uv run`; virtual-env discipline |
| [code-quality.md](standards/code-quality.md) | Pre-commit gates, quality checklist |
| [code-clarity.md](standards/code-clarity.md) | Naming, function length, indentation, readability |
| [code-organization.md](standards/code-organization.md) | Module layout and ownership rules |
| [typing-docstrings.md](standards/typing-docstrings.md) | Modern type hints (PEP 585/604), Google-style docstrings |
| [testing.md](standards/testing.md) | TDD, Arrange/Act/Assert, coverage, hypothesis |
| [api-design.md](standards/api-design.md) | Public API contracts, backward compatibility |
| [change-scope.md](standards/change-scope.md) | What belongs in a single PR |
| [version-control.md](standards/version-control.md) | Commit messages, branch naming |
| [linting-formatting.md](standards/linting-formatting.md) | Ruff, pydoclint, basedpyright config |
| [exploration-validation.md](standards/exploration-validation.md) | Scratch scripts and API validation |
| [jax-numerical.md](standards/jax-numerical.md) | Optional JAX history subsystem conventions |

---

## Workflows

Step-by-step procedures for common tasks:

| Workflow | When to use |
|---|---|
| [orientation.md](workflows/orientation.md) | First time on the project, or returning after a long absence |
| [feature.md](workflows/feature.md) | Implementing a new feature (TDD approach) |
| [bugfix.md](workflows/bugfix.md) | Fixing a reported bug |
| [refactor.md](workflows/refactor.md) | Improving structure without changing behavior |
| [api-validation.md](workflows/api-validation.md) | Exploring external APIs or uncertain design |
| [docs.md](workflows/docs.md) | Documentation-only changes |

---

## Agent personas

| Persona | Purpose |
|---|---|
| [agents/implementer.md](agents/implementer.md) | Guidance for the implementing agent role |
| [agents/reviewer.md](agents/reviewer.md) | Guidance for the reviewing agent role |

---

## Quick reference

```bash
# Run tests
uv run pytest

# Lint and format (runs on commit)
uv run pre-commit run --all-files

# Full pre-push checks (type-check + tests)
uv run pre-commit run --all-files --hook-stage pre-push

# Add a dependency
uv add <package>          # runtime
uv add --dev <package>    # dev-only
```
