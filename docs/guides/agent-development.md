# Agent-driven development guide

This guide explains how to use the Goggles documentation and workflows with different agentic systems (Claude Code, GitHub Copilot, custom agents, ...).

## Overview

Goggles is optimized for agent-driven development. All rules, workflows, and procedures are centralized in the `docs/` folder, and compatibility layers in `.agent/`, `.github/`, and `.claude/` reference them.

## Infrastructure at a glance

| System | Primary entry point | Always-on rules behavior | Workflow selection | Canonical rule source |
|---|---|---|---|---|
| Claude Code | `.claude/CLAUDE.md` | Loads project instructions from `.claude/CLAUDE.md`, which references `docs/standards/` | User prompt references `docs/workflows/*` | `docs/standards/` |
| GitHub Copilot | `.github/copilot-instructions.md` | Instruction file enforces orientation and links standards/workflows | Prompt and task context plus linked workflow docs | `docs/standards/` |
| Antigravity / other slash-command agents | `.agent/rules/rules.md` + `.agent/workflows/*` | `.agent/rules/rules.md` is `always_on` and points to standards | Slash commands or workflow wrappers in `.agent/workflows/*` | `docs/standards/` |
| Custom agents | Integrator-defined | Must be configured to consume project entry points and `docs/` | Integrator-defined | `docs/standards/` |

Interpretation: wrappers define how agents discover and trigger behavior; `docs/standards/` remains the single source of truth for policy.

## Using with Claude Code

### Entry point

Claude Code reads its configuration from:
1. `.claude/CLAUDE.md` (this project's rules) - **Start here**
2. `docs/` (full documentation)
3. `.agent/` (legacy workflows mirror)

### Quick start

When working with Claude Code:

```
User: "I want to add a handler for MLflow. Please follow the feature workflow."

Claude Code will:
1. Read .claude/CLAUDE.md for project context
2. Reference docs/workflows/feature.md for step-by-step procedure
3. Follow docs/standards/ for code quality rules
4. Return results
```

### Configuration files

- **`.claude/CLAUDE.md`** - Main rules file with links to all documentation
- **`.claude/settings.local.json`** - Local permissions (not agent rules)

### Running workflows

You can reference workflows directly:
- Feature work: See [docs/workflows/feature.md](../workflows/feature.md)
- Bug fixes: See [docs/workflows/bugfix.md](../workflows/bugfix.md)

## Using with GitHub Copilot

### Entry point

GitHub Copilot reads its configuration from `.github/copilot-instructions.md` (optional; create when/if Copilot is used).

### How it works

The copilot-instructions.md file is a thin wrapper that references:
- `docs/standards/` for all rules
- `docs/workflows/` for step-by-step procedures
- `docs/guides/` for architecture and contribution guides

## Using with slash-command agents (Antigravity, Cursor, ...)

### Entry point

Wrapper agents read from:
1. `.agent/rules/rules.md` (project rules wrapper)
2. `.agent/workflows/` (workflow references)

### How it works

- **`.agent/rules/rules.md`** references `docs/standards/` for all coding standards.
- **`.agent/workflows/`** contains thin wrappers that point to `docs/workflows/`.
- **Slash commands** like `/feature`, `/bugfix`, `/orientation` provide quick access to workflows.

Available slash commands (mapped to `.agent/workflows/*.md`):
- `/feature` - Standard feature implementation
- `/bugfix` - Bug fix workflow
- `/refactor` - Code refactoring
- `/api-validation` - External API exploration
- `/docs` - Documentation changes
- `/orientation` - Project orientation

## Using with custom agents

### Integration pattern

If you're building a custom agent for Goggles:

1. **Read `.claude/CLAUDE.md`** to understand the project
2. **Reference `docs/index.md`** as your documentation hub
3. **Use `docs/workflows/`** for step-by-step procedures
4. **Check `docs/standards/`** for specific rules

## Workflow reference

All workflows follow the same structure:

### Feature workflow
**Use when:** Implementing a standard feature without external API exploration
- Location: [docs/workflows/feature.md](../workflows/feature.md)
- Process: Write tests -> implement -> verify -> document

### Bugfix workflow
**Use when:** Fixing a reported bug
- Location: [docs/workflows/bugfix.md](../workflows/bugfix.md)
- Process: Reproduce -> fix -> verify -> document

### Refactor workflow
**Use when:** Improving structure without changing behavior
- Location: [docs/workflows/refactor.md](../workflows/refactor.md)
- Process: Understand current structure -> refactor -> verify behavior unchanged

### API validation workflow
**Use when:** Exploring external APIs or uncertain about design
- Location: [docs/workflows/api-validation.md](../workflows/api-validation.md)
- Process: Create scratch script -> validate -> document findings -> clean up

### Documentation workflow
**Use when:** Making documentation-only changes
- Location: [docs/workflows/docs.md](../workflows/docs.md)
- Process: Edit docs -> verify links -> commit

### Orientation workflow
**Use when:** First time working on the project, or returning after a long absence
- Location: [docs/workflows/orientation.md](../workflows/orientation.md)
- Process: Read architecture -> skim public API -> run tests -> pick workflow

## Standards quick reference

When implementing, keep these standards in mind:

### Tools
- **Always use `uv run`** for all commands
- See [docs/standards/environment-tooling.md](../standards/environment-tooling.md)

### Quality gates
- Must pass `uv run pre-commit run --all-files` (commit stage: ruff, pydoclint, hygiene)
- Must pass `uv run pre-commit run --all-files --hook-stage pre-push` (adds basedpyright + pytest)
- See [docs/standards/code-quality.md](../standards/code-quality.md)

### Code style
- Google-style docstrings with descriptions only
- Type hints in function signatures, not docstrings
- See [docs/standards/typing-docstrings.md](../standards/typing-docstrings.md)

### Testing
- Use pytest; parametrize over repeating tests
- Include descriptive assert messages
- See [docs/standards/testing.md](../standards/testing.md)

### JAX / numerical (optional history subsystem)
- Prefer JAX inside `goggles/history/`
- Make randomness explicit via PRNGKey; keep tests deterministic
- See [docs/standards/jax-numerical.md](../standards/jax-numerical.md)

## Architecture reference

For understanding the codebase structure:

- **Full guide:** [docs/guides/architecture.md](architecture.md)
- **Key components:**
  - `goggles/` - Public API
  - `goggles/_core/` - Implementation detail
  - `goggles/_core/integrations/` - Handlers (console, local storage, W&B)
  - `goggles/history/` - Optional JAX history subsystem
  - `tests/` - Test suite
  - `examples/` - Runnable demos

## Making changes to documentation

If you need to update project rules or workflows:

1. **Make changes in `docs/`** - This is the single source of truth
2. **All agents will automatically see the updates**
3. **No need to update multiple files** - The wrappers in `.agent/`, `.github/`, and `.claude/` reference the central docs

This ensures consistency across all agentic systems.

## Troubleshooting

### "I don't know what workflow to use"
Start with [docs/workflows/orientation.md](../workflows/orientation.md) to understand the project, then pick the workflow that matches your task.

### "Where do I find the architecture?"
See [docs/guides/architecture.md](architecture.md) for a complete overview of the codebase structure and responsibilities.

### "What rules apply to my code?"
Check [docs/standards/](../standards/) for the specific rule you're unsure about. Each file covers one topic.

## For project maintainers

When updating project standards:

1. Update the relevant file in `docs/standards/`
2. No need to update `.agent/rules/rules.md` or `.claude/CLAUDE.md` unless the change concerns the agent-discovery mechanism itself
3. All agents will automatically see the changes
4. Consider updating `docs/index.md` if you add new standards

When creating new workflows:

1. Create the file in `docs/workflows/`
2. Add a link to `docs/index.md`
3. Add a thin wrapper in `.agent/workflows/` so slash-command agents can trigger it
