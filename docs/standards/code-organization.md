# Code organization & readability

## Code style

- Prefer **clear, explicit code** over clever or compact code.
- Avoid introducing new abstractions unless they reduce complexity.
- Match existing patterns in the codebase before inventing new ones.

## Package layout

| Path | Purpose |
|---|---|
| `goggles/` | Public API: top-level modules re-exported from `goggles/__init__.py`. |
| `goggles/_core/` | Implementation details. Never import from here outside the package. |
| `goggles/_core/integrations/` | Backend handlers (console, local storage, W&B, ...). |
| `goggles/history/` | Optional JAX-based device-resident history subsystem. |
| `tests/` | Unit + benchmark + integration tests; mirror public modules where possible. |
| `examples/` | Runnable demos that drive the public API. |

## Internal vs public

- Anything not exported from `goggles/__init__.py` is considered internal.
- Rename or move internal symbols freely. Public symbols require a
  deprecation + version bump.
- New public exports must also appear in the relevant docs page and,
  where applicable, the top-level `README.md`.

## Handler scope naming

Loggers in Goggles carry a scope (for example `global`, `training`,
`episode_0`). When emitting logger names *inside* the library itself,
follow these rules:

- Scopes are free-form strings chosen by users, not library code.
- When the library creates its own logger (for example inside shutdown
  utilities), use `get_logger(__name__)` so the scope matches the module
  path under `goggles.`.
- Do not invent deeper scope hierarchies from within the library; let
  callers name their own scopes.
