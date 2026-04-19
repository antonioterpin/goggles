# JAX & numerical code

Goggles ships an **optional** JAX-based device-resident history subsystem
under [goggles/history/](../../goggles/history/), exposed via the `jax`
extras group. Rules in this file apply to that subsystem and any future
numerical code.

## Framework preferences

- **Prefer JAX** inside the history subsystem.
- Do not mix ML frameworks (PyTorch, TensorFlow) in the same module.
- Keep JAX **optional**: base install must not require JAX. Import inside
  `goggles/history/` is fine; top-level imports of JAX are not.

## Code style

- Write code with **functional style and explicit data flow** where
  possible. The history subsystem is already pure-functional; new
  additions should match.
- Updates to history buffers are JIT-safe: no Python-side state, no host
  synchronization during update.

## Shape conventions

- Batch-first: tensors are `(B, T, *shape)`.
- Reserved dimension names in local variables: `B`, `T`, `H`, `W`, `N`.

## Randomness & reproducibility

When randomness is involved:
- make it explicit via a JAX PRNGKey parameter
- set seeds at the top of examples/tests
- keep tests deterministic (no wall-clock RNG seeding inside the subsystem)
