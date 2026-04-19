# Architecture

This page is the map contributors and agents should consult before making
changes. It covers the package layout, what is public vs internal, and
the main subsystems. User-facing usage lives in the top-level
[README](../../README.md).

## Package layout

```
goggles/
|-- __init__.py        # Public API: re-exports + runtime patches
|-- config.py          # YAML config loading/saving (PrettyConfig)
|-- filters.py         # Built-in signal filters (median, std-reject, ...)
|-- media.py           # Image/video/vector-field helpers
|-- shutdown.py        # GracefulShutdown utility
|-- types.py           # Event, Kind, Image, Video, VectorField, Metrics
|-- validation.py      # Configuration validation helpers
|-- history/           # Optional JAX device-resident history buffers
|   |-- buffer.py
|   |-- spec.py
|   |-- types.py
|   `-- utils.py
`-- _core/             # Implementation detail; do not import outside
    |-- logger.py            # CoreTextLogger / CoreGogglesLogger impls
    |-- routing.py           # GogglesClient / EventBus routing
    |-- decorators.py        # @timeit / @trace_on_error impls
    `-- integrations/
        |-- console.py        # ConsoleHandler
        |-- storage.py        # LocalStorageHandler
        `-- wandb.py          # WandBHandler (W&B extra)
```

### Public vs internal

Anything reachable from `goggles.__all__` is **public**. The current
public surface is (see [goggles/__init__.py](../../goggles/__init__.py)):

- Log levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
- Loggers: `get_logger`, `TextLogger`, `GogglesLogger`
- Event model: `Event`, `Kind`, `Metrics`, `Image`, `Video`,
  `Vector`, `VectorField`
- Handlers: `ConsoleHandler`, `LocalStorageHandler`, `WandBHandler`
- Bus management: `attach`, `detach`, `register_handler`
- Decorators: `timeit`, `trace_on_error`
- Config: `PrettyConfig`, `load_configuration`, `save_configuration`
- Shutdown: `GracefulShutdown`
- Submodule: `filters`

Everything else is **internal**. Moving, renaming, or deleting
internal symbols does not require a deprecation cycle. Public changes
do (see [api-design.md](../standards/api-design.md)).

## Subsystems

### 1. Logger & bus routing (`_core/logger.py`, `_core/routing.py`)

Goggles maintains a **process-wide EventBus** that routes events to
attached handlers. The user-facing entry points are `get_logger()` and
`attach()` in `goggles/__init__.py`.

- Loggers carry a `scope` (free-form string, default `global`). Events
  are routed to handlers attached to matching scopes.
- Scope matching supports hierarchy via dot notation: a handler on
  `global` also receives events from `global.run1`.
- Events include logs (text) and structured data (`scalar`, `image`,
  `video`, `vector_field`, `histogram`, ...).

### 2. Event model (`types.py`)

`Event`, `Kind`, and the payload types (`Image`, `Video`,
`VectorField`, `Metrics`, `Vector`) define the wire format between
loggers and handlers. They are serializable so they can cross
process boundaries via portal/TinyROS.

### 3. Handlers / integrations (`_core/integrations/`)

Each handler implements the `Handler` protocol (see
`goggles/__init__.py`). Built-in handlers:

- **ConsoleHandler**: colored terminal output.
- **LocalStorageHandler**: JSON-Lines on disk.
- **WandBHandler** (`wandb` extra): Weights & Biases integration with
  multi-run grouping.

Adding a new handler: subclass an existing handler or implement the
`Handler` protocol directly, then register it with
`register_handler()` so it can be serialized for transport across the
bus.

### 4. Transport & resilience (`goggles/__init__.py`, top)

Goggles uses the `portal` library for inter-process RPC. The module
import patches portal in three places to prevent memory leaks and
hangs:

1. `SendBuffer.send` -> propagate `ConnectionResetError`.
2. `ServerSocket._loop` -> disconnect clients on write errors.
3. `ClientSocket._loop` -> clear send queue on disconnect.
4. `Client.call` -> honor `GOGGLES_TRANSPORT_TIMEOUT` while waiting
   for in-flight requests.

These patches are temporary. Remove them once upstream portal fixes
land (see the `# NOTE` block at the top of the file).

### 5. History subsystem (`history/`, optional)

Device-resident JAX buffers for metrics tracked during long GPU runs.
Activated via the `jax` extras group. Pure-functional,
JIT-safe updates; no host-device sync during `update_history`.

See [jax-numerical.md](../standards/jax-numerical.md) for the
conventions that apply inside this subsystem.

### 6. Config (`config.py`)

`load_configuration` / `save_configuration` provide YAML IO with
`PrettyConfig` for readable serialization. `validation.py` hosts
schema helpers used by handlers with structured config (for example
the WandB handler).

### 7. Shutdown (`shutdown.py`)

`GracefulShutdown` is the user-facing context manager for clean
teardown. Internally, `goggles.finish()` calls into the bus to flush
and close all handlers with a configurable timeout
(`GOGGLES_SHUTDOWN_TIMEOUT`).

## Cross-cutting concerns

- **Environment variables**:
  - `GOGGLES_PORT` (default `2304`) -- bus TCP port.
  - `GOGGLES_HOST` (default `localhost`) -- bus hostname.
  - `GOGGLES_ASYNC` (default `1`) -- async emit mode.
  - `GOGGLES_TRANSPORT_TIMEOUT` (default `30.0s`) -- client call timeout.
  - `GOGGLES_SHUTDOWN_TIMEOUT` (default `5.0s`) -- shutdown timeout.
  - `GOGGLES_SUPPRESS_CONNECTIVITY_LOGS` (default `1`) -- silence
    "Dropping message" chatter from portal.

- **Tests mirror this layout** (`tests/core/`, `tests/core/integrations/`,
  `tests/history/`, `tests/benchmark/`).

## Where to make changes

| Change | Where |
|---|---|
| New handler / integration | `goggles/_core/integrations/<name>.py`, register in `goggles/_core/integrations/__init__.py`, re-export from `goggles/__init__.py`. |
| New filter | `goggles/filters.py` (public module). |
| New logger method | Update the `TextLogger` / `DataLogger` / `GogglesLogger` Protocols in `goggles/__init__.py`, then `_core/logger.py`. |
| New config helper | `goggles/config.py` (keep the YAML-focused surface small). |
| New history buffer op | `goggles/history/buffer.py` + types in `goggles/history/types.py`. |
| Transport workaround | `goggles/__init__.py` under the existing monkey-patch block; leave a removal NOTE. |
