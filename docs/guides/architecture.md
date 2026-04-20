# Architecture

This page is the map contributors and agents should consult before making
changes. It covers the package layout, what is public vs internal, and
the main subsystems. User-facing usage lives in the top-level
[README](../../README.md).

## Package layout

```
goggles/
|-- __init__.py        # Public API: re-exports + EventBus
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
    |-- routing.py           # Transport singleton (get_bus / reset_bus)
    |-- transport.py         # Transport protocol + LocalTransport (UDS)
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

### 1. Logger & bus routing (`_core/logger.py`, `_core/routing.py`, `_core/transport.py`)

Goggles maintains a **single EventBus per machine**: one host process
owns it, the rest connect as clients. The user-facing entry points are
`get_logger()` and `attach()` in `goggles/__init__.py`.

- `_core/logger.py` holds the user-visible `CoreTextLogger` /
  `CoreGogglesLogger` implementations; each call builds an `Event` and
  hands it to the transport via `emit` (async) or `emit_sync`.
- `_core/routing.py` exposes `get_bus()` / `reset_bus()`, the
  process-wide transport singleton.
- `_core/transport.py` defines the `Transport` protocol and the default
  `LocalTransport` (see §4).

- Loggers carry a `scope` (free-form string, default `global`). Events
  are routed to handlers attached to matching scopes.
- Scope matching supports hierarchy via dot notation: a handler on
  `global` also receives events from `global.run1`.
- Events include logs (text) and structured data (`scalar`, `image`,
  `video`, `vector_field`, `histogram`, ...).

### 2. Event model (`types.py`)

`Event`, `Kind`, and the payload types (`Image`, `Video`,
`VectorField`, `Metrics`, `Vector`) define the wire format between
loggers and handlers. They are picklable so they can cross process
boundaries on the same machine via `LocalTransport`.

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

### 4. Transport (`_core/transport.py`)

Goggles uses a Unix-domain-socket transport (`LocalTransport`) to
route events within a machine. The first process to bind the socket
path (default `${XDG_RUNTIME_DIR:-/tmp}/goggles-<uid>.sock`, override
via `GOGGLES_SOCKET`) becomes the host: it owns an `EventBus`, runs
an accept thread, and dispatches events via a background drain
thread. Subsequent processes connect as clients; their events are
serialized with pickle protocol 5 (with out-of-band `PickleBuffer` for
numpy zero-copy) and forwarded over the socket.

Payloads whose numpy `.nbytes` is at or above `GOGGLES_SHM_THRESHOLD`
(default 64 KiB) take a zero-copy shared-memory side-channel: the
client writes the buffer into a `multiprocessing.shared_memory.SharedMemory`
block and sends only metadata over the socket; the host maps the
block, passes the event to handlers, then closes and unlinks it.

Cross-machine logging is out of scope for the built-in transport. To
add it, implement the `Transport` protocol in a new module and have
`goggles._core.routing.get_bus` return it.

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
  - `GOGGLES_SOCKET` (default
    `${XDG_RUNTIME_DIR:-/tmp}/goggles-<uid>.sock`) -- Unix-socket path
    used by `LocalTransport` to elect a host.
  - `GOGGLES_SHM_THRESHOLD` (default `65536`, bytes) -- numpy payload
    size at or above which events take the shared-memory side-channel.
    Set to `0` to disable the shm path.
  - `GOGGLES_ASYNC` (default `1`) -- async emit mode.
  - `GOGGLES_SHUTDOWN_TIMEOUT` (default `5.0s`) -- shutdown timeout.

- **Tests mirror this layout** (`tests/core/`, `tests/core/integrations/`,
  `tests/history/`). Performance benchmarks live in
  [examples/105_benchmark.py](../../examples/105_benchmark.py).

## Where to make changes

| Change | Where |
|---|---|
| New handler / integration | `goggles/_core/integrations/<name>.py`, register in `goggles/_core/integrations/__init__.py`, re-export from `goggles/__init__.py`. |
| New filter | `goggles/filters.py` (public module). |
| New logger method | Update the `TextLogger` / `DataLogger` / `GogglesLogger` Protocols in `goggles/__init__.py`, then `_core/logger.py`. |
| New config helper | `goggles/config.py` (keep the YAML-focused surface small). |
| New history buffer op | `goggles/history/buffer.py` + types in `goggles/history/types.py`. |
| New / alternative transport | Implement the `Transport` protocol in a new module under `goggles/_core/` (or a peer to `_core/transport.py`) and wire it in `goggles/_core/routing.py:get_bus`. |
