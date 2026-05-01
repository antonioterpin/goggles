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
    |-- transport/           # Transport protocol + LocalTransport — see transport.md
    |   |-- _frames.py       # Wire format, env knobs, shm helpers
    |   |-- _endpoints.py    # Platform-specific socket binding/connecting
    |   |-- _protocol.py     # Transport Protocol declaration
    |   `-- _local.py        # LocalTransport (UDS / TCP loopback)
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

### 1. Logger & bus routing (`_core/logger.py`, `_core/routing.py`, `_core/transport/`)

Goggles maintains a **single EventBus per machine**: one host process
owns it, the rest connect as clients. The user-facing entry points are
`get_logger()` and `attach()` in `goggles/__init__.py`.

- `_core/logger.py` holds the user-visible `CoreTextLogger` /
  `CoreGogglesLogger` implementations; each call builds an `Event` and
  hands it to the transport via `emit` (async) or `emit_sync`.
- `_core/routing.py` exposes `get_bus()` / `reset_bus()`, the
  process-wide transport singleton.
- `_core/transport/` defines the `Transport` protocol and the default
  `LocalTransport` (see §4 and [transport.md](transport.md) for the
  package layout).

- Loggers carry a `scope` (free-form string, default `global`). Events
  are routed to handlers attached to matching scopes.
- Events include logs (text) and structured data (`scalar`, `image`,
  `video`, `vector_field`, `histogram`, ...).

#### Namespaced scopes (dot notation)

Scope matching is hierarchical: a handler attached to scope `S` also
receives events emitted on any scope of the form `S.X`, `S.X.Y`, etc.
The match is a strict prefix on dot-separated segments — `globalA` does
**not** match a handler on `global`, but `global.run1` does.

```python
gg.attach(handler, scopes=["training"])

gg.get_logger(scope="training").info("seen by handler")
gg.get_logger(scope="training.epoch_3").info("also seen by handler")
gg.get_logger(scope="eval").info("not seen")
```

This lets a single handler subscribe to a whole subtree (e.g. one
`training` handler that captures every per-episode logger) without
having to enumerate the children up front. Implementation:
[goggles/__init__.py `EventBus.emit`](../../goggles/__init__.py).

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

### 4. Transport (`_core/transport/`)

See [transport.md](transport.md) for the package layout (frames,
endpoints, protocol, local impl) and the flat re-export rationale.

Goggles uses a local-machine transport (`LocalTransport`) to route
events within a machine. The first process to bind the configured
endpoint becomes the host: it owns an `EventBus`, runs an accept
thread, and dispatches events via a background drain thread.
Subsequent processes connect as clients; their events are serialized
with pickle protocol 5 (with out-of-band `PickleBuffer` for numpy
zero-copy on the wire) and forwarded over the endpoint.

The endpoint is platform-dependent. On Unix (Linux, macOS) it is an
`AF_UNIX` stream socket at the path indicated by `GOGGLES_SOCKET`,
protected at `0o600` (owner-only). On Windows, where `AF_UNIX` is
unreliable across Python versions, the host binds a TCP loopback
socket on `127.0.0.1:<random-port>` and writes the chosen port plus a
per-host random token to a JSON sidecar discovery file at the same
logical `GOGGLES_SOCKET` path; clients send the token as the first
frame, and the host drops the connection unless it matches. Both
paths share the same framing protocol.

#### Trust model

The host calls `pickle.loads` on bytes received from connected peers,
so the transport assumes its peers are trusted local processes. The
threat actor in scope is **another local user on the same machine**
who could otherwise connect to the host and feed it crafted bytes;
remote network attackers are out of scope (the host never listens
off-loopback). Each backend mitigates this differently:

- **Unix (`AF_UNIX`)**: the socket file is created at `0o600` under a
  parent dir narrowed to `0o700`. Only the owning UID can `connect()`,
  so other local users are excluded by the kernel before any byte is
  read.
- **Windows (TCP loopback)**: any local user can `connect()` to
  `127.0.0.1:<port>`, so isolation is enforced on the wire instead of
  the filesystem. The host mints a 256-bit random token at bind time,
  writes it into the discovery file, and the accept loop requires a
  matching token frame within a 1-second handshake window before any
  payload bytes are read. The default discovery path is
  `tempfile.gettempdir()`, which on stock Windows resolves under
  `%LOCALAPPDATA%\Temp` (already user-private); the token defends the
  case where another user can locate the listener (port scan, or an
  explicitly-shared `GOGGLES_SOCKET`). Other local users still
  consume one accept slot per probe but never reach the unpickler.

Setting `GOGGLES_SOCKET` to a path another user can read does not
weaken the Unix story (file mode `0o600` still gates `connect`) but
does weaken the Windows story (the token sits in that file). On a
shared-tenant Windows host, leave `GOGGLES_SOCKET` at its default
under the per-user temp dir.

Payloads whose numpy `.nbytes` is at or above `GOGGLES_SHM_THRESHOLD`
(default 64 KiB) take a shared-memory side-channel: the client writes
the buffer into a `multiprocessing.shared_memory.SharedMemory` block
and sends only metadata over the socket; the host maps the same
block, copies it into a private `numpy.ndarray`, then unlinks the
block before dispatching the event to handlers. The wire side is
zero-copy; the handler-visible array is a private copy so the segment
can be released immediately.

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
    `${XDG_RUNTIME_DIR:-<tempdir>}/goggles-<user>.sock`) -- endpoint
    path used by `LocalTransport` to elect a host. On Unix this is
    the AF_UNIX socket; on Windows it is the sidecar discovery file
    that records the TCP loopback port (see §4).
  - `GOGGLES_SHM_THRESHOLD` (default `65536`, bytes) -- numpy payload
    size at or above which events take the shared-memory side-channel.
    Set to `0` to disable the shm path.
  - `GOGGLES_ASYNC` (default `1`) -- async emit mode.
  - `GOGGLES_SHUTDOWN_TIMEOUT` (default `5.0s`) -- shutdown timeout.

- **Tests mirror this layout** (`tests/core/`, `tests/core/integrations/`,
  `tests/history/`). Performance benchmarks live in
  [examples/105_benchmark.py](../../examples/105_benchmark.py) rather
  than under `tests/`: they take seconds to minutes per run, depend on
  the optional `hydra-core` dev extra, and produce a report rather
  than a pass/fail assertion, so they don't belong in the default
  `pytest` run.

## Where to make changes

| Change | Where |
|---|---|
| New handler / integration | `goggles/_core/integrations/<name>.py`, register in `goggles/_core/integrations/__init__.py`, re-export from `goggles/__init__.py`. |
| New filter | `goggles/filters.py` (public module). |
| New logger method | Update the `TextLogger` / `DataLogger` / `GogglesLogger` Protocols in `goggles/__init__.py`, then `_core/logger.py`. |
| New config helper | `goggles/config.py` (keep the YAML-focused surface small). |
| New history buffer op | `goggles/history/buffer.py` + types in `goggles/history/types.py`. |
| New / alternative transport | Implement the `Transport` protocol in a new module under `goggles/_core/` (peer to `_core/transport/`) and wire it in `goggles/_core/routing.py:get_bus`. See [transport.md](transport.md). |
