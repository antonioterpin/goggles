# Changelog

All notable changes to `robo-goggles` are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
(pre-1.0: minor bumps may carry breaking changes).

## [Unreleased]

### Fixed

- **Dedicated host: multi-process apps no longer fragment into duplicate runs.**
  The shared host was reaped (SIGTERM) by whichever process called `finish()`
  or exited first, even while other processes were still connected and logging.
  Each surviving client then respawned its own host, so one multi-process run
  cycled through many hosts -- each opening its own W&B runs, splitting a single
  scope's data across several runs (and leaving timing gaps during the churn).
  The host now **self-reaps only once its last client disconnects** (after
  `GOGGLES_HOST_IDLE_TIMEOUT`, default 5s -- a grace that bridges a transient
  reconnect, shortened on a clean `finish()` so a single-process run winds the
  host down promptly); `finish()`/atexit shut down only the local client,
  never a host other processes are using. One host = one set of handlers = one
  run per scope, for the whole app's lifetime. `finish()` still **waits for the
  host to finalize its handlers (drain + close, e.g. finishing a W&B run) when
  this process was the host's last client**, so the "everything is delivered and
  finalized once `finish()` returns" guarantee is preserved -- without blocking
  one process on its siblings. The host still inherits the spawning process's
  stdout/stderr (so `ConsoleHandler` output appears as before); set the new
  `GOGGLES_HOST_LOG` to capture it to a file instead.

## [0.2.3] - 2026-06-10

### Changed

- **W&B: nested-dict metric metadata is flattened to dotted scalar keys.** Any
  nested dict logged alongside a metric/media event (e.g.
  `logger.scalar("loss", v, step=i, custom_step={"time": t})`) now reaches
  `wandb.log` as flat keys (`custom_step.time`) instead of a nested *object*
  W&B can't select as a chart axis -- so a custom step / physical-time x-axis
  works. This applies to all such values, not only the axis case; charts or
  queries referencing the old nested key must switch to the dotted key. See
  `examples/09_custom_step_axis.py`.

## [0.2.2] - 2026-06-10

### Added

- **Dedicated host process for handlers (on by default).** goggles spawns a
  dedicated subprocess to be the transport host (owning the `EventBus` and
  running every handler, notably the blocking W&B uploader), so the
  application and every other process connect as clients -- keeping logging
  back-pressure from starving the app's latency-critical paths (RPC servers,
  control/sim loops). Set `GOGGLES_DEDICATED_HOST=0` (or `false`/`no`/`off`)
  to host in-process instead (the first process to bind the socket becomes the
  host). Handlers are unchanged (`attach(...)` already ships them to the host
  over the wire); `gg.finish()` drains and terminates the host (with an
  `atexit` backstop). See `examples/08_dedicated_host.py` and
  [docs/guides/transport.md](docs/guides/transport.md).
- **`GOGGLES_HOST_IMPORTS`** -- comma/space-separated modules the dedicated
  host imports at startup, so custom handlers registered via
  `register_handler` can be reconstructed there.
- **`is_host`** is now part of the `Transport` protocol, so callers can tell
  whether they own the bus or connect to a (possibly dedicated) host.

### Fixed

- **Transport host shutdown no longer drops buffered events.** On shutdown
  the host now lets its reader threads drain frames already buffered on
  client sockets (consuming up to EOF for cleanly-disconnected clients)
  before force-waking any still-blocked reader, instead of discarding that
  tail via `SHUT_RDWR`. This is the host-side analogue of the client-side
  `BYE` flush fix and makes `finish()` deliver every event emitted before it
  within the shutdown timeout -- including across the dedicated host process.
- **Shared-memory transport no longer floods `resource_tracker` warnings.**
  The process that creates a shared-memory block for a large payload now
  detaches it from its multiprocessing resource tracker (the consumer unlinks
  the block; a host-startup sweep handles crash leftovers), so logging large
  images/videos no longer prints a "leaked shared_memory" / "No such file"
  warning per payload at exit. Most visible now that handlers run in a
  dedicated host by default, since single-process apps then take the shm path.

## [0.2.1] - 2026-05-24

### Added

- **W&B: `wandb_init_kwargs` escape hatch.** The handler accepts an
  optional `wandb_init_kwargs` mapping forwarded verbatim to
  `wandb.init`, with validation that rejects invalid or handler-owned
  keys. (#210)
- **W&B artifact: directory uploads + aliases.** The artifact payload
  `path` may now point to a directory (uploaded recursively via
  `Artifact.add_dir`), and an optional `aliases` field is forwarded to
  `run.log_artifact` so callers can tag versions (e.g. `["best"]`,
  `["latest"]`). Enables clean Orbax/PyTorch checkpoint uploads
  without tar-balling.

### Fixed

- **W&B artifact**: skip with a warning when `path` does not exist
  instead of crashing the dispatch loop on `Artifact.add_file`.

## [0.2.0] - 2026-05-01

First release after a substantial rework of the transport layer, the
configuration surface, and the public API. **Pre-1.0 versioning rules
apply: this is a breaking release.** Pin to `>=0.1.9,<0.2` if you need
the old behavior.

### Breaking

- **Transport: replace portal RPC with Unix-socket `LocalTransport`.**
  The cross-process backbone is now an in-tree Unix-socket transport
  with explicit framing; the `portal` runtime dependency is gone.
  (#143)
- **Transport: split `transport.py` into a focused package.** Internal
  imports moved under `goggles._core.transport.*`. Public surface
  imported via `goggles` is unchanged; direct imports of internal
  paths must be updated. (#163)
- **Config: drop CLI override helpers.** The `from_cli`-style override
  helpers were removed. Use Hydra/argparse upstream and pass the
  resolved config to `from_config`. (#150)
- **Config: reject private-field overrides in `from_config`.** Keys
  with a leading underscore now raise instead of being silently
  applied. (#155)

### Added

- **`gg.configure()`** shortcut for the common console-only setup. (#166)
- **Per-logger `set_level`** + `level=` kwarg on `get_logger`. (#169)
- **`logger.trajectories`** for particle trajectory logging. (#148)
- **JIT-compatible filter API** (`init_state` + `apply`). (#176)
- **Outliers-rejection filters** + filter registry pluggability. (#119, #105)
- **W&B handler**: vector-field support, `tags=` forwarded to
  `wandb.init`, per-scope monotonic-step guard. (#125, #174, #172)
- **Image promotion**: image-shaped values passed to `push()` are
  routed to the image handlers. (#156)
- **Out-of-order step policy** unified across `LocalStorage` and
  console handlers. (#165)
- **Dict logging.** (#109)
- **`PrettyConfig`** as a drop-in basic config class. (#100)
- **`py.typed` marker** + `Typing :: Typed` classifier so downstream
  type checkers consume goggles' types. (#190)

### Changed

- `basedpyright` and `pydoclint` are no longer runtime dependencies;
  install with `pip install robo-goggles[dev]` to get them. (#189)
- License metadata modernised to PEP 639 SPDX form. (#191)
- CI workflows trigger automatically on push and pull-request; the
  `code-style` job now installs `uv` so the `pydoclint` hook runs
  in CI. (#185)
- Dropped the `pyyaml` runtime dependency in favor of
  `ruamel.yaml`. (#149)

### Fixed

- **Transport (Windows)**: require token handshake on TCP loopback so
  unrelated local clients cannot attach. (#175)
- **Shutdown**: default to unbounded wait and confirm per-handler
  close, fixing premature exits. (#180)
- **W&B**: batch same-step commits to fix unreliable display; do not
  mutate shared `event.extra`; accept channels-last video tensors.
  (#177, #147, #146)
- **Logger**: resolve transport lazily so class-level loggers work
  before the bus is attached. (#164)
- **Filters**: initialize ring-buffer index in `__init__`. (#144)
- **Typing**: covariant `Metrics` type. (#135)

[Unreleased]: https://github.com/antonioterpin/goggles/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/antonioterpin/goggles/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/antonioterpin/goggles/compare/v0.1.9...v0.2.0
