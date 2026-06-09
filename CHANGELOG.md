# Changelog

All notable changes to `robo-goggles` are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
(pre-1.0: minor bumps may carry breaking changes).

## [Unreleased]

### Added

- **Dedicated host process (`GOGGLES_DEDICATED_HOST`).** Setting
  `GOGGLES_DEDICATED_HOST=1` makes goggles spawn a dedicated subprocess to be
  the transport host, so the `EventBus` and every handler (notably the
  blocking W&B uploader) run there instead of on the application's
  interpreter -- keeping logging back-pressure from starving the app's
  latency-critical paths (RPC servers, control/sim loops). The application
  and every other process connect as clients; handlers are unchanged
  (`attach(...)` already ships them to the host over the wire). `gg.finish()`
  drains and terminates the host (with an `atexit` backstop), and
  `GOGGLES_HOST_IMPORTS` lets the host import modules that define custom
  handlers. See `examples/08_dedicated_host.py` and
  [docs/guides/transport.md](docs/guides/transport.md).

### Fixed

- **Transport host shutdown no longer drops buffered events.** On shutdown
  the host now lets its reader threads drain frames already buffered on
  client sockets (consuming up to EOF for cleanly-disconnected clients)
  before force-waking any still-blocked reader, instead of discarding that
  tail via `SHUT_RDWR`. This is the host-side analogue of the client-side
  `BYE` flush fix and makes `finish()` deliver every event emitted before it
  within the shutdown timeout -- including across the dedicated host process.

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
