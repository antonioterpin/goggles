# Transport package

`goggles/_core/transport/` provides cross-platform same-machine event
routing. The package is internal — public callers should use
`goggles.attach`, `goggles.get_logger`, and the `Transport` protocol
re-exported from `goggles._core.transport`.

## Public surface

- `Transport` — the [protocol](../../goggles/_core/transport/_protocol.py)
  every implementation satisfies. Decorated `@runtime_checkable` so
  user code can do `isinstance(x, Transport)`.
- `LocalTransport` — the default local-machine implementation
  (auto-elected host, `AF_UNIX` on Unix / TCP loopback on Windows).
  Lives in [_local.py](../../goggles/_core/transport/_local.py).

Both are importable from `goggles._core.transport` directly:

```python
from goggles._core.transport import Transport, LocalTransport
```

## Internal layout

| Module | Responsibility |
|---|---|
| [`_frames.py`](../../goggles/_core/transport/_frames.py) | Wire format (header struct, message kinds, pack/unpack), environment knobs (`GOGGLES_SOCKET`, `GOGGLES_SHM_THRESHOLD`), and shared-memory housekeeping (naming, orphan reaping). |
| [`_endpoints.py`](../../goggles/_core/transport/_endpoints.py) | Platform-specific socket binding/connecting. `_UnixEndpoint` for `AF_UNIX` (Linux, macOS); `_WindowsEndpoint` for TCP loopback with a sidecar discovery file. `_endpoint(...)` selects the right one. |
| [`_protocol.py`](../../goggles/_core/transport/_protocol.py) | The `Transport` protocol declaration — `is_running`, `emit`, `emit_sync`, `attach`, `detach`, `shutdown`. |
| [`_local.py`](../../goggles/_core/transport/_local.py) | `LocalTransport` itself, plus the private `_SendItem` / `_SENTINEL` used by its drain thread. |
| [`__init__.py`](../../goggles/_core/transport/__init__.py) | Re-exports the flat `from goggles._core.transport import X` surface that tests and routing rely on. |

## Dedicated host process

By default the host is whichever process first binds `GOGGLES_SOCKET`, so
the `EventBus` and every handler run in-process — usually the application,
where a blocking handler (e.g. the W&B uploader) can starve latency-critical
work. Setting `GOGGLES_DEDICATED_HOST=1` moves the host into a dedicated
subprocess:

- [`goggles/_core/routing.py`](../../goggles/_core/routing.py) — `get_bus()`
  spawns the host subprocess (once per process, and only if no host already
  listens) before constructing the transport, so the caller comes up as a
  **client**. `gg.finish()` calls `_terminate_dedicated_host()` to drain and
  reap it (an `atexit` backstop prevents orphans).
- [`goggles/_core/host.py`](../../goggles/_core/host.py) — the entry point
  (`python -m goggles._core.host`) the subprocess runs: it binds the socket
  (becoming host), signals readiness, then idles until `SIGTERM`/`SIGINT`,
  on which it shuts the transport down gracefully (drain queued events, close
  handlers — finishing W&B runs). `GOGGLES_HOST_IMPORTS` lets it import
  modules defining custom handlers.

No new wire format is needed: handlers already cross to the host as
serialized specs in an `ATTACH` frame (`Handler.to_dict` / `from_dict`), so
`attach(WandBHandler(...))` in the application is reconstructed and run in the
subprocess. `finish()` flushes this client, then drains the host (its readers
consume any frames still buffered in the socket before teardown) and closes
the handlers before reaping the subprocess — so events emitted before
`finish()` are delivered within the shutdown timeout, the same guarantee as an
in-process host.

## Why the flat re-exports

Tests and `goggles._core.routing` import private symbols from
`goggles._core.transport` directly (e.g. `_pack_small_frame`,
`_HEADER_FMT`, `_default_socket_path`). Those imports kept working
across the package split because `__init__.py` re-exports the entire
prior surface verbatim, with `# noqa: F401` to mark each entry as an
intentional re-export.

If you add a new symbol that callers should access via
`goggles._core.transport.<name>`, add it to the matching block in
`__init__.py` (and to `__all__` if it is public).

## See also

- [architecture.md](architecture.md) §4 for how transport fits into the
  bus/routing/handler pipeline.
- [environment-variables.md](environment-variables.md) for the runtime
  knobs.
