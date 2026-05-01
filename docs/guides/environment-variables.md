# Environment variables

Goggles reads a small number of environment variables to tune transport
behaviour, observability, and shutdown semantics. They all default to
sensible values and are read once at process start (or first transport
construction); changing them at runtime has no effect.

| Variable | Default | Effect |
|---|---|---|
| `GOGGLES_ASYNC` | `1` | When `1`/`true`/`yes`, logging calls return immediately and dispatch in the background. Set to `0`/`false`/`no` to make every call block until the bus has seen the event. |
| `GOGGLES_SOCKET` | platform default (Unix: `${XDG_RUNTIME_DIR:-/tmp}/goggles-${USER}.sock`; Windows: TCP loopback discovery file) | Path of the rendezvous socket the transport host binds and clients connect to. Set this to isolate independent test runs or share one bus across users. |
| `GOGGLES_SHM_THRESHOLD` | `262144` (256 KiB) | Payload byte threshold above which the transport switches from inline pickle frames to shared-memory side-channel transfer. Lower it to exercise the SHM path in tests; raise it to keep small payloads inline. Invalid values fall back to the default. |
| `GOGGLES_CAPTURE_CALLER` | `1` | When `1`/`true`/`yes`, the logger walks the call stack on each emit (~5–15 µs per call) to record `filepath`/`lineno`. Set to `0`/`false`/`no` on hot loops above ~10 kHz; events will then carry `("<unknown>", 0)`, which only the console formatter uses. |
| `GOGGLES_SHUTDOWN_TIMEOUT` | `5.0` | Seconds the `gg.finish()` shutdown path waits for the transport to drain pending events before forcibly tearing down. |

## Where each one lives

- `GOGGLES_ASYNC` is read in [goggles/__init__.py](../../goggles/__init__.py) and exported as the public `GOGGLES_ASYNC` constant.
- `GOGGLES_SHUTDOWN_TIMEOUT` is read by `gg.finish()` in the same module.
- `GOGGLES_SOCKET` and `GOGGLES_SHM_THRESHOLD` are read by `LocalTransport` in [goggles/_core/transport.py](../../goggles/_core/transport.py).
- `GOGGLES_CAPTURE_CALLER` is read by the core logger in [goggles/_core/logger.py](../../goggles/_core/logger.py).

If you add a new knob, add a row here and link the read site so future
readers can find both the contract and the implementation in one hop.
