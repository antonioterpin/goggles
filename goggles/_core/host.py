"""Dedicated host process for the goggles transport.

Run as ``python -m goggles._core.host``. goggles spawns this automatically by
default (see :mod:`goggles._core.routing`); set ``GOGGLES_DEDICATED_HOST=0`` to
host in-process instead.

The host binds the ``GOGGLES_SOCKET`` endpoint and becomes the transport HOST:
it owns the :class:`~goggles.EventBus` and runs every attached handler --
including blocking ones such as the W&B uploader. Running the host (and
therefore the handlers) in a *separate* process keeps that work off the
application's interpreter, so the application's latency-critical paths (RPC
servers, control loops) are never starved by logging back-pressure. Handlers
cross to the host over the wire as serialized specs (``Handler.to_dict`` /
``from_dict``), so ``goggles.attach(WandBHandler(...))`` in the application is
constructed and executed *here*.

On ``SIGTERM`` / ``SIGINT`` the host shuts the transport down gracefully -- it
drains any queued events into the handlers and closes them (which is where
``wandb`` runs are finished and flushed) -- then exits.
"""

from __future__ import annotations

import importlib
import os
import signal
import sys
import threading

from goggles._core.transport import LocalTransport


def run() -> int:  # pragma: no cover - entrypoint, run only in a subprocess
    """Run the dedicated host until it is signalled to stop.

    Binds the endpoint, signals readiness (so a spawning parent can safely
    proceed and connect as a client), then blocks until ``SIGTERM`` /
    ``SIGINT`` arrives and shuts the transport down gracefully.

    Returns:
        ``0`` on a clean run, or ``1`` if the endpoint is already owned by
        another host (there is nothing for this process to host).
    """
    # A dedicated host binds the socket itself; make sure it never recurses
    # into spawning *another* host from inside the host process.
    os.environ.pop("GOGGLES_DEDICATED_HOST", None)

    # Built-in handlers resolve by name from goggles' globals, but CUSTOM
    # handlers (registered via ``goggles.register_handler``) live in the
    # application's modules, which this fresh process has not imported. Import
    # the modules named in ``GOGGLES_HOST_IMPORTS`` (comma/space separated) so
    # their handlers can be reconstructed here.
    _import_host_modules()

    transport = LocalTransport()
    if not transport.is_host:
        print(
            "goggles host: endpoint already owned by another host; exiting.",
            file=sys.stderr,
        )
        return 1

    stop = threading.Event()

    def _request_stop(*_: object) -> None:
        stop.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _request_stop)
        except (ValueError, OSError):  # pragma: no cover - platform-specific
            # Not on the main thread, or signal unavailable on this platform.
            pass

    # Signal readiness only after the accept loop is up, so a parent that
    # waits on this file then reliably connects as a client (not a host).
    ready_path = os.environ.get("GOGGLES_HOST_READY")
    if ready_path:
        try:
            with open(ready_path, "w", encoding="utf-8") as handle:
                handle.write(str(os.getpid()))
        except OSError:  # pragma: no cover - best effort
            pass

    # Self-reap: wind down once the last client disconnects (and none
    # reconnects within the idle grace), so the host's lifetime follows
    # "any client connected" rather than whichever process spawned it.
    transport.set_idle_callback(stop.set)

    stop.wait()

    transport.shutdown(timeout=_shutdown_timeout())
    return 0


def _import_host_modules() -> None:
    """Import modules named in ``GOGGLES_HOST_IMPORTS`` (comma/space sep).

    Lets the host reconstruct custom handlers whose classes (and
    ``register_handler`` calls) live in application modules the host would
    not otherwise import. Failures are logged and skipped.
    """
    spec = os.environ.get("GOGGLES_HOST_IMPORTS", "")
    for name in spec.replace(",", " ").split():
        try:
            importlib.import_module(name)
        except Exception as exc:  # report and continue
            print(
                f"goggles host: failed to import {name!r}: {exc}",
                file=sys.stderr,
            )


def _shutdown_timeout() -> float | None:
    """Resolve the graceful-shutdown budget from ``GOGGLES_SHUTDOWN_TIMEOUT``.

    Returns:
        The timeout in seconds, or ``None`` to wait indefinitely (which is
        what a value of ``0`` or an unparsable value maps to, matching
        :func:`goggles.finish`).
    """
    try:
        timeout = float(os.getenv("GOGGLES_SHUTDOWN_TIMEOUT", "0"))
    except ValueError:
        return None
    return timeout if timeout > 0 else None


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    raise SystemExit(run())
