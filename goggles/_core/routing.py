"""Process-wide transport singleton.

This module owns the process-global transport instance used by loggers and
:func:`goggles.attach` / :func:`goggles.detach`. The default transport is
:class:`goggles._core.transport.LocalTransport`, which routes events through
a Unix domain socket to the host process on the same machine.

Dedicated host process
----------------------
By default the *host* (the process that owns the :class:`~goggles.EventBus`
and runs the handlers) is simply the first process to bind the socket -- which
is usually the application itself, so heavy handlers (e.g. the W&B uploader)
run on the application's interpreter and can starve its latency-critical
paths. Setting ``GOGGLES_DEDICATED_HOST=1`` makes goggles spawn a dedicated
:mod:`goggles._core.host` subprocess to be the host instead; the application
(and every other process) then connects as a *client*, and the spawning
process terminates the host on :func:`goggles.finish` (with an ``atexit``
backstop). Handlers are unaffected: they already cross to the host as
serialized specs, so ``attach(WandBHandler(...))`` runs inside the subprocess.
"""

from __future__ import annotations

import atexit
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from typing import TYPE_CHECKING

from goggles.types import Event  # re-export for compatibility

if TYPE_CHECKING:
    from goggles._core.transport import Transport


__singleton_transport: Transport | None = None

# Dedicated-host bookkeeping. ``__host_proc`` is the host subprocess this
# process spawned (``None`` if we did not spawn one); only the spawning
# process ever terminates it. Guarded by ``__host_lock`` so concurrent first
# emits don't spawn duplicates.
__host_proc: subprocess.Popen | None = None
__host_lock = threading.Lock()
__atexit_registered = False

_DEDICATED_HOST_ENV = "GOGGLES_DEDICATED_HOST"
# How long to wait for a freshly spawned host to bind + signal readiness.
_HOST_SPAWN_TIMEOUT_S = 30.0
# Bounded wait used by the atexit backstop so interpreter shutdown never hangs
# on a wedged host (an explicit ``finish()`` may pass a different budget).
_HOST_ATEXIT_TIMEOUT_S = 30.0

_log = logging.getLogger(__name__)


def get_bus() -> Transport:
    """Return the process-wide transport singleton.

    The first call in the process constructs a :class:`LocalTransport`
    bound to the socket path indicated by ``GOGGLES_SOCKET`` (or its
    default). Subsequent calls return the same instance; if the prior
    singleton was shut down, a fresh one is constructed.

    When ``GOGGLES_DEDICATED_HOST`` is set, a dedicated host subprocess is
    spawned (once per process, and only if no host is already listening)
    before the transport is built, so this process comes up as a client.

    Returns:
        The singleton transport.
    """
    global __singleton_transport  # noqa: PLW0603
    current = __singleton_transport
    if current is None or not current.is_running:
        from goggles._core.transport import LocalTransport  # noqa: PLC0415

        _spawn_dedicated_host()
        current = LocalTransport()
        __singleton_transport = current
    return current


def reset_bus() -> None:
    """Drop the process-wide transport singleton.

    Intended for tests and long-running processes that need to rebuild
    the transport after :meth:`Transport.shutdown` has been called.
    """
    global __singleton_transport  # noqa: PLW0603
    __singleton_transport = None


# ----- dedicated host process ---------------------------------------------


def _dedicated_host_enabled() -> bool:
    """Whether ``GOGGLES_DEDICATED_HOST`` requests a dedicated host process.

    Returns:
        ``True`` when the env var is a truthy value.
    """
    return os.getenv(_DEDICATED_HOST_ENV, "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _resolve_socket_path() -> str:
    """Resolve the endpoint path the transport will use.

    Returns:
        The ``GOGGLES_SOCKET`` path, or its per-user default.
    """
    from goggles._core.transport._frames import (  # noqa: PLC0415
        _default_socket_path,
    )

    return _default_socket_path()


def _host_is_listening(socket_path: str) -> bool:
    """Whether a goggles host already accepts connections at ``socket_path``.

    Args:
        socket_path: Endpoint path to probe.

    Returns:
        ``True`` if a host accepted a probe connection, else ``False``.
    """
    from goggles._core.transport._endpoints import _endpoint  # noqa: PLC0415

    try:
        sock = _endpoint().connect(socket_path, timeout=0.2)
    except Exception:  # any failure means "not listening"
        return False
    if sock is None:
        return False
    try:
        sock.close()
    except OSError:
        pass
    return True


def _spawn_dedicated_host() -> None:
    """Ensure a dedicated host subprocess owns the endpoint (idempotent).

    No-op unless ``GOGGLES_DEDICATED_HOST`` is set. Spawns at most one host
    per process; if another host already listens on the socket this process
    simply becomes a client. On failure it logs and returns, leaving the
    caller to fall back to an in-process host.
    """
    global __host_proc, __atexit_registered  # noqa: PLW0603
    if not _dedicated_host_enabled():
        return
    with __host_lock:
        if __host_proc is not None and __host_proc.poll() is None:
            return  # we already spawned a live host
        socket_path = _resolve_socket_path()
        if _host_is_listening(socket_path):
            return  # already hosted (by us or elsewhere); be a client

        ready_path = os.path.join(
            tempfile.gettempdir(),
            f"goggles-host-ready-{uuid.uuid4().hex}",
        )
        env = os.environ.copy()
        env["GOGGLES_SOCKET"] = socket_path
        env["GOGGLES_HOST_READY"] = ready_path
        # The child binds the socket directly; never let it recurse into
        # spawning another host.
        env.pop(_DEDICATED_HOST_ENV, None)

        proc = subprocess.Popen(
            [sys.executable, "-m", "goggles._core.host"],
            env=env,
        )

        if not _wait_for_ready(proc, ready_path):
            _log.warning(
                "goggles dedicated host did not become ready within %.0fs; "
                "falling back to an in-process host.",
                _HOST_SPAWN_TIMEOUT_S,
            )
            _kill(proc)
            return

        __host_proc = proc
        if not __atexit_registered:
            atexit.register(_atexit_terminate_host)
            __atexit_registered = True


def _wait_for_ready(proc: subprocess.Popen, ready_path: str) -> bool:
    """Poll until the host writes its ready file (or dies / times out).

    Args:
        proc: The spawned host process.
        ready_path: Path the host writes once it is accepting connections.

    Returns:
        ``True`` once the host signalled readiness, else ``False``.
    """
    deadline = time.monotonic() + _HOST_SPAWN_TIMEOUT_S
    ready = False
    while time.monotonic() < deadline:
        if os.path.exists(ready_path):
            ready = True
            break
        if proc.poll() is not None:
            break  # child exited before becoming ready
        time.sleep(0.01)
    try:
        os.unlink(ready_path)
    except OSError:
        pass
    return ready


def _kill(proc: subprocess.Popen) -> None:
    """Terminate then (if needed) force-kill a process, ignoring errors.

    Args:
        proc: The process to reap.
    """
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
            proc.wait(timeout=5.0)
        except Exception:
            pass
    except Exception:  # best effort
        pass


def _terminate_dedicated_host(timeout: float | None = None) -> None:
    """Stop and reap the dedicated host subprocess spawned by this process.

    Sends ``SIGTERM`` so the host drains queued events and finishes its
    handlers (e.g. W&B runs), waits up to ``timeout`` (``None`` waits
    indefinitely), then force-kills if necessary. Safe to call repeatedly and
    a no-op in processes that did not spawn a host.

    Args:
        timeout: Seconds to wait for graceful shutdown, or ``None`` to wait
            indefinitely.
    """
    global __host_proc  # noqa: PLW0603
    with __host_lock:
        proc = __host_proc
        __host_proc = None
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()  # SIGTERM on POSIX -> graceful host shutdown
    except Exception:  # best effort
        pass
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
            proc.wait(timeout=5.0)
        except Exception:
            pass


def _atexit_terminate_host() -> None:
    """``atexit`` backstop: reap the host with a bounded wait (no hang)."""
    _terminate_dedicated_host(timeout=_HOST_ATEXIT_TIMEOUT_S)


def _dedicated_host_process() -> subprocess.Popen | None:
    """The dedicated host subprocess this process spawned, if any.

    Returns:
        The host :class:`subprocess.Popen`, or ``None`` if this process did
        not spawn a dedicated host (not in dedicated mode, or a client of an
        externally managed host).
    """
    return __host_proc


__all__ = ["Event", "get_bus", "reset_bus"]
