"""Process-wide transport singleton.

This module owns the process-global transport instance used by loggers and
:func:`goggles.attach` / :func:`goggles.detach`. The default transport is
:class:`goggles._core.transport.LocalTransport`, which routes events through
a Unix domain socket to the host process on the same machine.

Dedicated host process
----------------------
By default goggles spawns a dedicated :mod:`goggles._core.host` subprocess to
be the *host* (the process that owns the :class:`~goggles.EventBus` and runs
the handlers); the application -- and every other process -- then connects as
a *client*, and the spawning process terminates the host on
:func:`goggles.finish` (with an ``atexit`` backstop). This keeps heavy handlers
(e.g. the W&B uploader) off the application's interpreter so they cannot starve
its latency-critical paths. Handlers are unaffected: they already cross to the
host as serialized specs, so ``attach(WandBHandler(...))`` runs inside the
subprocess.

Set ``GOGGLES_DEDICATED_HOST=0`` (or ``false``/``no``/``off``) to opt out and
host in-process instead -- the first process to bind the socket becomes the
host, as before.
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
from typing import IO, TYPE_CHECKING

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
# How long to wait for a freshly spawned host to bind + signal readiness
# before falling back to an in-process host. Binding is normally well under a
# second; this only bounds the pathological "child started but hung" case,
# since the first log blocks on it.
_HOST_SPAWN_TIMEOUT_S = 10.0
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

    By default a dedicated host subprocess is spawned (once per process, and
    only if no host is already listening) before the transport is built, so
    this process comes up as a client. The first call therefore blocks
    briefly while the host binds. Set ``GOGGLES_DEDICATED_HOST=0`` to skip
    this and host in-process. A spawn failure is non-fatal: it logs and
    falls back to an in-process host.

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
    """Whether goggles should run the host in a dedicated subprocess.

    Dedicated hosting is the default; set ``GOGGLES_DEDICATED_HOST`` to a
    falsy value (``0``/``false``/``no``/``off``) to host in-process instead.

    Returns:
        ``True`` unless the env var is explicitly set to a falsy value.
    """
    return os.getenv(_DEDICATED_HOST_ENV, "1").lower() not in (
        "0",
        "false",
        "no",
        "off",
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

    Spawns a dedicated host by default; a no-op only when
    ``GOGGLES_DEDICATED_HOST`` is falsy (``0``/``false``/``no``/``off``).
    Spawns at most one host per process; if another host already listens on
    the socket this process simply becomes a client. Any failure -- the
    subprocess cannot be started, or it never signals readiness -- is
    non-fatal: it logs and returns, leaving the caller to fall back to an
    in-process host.
    """
    global __host_proc, __atexit_registered  # noqa: PLW0603
    if not _dedicated_host_enabled():
        return
    with __host_lock:
        if __host_proc is not None and __host_proc.poll() is None:
            return  # we already spawned a live host
        try:
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
            # The host outlives the process that spawns it, so keep its output
            # off that process's (eventually closed) terminal: discard it, or
            # capture it via GOGGLES_HOST_LOG when debugging.
            host_stdio = _open_host_log()
            try:
                proc = subprocess.Popen(
                    [sys.executable, "-m", "goggles._core.host"],
                    env=env,
                    stdout=host_stdio,
                    stderr=subprocess.STDOUT,
                )
            finally:
                if not isinstance(host_stdio, int):
                    host_stdio.close()  # the child kept its own dup
        except Exception:
            # Spawning can fail in restricted environments (no fork/exec,
            # resource limits, ...). Never let that crash the caller's first
            # log -- fall back to an in-process host.
            _log.warning(
                "goggles: could not spawn a dedicated host; "
                "falling back to an in-process host.",
                exc_info=True,
            )
            return

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


def _open_host_log() -> IO[str] | int:
    """stdout/stderr target for the spawned host (it outlives its spawner).

    Returns an opened ``GOGGLES_HOST_LOG`` file when that env var is set -- so
    the host's output is captured rather than written to the spawning
    process's terminal after it exits -- otherwise ``subprocess.DEVNULL``.

    Returns:
        An appendable text file, or ``subprocess.DEVNULL``.
    """
    log_path = os.getenv("GOGGLES_HOST_LOG")
    if log_path:
        try:
            return open(log_path, "a", encoding="utf-8")
        except OSError:
            _log.warning(
                "goggles: could not open GOGGLES_HOST_LOG=%r; "
                "discarding host output.",
                log_path,
            )
    return subprocess.DEVNULL


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
    """``atexit`` backstop: flush this process's transport on interpreter exit.

    Mirrors :func:`goggles.finish` so a program that exits without calling it
    still ships its queued events and, on a client, disconnects cleanly from
    the dedicated host (so the host's client count is accurate). The shared
    host is NOT reaped here -- it self-reaps once its last client disconnects,
    so a sibling process still logging keeps its host. No-op once ``finish()``
    has run. Bounded so interpreter shutdown never hangs.
    """
    transport = __singleton_transport
    if transport is not None and transport.is_running:
        try:
            transport.shutdown(timeout=_HOST_ATEXIT_TIMEOUT_S)
        except Exception:  # best effort
            pass


def _dedicated_host_process() -> subprocess.Popen | None:
    """The dedicated host subprocess this process spawned, if any.

    Returns:
        The host :class:`subprocess.Popen`, or ``None`` if this process did
        not spawn a dedicated host (not in dedicated mode, or a client of an
        externally managed host).
    """
    return __host_proc


__all__ = ["Event", "get_bus", "reset_bus"]
