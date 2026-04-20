"""Process-wide transport singleton.

This module owns the process-global transport instance used by loggers and
:func:`goggles.attach` / :func:`goggles.detach`. The default transport is
:class:`goggles._core.transport.LocalTransport`, which routes events through
a Unix domain socket to the host process on the same machine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from goggles.types import Event  # re-export for compatibility

if TYPE_CHECKING:
    from goggles._core.transport import Transport


__singleton_transport: Transport | None = None


def get_bus() -> Transport:
    """Return the process-wide transport singleton.

    The first call in the process constructs a :class:`LocalTransport`
    bound to the socket path indicated by ``GOGGLES_SOCKET`` (or its
    default). Subsequent calls return the same instance; if the prior
    singleton was shut down, a fresh one is constructed.

    Returns:
        The singleton transport.
    """
    global __singleton_transport  # noqa: PLW0603
    current = __singleton_transport
    if current is None or not getattr(current, "_running", True):
        from goggles._core.transport import LocalTransport  # noqa: PLC0415

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


__all__ = ["Event", "get_bus", "reset_bus"]
