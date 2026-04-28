"""Cross-platform endpoint abstraction for the local transport.

``_Endpoint`` is the protocol every platform implementation has to
satisfy: ``connect`` / ``bind`` / ``cleanup`` / ``accept_address_hint``.
``_UnixEndpoint`` uses ``AF_UNIX`` with the socket file pinned at
``0o600``; ``_WindowsEndpoint`` binds a TCP loopback port and writes a
sidecar discovery file. ``_endpoint()`` returns the right class for the
current platform.
"""

from __future__ import annotations

import logging
import os
import socket
from typing import Protocol

from ._frames import _IS_WINDOWS

_log = logging.getLogger(__name__)


class _Endpoint(Protocol):
    """Platform abstraction for binding / connecting to a logical path."""

    @staticmethod
    def connect(path: str, timeout: float = 1.0) -> socket.socket | None:
        """Attempt to connect to a host at ``path``.

        Args:
            path: Logical endpoint identifier.
            timeout: Connect timeout in seconds.

        Returns:
            Connected stream socket, or None if no host is listening.
        """
        ...

    @staticmethod
    def bind(path: str) -> socket.socket:
        """Bind and listen at ``path``.

        Args:
            path: Logical endpoint identifier.

        Returns:
            The bound server socket.
        """
        ...

    @staticmethod
    def cleanup(path: str) -> None:
        """Remove filesystem artifacts associated with ``path``.

        Best-effort; callers should not rely on failure modes.

        Args:
            path: Logical endpoint identifier.
        """
        ...

    @staticmethod
    def accept_address_hint() -> str:
        """Return a human-readable label for logs.

        Returns:
            A short string identifying the endpoint family.
        """
        ...


class _UnixEndpoint:
    """AF_UNIX-based endpoint for Linux and macOS."""

    @staticmethod
    def connect(path: str, timeout: float = 1.0) -> socket.socket | None:
        if not os.path.exists(path):
            return None
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            sock.connect(path)
        except OSError:
            sock.close()
            return None
        sock.settimeout(None)
        return sock

    @staticmethod
    def bind(path: str) -> socket.socket:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
            try:
                os.chmod(parent, 0o700)
            except OSError:
                # Parent may be a shared dir (e.g. /tmp); don't fail here.
                pass

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        old_umask = os.umask(0o077)
        try:
            server.bind(path)
        finally:
            os.umask(old_umask)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        server.listen(64)
        return server

    @staticmethod
    def cleanup(path: str) -> None:
        try:
            os.unlink(path)
        except (FileNotFoundError, IsADirectoryError):
            pass
        except OSError:
            _log.exception("Failed to unlink socket file at %s", path)

    @staticmethod
    def accept_address_hint() -> str:
        return "AF_UNIX"


class _WindowsEndpoint:
    """TCP loopback endpoint for Windows.

    The "socket path" becomes a sidecar file recording the port the host
    chose; clients read it to find the host.
    """

    _LOOPBACK = "127.0.0.1"

    @staticmethod
    def _read_port(path: str) -> int | None:
        try:
            with open(path, encoding="utf-8") as f:
                data = f.read().strip()
        except OSError:
            return None
        try:
            return int(data)
        except ValueError:
            return None

    @staticmethod
    def connect(path: str, timeout: float = 1.0) -> socket.socket | None:
        port = _WindowsEndpoint._read_port(path)
        if port is None:
            return None
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            sock.connect((_WindowsEndpoint._LOOPBACK, port))
        except OSError:
            sock.close()
            return None
        sock.settimeout(None)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return sock

    @staticmethod
    def bind(path: str) -> socket.socket:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((_WindowsEndpoint._LOOPBACK, 0))
        server.listen(64)
        port = server.getsockname()[1]
        # Write port atomically: write to tmp, then rename.
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(str(port))
        os.replace(tmp, path)
        return server

    @staticmethod
    def cleanup(path: str) -> None:
        try:
            os.unlink(path)
        except (FileNotFoundError, IsADirectoryError):
            pass
        except OSError:
            _log.exception("Failed to remove discovery file at %s", path)

    @staticmethod
    def accept_address_hint() -> str:
        return "TCP loopback"


def _endpoint() -> type[_Endpoint]:
    if _IS_WINDOWS:
        return _WindowsEndpoint
    return _UnixEndpoint
