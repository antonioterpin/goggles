"""Cross-platform endpoint abstraction for the local transport.

``_Endpoint`` is the protocol every platform implementation has to
satisfy: ``connect`` / ``bind`` / ``authorize`` / ``cleanup`` /
``accept_address_hint``. ``_UnixEndpoint`` uses ``AF_UNIX`` with the
socket file pinned at ``0o600``; ``_WindowsEndpoint`` binds a TCP
loopback port and writes a sidecar discovery file containing the port
and a per-host random token, then enforces a token handshake on every
accepted connection. ``_endpoint()`` returns the right class for the
current platform.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import secrets
import socket
import struct
from typing import Protocol

from ._frames import _IS_WINDOWS

_log = logging.getLogger(__name__)

# Length-prefixed token frame: 4-byte big-endian length + token bytes.
_TOKEN_LEN_FMT = "!I"
_TOKEN_LEN_SIZE = struct.calcsize(_TOKEN_LEN_FMT)
# 32 bytes of entropy = 256 bits, encoded as 64 hex chars on the wire.
_TOKEN_BYTES = 32
# Cap on accepted token length so a malicious client can't make the
# host allocate an arbitrary buffer before authentication.
_MAX_TOKEN_BYTES = 256
# Time the host will wait for a fresh client to send its token frame
# before dropping the connection. A real client sends it inline with
# connect(), so anything beyond this is a stalled or hostile peer.
_HANDSHAKE_TIMEOUT_S = 1.0


class _Endpoint(Protocol):
    """Platform abstraction for binding / connecting to a logical path."""

    @staticmethod
    def connect(path: str, timeout: float = 1.0) -> socket.socket | None:
        """Attempt to connect to a host at ``path``.

        Performs any platform-specific handshake (e.g. token exchange on
        Windows) before returning the socket, so callers can use the
        returned socket immediately.

        Args:
            path: Logical endpoint identifier.
            timeout: Connect timeout in seconds.

        Returns:
            Connected stream socket, or None if no host is listening
            or the handshake failed.
        """
        ...

    @staticmethod
    def bind(path: str) -> tuple[socket.socket, bytes | None]:
        """Bind and listen at ``path``.

        Args:
            path: Logical endpoint identifier.

        Returns:
            ``(server_socket, secret)``: the bound listening socket plus
            the per-host secret the accept loop must check incoming
            connections against. ``None`` for endpoints whose isolation
            is enforced outside the wire (e.g. AF_UNIX file permissions).
        """
        ...

    @staticmethod
    def authorize(conn: socket.socket, secret: bytes | None) -> bool:
        """Validate a freshly-accepted client connection.

        Args:
            conn: Newly-accepted client socket.
            secret: The secret returned by :meth:`bind` for this host,
                or ``None`` if the platform does not need a handshake.

        Returns:
            True if the client may proceed, False if the connection
            should be dropped.
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
    def bind(path: str) -> tuple[socket.socket, bytes | None]:
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
        return server, None

    @staticmethod
    def authorize(conn: socket.socket, secret: bytes | None) -> bool:
        del conn, secret
        return True

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

    Loopback TCP on a multi-user host is reachable from any local user,
    so the bind side mints a per-host secret, writes it into the sidecar
    discovery file, and the accept side requires every new connection
    to present the matching token before any payload bytes are read.

    The discovery file is JSON ``{"port": <int>, "token": <hex>}``,
    written atomically. On default Windows installations
    ``tempfile.gettempdir()`` resolves under the per-user ``Temp``
    directory, which is already user-private; the token defends
    against the case where another user can locate the listener (e.g.
    by port scanning or by an explicitly-shared ``GOGGLES_SOCKET``).
    """

    _LOOPBACK = "127.0.0.1"

    @staticmethod
    def _read_discovery(path: str) -> tuple[int, bytes] | None:
        try:
            with open(path, encoding="utf-8") as f:
                data = f.read()
        except OSError:
            return None
        try:
            obj = json.loads(data)
            port = int(obj["port"])
            token = bytes.fromhex(obj["token"])
        except (ValueError, KeyError, TypeError):
            return None
        if not 0 < port < 65536 or not 0 < len(token) <= _MAX_TOKEN_BYTES:
            return None
        return port, token

    @staticmethod
    def connect(path: str, timeout: float = 1.0) -> socket.socket | None:
        discovery = _WindowsEndpoint._read_discovery(path)
        if discovery is None:
            return None
        port, token = discovery
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            sock.connect((_WindowsEndpoint._LOOPBACK, port))
            frame = struct.pack(_TOKEN_LEN_FMT, len(token)) + token
            sock.sendall(frame)
        except OSError:
            sock.close()
            return None
        sock.settimeout(None)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return sock

    @staticmethod
    def bind(path: str) -> tuple[socket.socket, bytes | None]:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((_WindowsEndpoint._LOOPBACK, 0))
        server.listen(64)
        port = server.getsockname()[1]
        token = secrets.token_bytes(_TOKEN_BYTES)
        payload = json.dumps({"port": port, "token": token.hex()})
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp, path)
        return server, token

    @staticmethod
    def authorize(conn: socket.socket, secret: bytes | None) -> bool:
        if secret is None:
            # Internal misuse: WindowsEndpoint always mints a secret.
            return False
        prev_timeout = conn.gettimeout()
        conn.settimeout(_HANDSHAKE_TIMEOUT_S)
        try:
            header = _WindowsEndpoint._recv_exact(conn, _TOKEN_LEN_SIZE)
            if header is None:
                return False
            (length,) = struct.unpack(_TOKEN_LEN_FMT, header)
            if length <= 0 or length > _MAX_TOKEN_BYTES:
                return False
            token = _WindowsEndpoint._recv_exact(conn, length)
            if token is None:
                return False
        except OSError:
            return False
        finally:
            try:
                conn.settimeout(prev_timeout)
            except OSError:
                pass
        return hmac.compare_digest(token, secret)

    @staticmethod
    def _recv_exact(conn: socket.socket, n: int) -> bytes | None:
        buf = bytearray()
        while len(buf) < n:
            try:
                chunk = conn.recv(n - len(buf))
            except (TimeoutError, OSError):
                return None
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)

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
