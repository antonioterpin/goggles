"""Framing primitives, environment knobs, and shared-memory helpers.

Wire format constants (kind bytes, header layout), pack/unpack helpers
for the SMALL inline path and the LARGE shared-memory side-channel,
and the ``goggles_*`` shared-memory housekeeping.

Pure functions; no transport state.
"""

from __future__ import annotations

import logging
import os
import pickle
import secrets
import socket
import struct
import sys
import tempfile
import time
from multiprocessing import shared_memory
from typing import Any

import numpy as np

from goggles.types import Event

_log = logging.getLogger(__name__)


# --- Wire format ----------------------------------------------------------

# Every message on the wire is prefixed with a 1-byte kind and a 4-byte
# big-endian length. The payload format depends on kind.
_MSG_SMALL = 1  # inline pickle protocol 5 with out-of-band buffers
_MSG_LARGE = 2  # shared-memory side-channel; payload is pickled metadata
_MSG_ATTACH = 3
_MSG_DETACH = 4
_MSG_BYE = 5

_HEADER_FMT = "!BI"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)

_DEFAULT_SHM_THRESHOLD = 65536

_IS_WINDOWS = sys.platform == "win32"


# --- Environment knobs ----------------------------------------------------


def _user_tag() -> str:
    """Portable per-user identifier for default socket-path naming.

    Returns:
        A short string unique per local user.
    """
    getuid = getattr(os, "getuid", None)
    if getuid is not None:
        return str(getuid())
    return os.environ.get("USERNAME") or os.environ.get("USER") or "default"


def _default_socket_path() -> str:
    """Default socket path, overridable via ``GOGGLES_SOCKET``.

    On Unix this is the Unix-domain-socket path. On Windows it is the
    sidecar discovery file that records the TCP port the host is listening
    on; the file itself is not a socket.

    Returns:
        Absolute filesystem path.
    """
    override = os.getenv("GOGGLES_SOCKET")
    if override:
        return override
    runtime_dir = os.getenv("XDG_RUNTIME_DIR") or tempfile.gettempdir()
    return os.path.join(runtime_dir, f"goggles-{_user_tag()}.sock")


def _default_shm_threshold() -> int:
    """Default shared-memory threshold in bytes.

    Returns:
        Minimum payload size (bytes) that triggers the shm side-channel.
    """
    raw = os.getenv("GOGGLES_SHM_THRESHOLD")
    if raw is None:
        return _DEFAULT_SHM_THRESHOLD
    try:
        return max(0, int(raw))
    except ValueError:
        _log.warning(
            "Invalid GOGGLES_SHM_THRESHOLD=%r; using default %d",
            raw,
            _DEFAULT_SHM_THRESHOLD,
        )
        return _DEFAULT_SHM_THRESHOLD


# --- Socket helpers -------------------------------------------------------


def _recvall(sock: socket.socket, n: int) -> bytes | None:
    """Read exactly ``n`` bytes from ``sock``.

    Args:
        sock: Connected stream socket.
        n: Number of bytes to read.

    Returns:
        The read bytes, or None if the peer closed the connection.
    """
    buf = bytearray()
    while len(buf) < n:
        try:
            chunk = sock.recv(n - len(buf))
        except (ConnectionResetError, OSError):
            return None
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


# --- SMALL frame (inline pickle) ------------------------------------------


def _pack_small_frame(event: Event) -> bytearray:
    """Build the full MSG_SMALL wire frame in a single allocation.

    The total frame size (5-byte header + body) is computed up front,
    the bytearray is sized once, and every part (header, length
    prefixes, main pickle, each out-of-band buffer) is written via
    ``struct.pack_into`` / slice-assign — one memcpy per part. The send
    loop then hands this bytearray straight to ``sendall``, so there is
    no intermediate ``bytes(...)`` cast and no header concatenation.

    Args:
        event: Event to serialize.

    Returns:
        A complete wire frame (header + body), ready for ``sendall``.
    """
    buffers: list[pickle.PickleBuffer] = []
    main = pickle.dumps(event, protocol=5, buffer_callback=buffers.append)
    raws = [buf.raw() for buf in buffers]

    body_len = 4 + len(main) + 4 + sum(4 + len(r) for r in raws)
    out = bytearray(_HEADER_SIZE + body_len)

    struct.pack_into(_HEADER_FMT, out, 0, _MSG_SMALL, body_len)
    offset = _HEADER_SIZE

    struct.pack_into("!I", out, offset, len(main))
    offset += 4
    out[offset : offset + len(main)] = main
    offset += len(main)

    struct.pack_into("!I", out, offset, len(buffers))
    offset += 4

    for raw in raws:
        n = len(raw)
        struct.pack_into("!I", out, offset, n)
        offset += 4
        out[offset : offset + n] = raw
        offset += n

    return out


def _unpack_small(payload: bytes) -> Event:
    """Reverse the body half of :func:`_pack_small_frame`.

    The reader strips the 5-byte header and passes the remaining body
    to this function.

    Args:
        payload: Frame body (everything after ``_HEADER_SIZE``).

    Returns:
        Reconstructed Event.
    """
    offset = 0
    (mlen,) = struct.unpack_from("!I", payload, offset)
    offset += 4
    main = payload[offset : offset + mlen]
    offset += mlen
    (num_bufs,) = struct.unpack_from("!I", payload, offset)
    offset += 4
    buffers: list[bytes] = []
    for _ in range(num_bufs):
        (blen,) = struct.unpack_from("!I", payload, offset)
        offset += 4
        buffers.append(payload[offset : offset + blen])
        offset += blen
    return pickle.loads(main, buffers=buffers)


# --- LARGE frame (shared-memory side-channel) -----------------------------


def _pack_large(event: Event, shm_name: str) -> bytes:
    """Serialize a LARGE-mode event.

    The numpy payload has already been copied into shared memory named
    ``shm_name``; the wire payload carries metadata needed to reconstruct
    the array on the host side.

    Args:
        event: Event whose ``payload`` is a numpy ndarray.
        shm_name: Name of the shared memory block holding the array bytes.

    Returns:
        Pickled metadata + a stripped event (payload replaced by None).

    Raises:
        TypeError: If ``event.payload`` is not a ``numpy.ndarray``.
    """
    arr = event.payload
    if not isinstance(arr, np.ndarray):
        raise TypeError(
            "LARGE path requires numpy.ndarray payload, got "
            f"{type(arr).__name__}"
        )
    stripped = Event(
        kind=event.kind,
        scope=event.scope,
        payload=None,
        filepath=event.filepath,
        lineno=event.lineno,
        level=event.level,
        step=event.step,
        time=event.time,
        extra=event.extra,
    )
    meta: dict[str, Any] = {
        "shm_name": shm_name,
        "dtype": str(arr.dtype),
        "shape": tuple(arr.shape),
        "nbytes": int(arr.nbytes),
        "event": stripped,
    }
    return pickle.dumps(meta, protocol=5)


def _unpack_large(payload: bytes) -> Event:
    """Reconstruct an Event from a LARGE frame and unlink the named shm.

    Args:
        payload: Pickled metadata produced by :func:`_pack_large`.

    Returns:
        The reconstructed Event with its numpy payload materialized.
    """
    meta = pickle.loads(payload)
    shm_name: str = meta["shm_name"]
    dtype = np.dtype(meta["dtype"])
    shape: tuple[int, ...] = tuple(meta["shape"])
    stripped: Event = meta["event"]

    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        view = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        arr = np.array(view, copy=True)
    finally:
        shm.close()
        _try_unlink_shm(shm_name)
    return Event(
        kind=stripped.kind,
        scope=stripped.scope,
        payload=arr,
        filepath=stripped.filepath,
        lineno=stripped.lineno,
        level=stripped.level,
        step=stripped.step,
        time=stripped.time,
        extra=stripped.extra,
    )


# --- Shared-memory housekeeping -------------------------------------------

_SHM_NAME_PREFIX = "goggles_"
_SHM_REAP_AGE_S = 300.0
_LINUX_SHM_DIR = "/dev/shm"


def _next_shm_name() -> str:
    """Return a uniquely-prefixed shared-memory name for this process.

    The ``goggles_`` prefix lets the host opportunistically reap blocks
    that survived a crash without the consumer ever processing them.

    Returns:
        A name suitable for ``SharedMemory(create=True, name=...)``.
    """
    return f"{_SHM_NAME_PREFIX}{os.getpid()}_{secrets.token_hex(8)}"


def _reap_orphan_shm(max_age_s: float = _SHM_REAP_AGE_S) -> int:
    """Best-effort sweep of stale ``goggles_*`` shared-memory blocks.

    The :mod:`multiprocessing.shared_memory` resource tracker covers
    in-process leaks, but a host that crashes between receiving a
    LARGE frame and unlinking the named segment leaves the block
    behind. This sweep runs at host startup and unlinks any
    ``goggles_*`` segment older than ``max_age_s`` seconds. Linux only
    (``/dev/shm`` is the visible mount); a no-op elsewhere.

    Args:
        max_age_s: Reap blocks whose mtime is older than this many
            seconds. The default leaves enough headroom that a busy
            client's just-allocated segments are never touched.

    Returns:
        Count of segments unlinked (zero on non-Linux or empty dir).
    """
    if not os.path.isdir(_LINUX_SHM_DIR):
        return 0
    try:
        names = os.listdir(_LINUX_SHM_DIR)
    except OSError:
        return 0
    cutoff = time.time() - max_age_s
    reaped = 0
    for name in names:
        if not name.startswith(_SHM_NAME_PREFIX):
            continue
        path = os.path.join(_LINUX_SHM_DIR, name)
        try:
            if os.path.getmtime(path) >= cutoff:
                continue
        except OSError:
            continue
        try:
            os.unlink(path)
            reaped += 1
        except OSError:
            pass
    return reaped


def _try_unlink_shm(name: str) -> None:
    """Best-effort unlink of a named shm block, tolerant to already-gone.

    Args:
        name: Name of the shared-memory block to remove.
    """
    try:
        shm = shared_memory.SharedMemory(name=name)
    except FileNotFoundError:
        return
    except OSError:
        return
    try:
        shm.close()
    except OSError:
        pass
    try:
        shm.unlink()
    except (FileNotFoundError, OSError):
        pass
