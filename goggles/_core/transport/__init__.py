"""Transport package: cross-platform same-machine event routing.

Public surface:
    - :class:`Transport` — the protocol every implementation satisfies.
    - :class:`LocalTransport` — the default local-machine implementation
      (auto-elected host, AF_UNIX on Unix / TCP loopback on Windows).

Internals:
    - :mod:`._frames` — wire format, environment knobs, shm housekeeping.
    - :mod:`._endpoints` — platform-specific socket binding/connecting.
    - :mod:`._protocol` — the :class:`Transport` protocol declaration.
    - :mod:`._local` — :class:`LocalTransport` and its private state.

Imports below preserve the flat ``from goggles._core.transport import X``
pattern that tests and routing code rely on.
"""

from __future__ import annotations

from ._endpoints import (  # noqa: F401
    _Endpoint,
    _endpoint,
    _UnixEndpoint,
    _WindowsEndpoint,
)
from ._frames import (  # noqa: F401
    _DEFAULT_SHM_THRESHOLD,
    _HEADER_FMT,
    _HEADER_SIZE,
    _IS_WINDOWS,
    _LINUX_SHM_DIR,
    _MSG_ATTACH,
    _MSG_BYE,
    _MSG_DETACH,
    _MSG_LARGE,
    _MSG_SMALL,
    _SHM_NAME_PREFIX,
    _SHM_REAP_AGE_S,
    _default_shm_threshold,
    _default_socket_path,
    _next_shm_name,
    _pack_large,
    _pack_small_frame,
    _reap_orphan_shm,
    _recvall,
    _try_unlink_shm,
    _unpack_large,
    _unpack_small,
    _user_tag,
)
from ._local import _SENTINEL, LocalTransport, _SendItem  # noqa: F401
from ._protocol import Transport

__all__ = [
    "LocalTransport",
    "Transport",
]
