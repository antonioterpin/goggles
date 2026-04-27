"""Transport package: cross-platform same-machine event routing.

See ``docs/guides/transport.md`` for the package layout and the
rationale behind the flat re-exports below.
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
