"""Transport package: iceoryx2-backed same-machine event routing.

See ``docs/guides/transport.md`` for the package layout.
"""

from __future__ import annotations

from ._iox import LocalTransport
from ._protocol import Transport

__all__ = [
    "LocalTransport",
    "Transport",
]
