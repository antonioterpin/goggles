"""Importable handlers for ``examples/105_benchmark.py``.

This lives in its own module (not the benchmark's ``__main__``) so the goggles
dedicated host -- a separate subprocess by default -- can import it via
``GOGGLES_HOST_IMPORTS`` and reconstruct the handler there. Handlers always run
in the host; the benchmark process is a client, so it cannot read the handler's
state directly. The counter therefore reports its per-kind totals by writing
them to a file, which the benchmark reads back after ``gg.finish()`` (this also
works unchanged when the host is in-process).
"""

from __future__ import annotations

import json
import threading
from typing import ClassVar

import goggles as gg
from goggles import Event, Kind


class DeliveryCounter:
    """Counts events delivered to the host, persisting totals to a file.

    Used to verify end-to-end reliability: the producer emits N events, then
    ``finish()`` drains everything; the recorded count must equal N. The
    totals are written to ``out_path`` on ``close()`` so the (client) benchmark
    process can read them after shutdown.

    Attributes:
        name: Handler identifier used by the bus's dedup logic.
        capabilities: Event kinds this handler claims to handle.
    """

    name = "goggles.benchmark.counter"
    capabilities: ClassVar[frozenset[Kind]] = frozenset(
        {
            "log",
            "metric",
            "image",
            "video",
            "artifact",
            "histogram",
            "vector",
            "vector_field",
        }
    )

    def __init__(self, out_path: str | None = None) -> None:
        """Initialize the counter.

        Args:
            out_path: File the per-kind totals are written to on ``close()``.
        """
        self._lock = threading.Lock()
        self._count_by_kind: dict[str, int] = {}
        self._out_path = out_path

    def can_handle(self, kind: Kind) -> bool:
        """Accept every event kind.

        Args:
            kind: Event kind (ignored).

        Returns:
            Always ``True``.
        """
        del kind
        return True

    def handle(self, event: Event) -> None:
        """Increment the count for ``event``'s kind.

        Args:
            event: The delivered event.
        """
        with self._lock:
            self._count_by_kind[event.kind] = (
                self._count_by_kind.get(event.kind, 0) + 1
            )

    def open(self) -> None:
        """No-op; nothing to open."""

    def close(self) -> None:
        """Persist the totals so the benchmark process can read them."""
        if not self._out_path:
            return
        with self._lock:
            data = dict(self._count_by_kind)
        try:
            with open(self._out_path, "w", encoding="utf-8") as handle:
                json.dump(data, handle)
        except OSError:
            pass

    @property
    def totals(self) -> dict[str, int]:
        """A snapshot of the per-kind delivery counts.

        Returns:
            Mapping of event kind to count seen so far.
        """
        with self._lock:
            return dict(self._count_by_kind)

    def to_dict(self) -> dict:
        """Serialize for transport to the host.

        Returns:
            The handler spec (class name + reconstruction data).
        """
        return {"cls": "DeliveryCounter", "data": {"out_path": self._out_path}}

    @classmethod
    def from_dict(cls, serialized: dict) -> DeliveryCounter:
        """Reconstruct a counter in the host process.

        Args:
            serialized: The ``data`` mapping produced by :meth:`to_dict`.

        Returns:
            A fresh :class:`DeliveryCounter` writing to the same file.
        """
        return cls(out_path=serialized.get("out_path"))


gg.register_handler(DeliveryCounter)
