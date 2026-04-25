"""Public Transport protocol.

The contract every transport implementation must satisfy. ``runtime_checkable``
so ``isinstance(x, Transport)`` works for the singleton-rebuild check in
``goggles._core.routing``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from goggles.types import Event


@runtime_checkable
class Transport(Protocol):
    """Contract for routing Goggles events to the EventBus."""

    @property
    def is_running(self) -> bool:
        """Whether the transport is accepting new events.

        Returns:
            True while the transport is live, False after shutdown or an
            unrecoverable send failure.
        """
        ...

    def emit(self, event: Event) -> None:
        """Route an event asynchronously (fire-and-forget).

        Args:
            event: Event to route.
        """
        ...

    def emit_sync(self, event: Event) -> None:
        """Route an event, blocking until it has reached the bus.

        In client mode this is best-effort: there is no cross-process ack,
        so it behaves like :meth:`emit` plus a local queue flush.

        Args:
            event: Event to route.
        """
        ...

    def attach(self, handlers: list[dict], scopes: list[str]) -> None:
        """Attach handlers under the given scopes.

        Args:
            handlers: Serialized handler dicts (see ``Handler.to_dict``).
            scopes: Scopes under which to attach.
        """
        ...

    def detach(self, handler_name: str, scope: str) -> None:
        """Detach a handler from the given scope.

        Args:
            handler_name: Name of the handler to detach.
            scope: Scope to detach from.
        """
        ...

    def shutdown(self, timeout: float | None = None) -> None:
        """Flush pending events and release resources.

        Args:
            timeout: Max seconds per background thread for clean exit.
                ``None`` means wait indefinitely.
        """
        ...
