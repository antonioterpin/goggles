"""Event bus for handling log events and metrics.

This module implements a synchronous, scope-aware Event Bus for routing
structured events to registered handlers. Handlers can be attached under either
a global scope (always active) or a run scope (active only during an active run).
Handlers can specify filters based on event kind, namespace, and log level.
Handlers must implement the `Handler` protocol defined in `goggles`.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from time import time
from typing import Any, Iterable, Literal, Optional, Protocol

from goggles import EventBus, Handler


Scope = Literal["global", "run"]
Kind = Literal["log", "metric", "image", "artifact"]


@dataclass(frozen=True)
class Event:
    """Structured event routed through the EventBus.

    Args:
        kind (Kind): Event kind ("log", "metric", "image", "artifact").
        namespace (str): Dotted namespace, e.g. "goggles.core".
        payload (dict[str, Any] | Any): Event payload. Handlers must be robust
            to arbitrary payloads.
        level (int | None): Numeric level for log-like events (e.g., logging levels).
        run_id (str | None): Run identifier, if any.
        ts (float | None): POSIX timestamp in seconds. Defaults to `time()`.

    Notes:
        Only `kind` and `namespace` are used by the core router. Handlers may
        inspect the full event.

    """

    kind: Kind
    namespace: str
    payload: Any
    level: Optional[int] = None
    run_id: Optional[str] = None
    ts: Optional[float] = None

    def with_defaults(self) -> "Event":
        """Return a copy with defaulted timestamp.

        Returns:
            Event: Event with `ts` filled if it was None.

        """
        if self.ts is None:
            return Event(
                kind=self.kind,
                namespace=self.namespace,
                payload=self.payload,
                level=self.level,
                run_id=self.run_id,
                ts=time(),
            )
        return self


@dataclass(frozen=True)
class _Filters:
    kinds: Optional[set[Kind]] = None
    min_level: Optional[int] = None
    namespaces: Optional[tuple[str, ...]] = None  # Prefix match

    def matches(self, event: Event) -> bool:
        if self.kinds is not None and event.kind not in self.kinds:
            return False
        if self.min_level is not None and event.level is not None:
            if event.level < self.min_level:
                return False
        if self.namespaces is not None:
            ns = event.namespace
            if not any(ns == p or ns.startswith(p + ".") for p in self.namespaces):
                return False
        return True


@dataclass
class _Entry:
    handler: Handler
    scope: Scope
    filters: _Filters


class DetachHandle:
    """A context-manager handle that detaches its handler on exit."""

    def __init__(self, bus: "EventBus", handler: Handler) -> None:
        self._bus = bus
        self._handler = handler
        self._detached = False

    def detach(self) -> None:
        """Detach the handler if still attached (idempotent)."""
        if not self._detached:
            self._bus.detach(self._handler)
            self._detached = True

    # Context manager API -------------------------------------------------
    def __enter__(self) -> "DetachHandle":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.detach()


class CoreEventBus:
    """Scope-aware synchronous Event Bus.

    Thread-safe registry of handlers attached under either the `global` or the
    `run` scope. `emit()` routes events to all matching handlers.
    This class implements the EventBus protocol.

    Attributes:
        _lock (RLock): Lock for thread-safe access.
        _entries (list[_Entry]): Registered handler entries.
        _run_active (bool): Whether the `run` scope is currently active.
        _current_run_id (Optional[str]): Current active run identifier.

    Examples:
        >>> bus = get_bus()
        >>> class Collector:
        ...     def __init__(self): self.events = []
        ...     def handle(self, event: Event) -> None: self.events.append(event)
        >>> collector = Collector()
        >>> _ = bus.attach(collector, scope="global", kinds={"log"})
        >>> bus.emit(Event(kind="log", namespace="goggles.core", payload={}))
        >>> len(collector.events)
        1

    """

    def __init__(self) -> None:
        """Initialize an empty CoreEventBus."""
        self._lock = RLock()
        self._entries: list[_Entry] = []
        self._run_active: bool = False
        self._current_run_id: Optional[str] = None

    def activate_run(self, run_id: str) -> None:
        """Mark the `run` scope as active.

        Args:
            run_id (str): Identifier of the active run.

        """
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id must be a non-empty string")
        with self._lock:
            self._run_active = True
            self._current_run_id = run_id

    def deactivate_run(self) -> None:
        """Deactivate the `run` scope."""
        with self._lock:
            self._run_active = False
            self._current_run_id = None

    def run_scope(self, run_id: str):
        """Context manager to activate/deactivate a run scope.

        Args:
            run_id (str): Identifier for the run.

        Returns:
            contextlib.AbstractContextManager: Context manager.

        """
        from contextlib import contextmanager

        @contextmanager
        def _cm():
            self.activate_run(run_id)
            try:
                yield self
            finally:
                self.deactivate_run()

        return _cm()

    def attach(
        self,
        handler: Handler,
        /,
        *,
        scope: Scope,
        kinds: Optional[Iterable[Kind]] = None,
        min_level: Optional[int] = None,
        namespaces: Optional[Iterable[str]] = None,
    ) -> DetachHandle:
        """Attach a handler with optional filters.

        Args:
            handler (Handler): Handler to register.
            scope (Scope): Attachment scope ("global" or "run").
            kinds (Iterable[Kind] | None): Allowed kinds for this handler.
            min_level (int | None): Minimum numeric level for log events.
            namespaces (Iterable[str] | None): Namespace prefixes to match.

        Returns:
            DetachHandle: A handle that can detach the handler or be used as a
                context manager.

        Raises:
            TypeError: If `handler` does not implement `handle`.
            ValueError: If filters are invalid.

        """
        if not isinstance(handler, Handler):
            raise TypeError("handler must implement the Handler protocol")
        if kinds is not None:
            kinds_set = set(kinds)
            if not kinds_set:
                raise ValueError("kinds cannot be empty if provided")
            invalid = kinds_set - {"log", "metric", "image", "artifact"}
            if invalid:
                raise ValueError(f"invalid kinds: {sorted(invalid)}")
        else:
            kinds_set = None
        ns_tuple = tuple(namespaces) if namespaces is not None else None
        filt = _Filters(kinds=kinds_set, min_level=min_level, namespaces=ns_tuple)
        entry = _Entry(handler=handler, scope=scope, filters=filt)
        with self._lock:
            self._entries.append(entry)
        return DetachHandle(self, handler)

    def detach(self, handler: Handler) -> None:
        """Detach all registrations for `handler`.

        If the handler exposes a `close()` method, it will be called once.
        """
        to_close: Optional[Handler] = None
        with self._lock:
            before = len(self._entries)
            self._entries = [e for e in self._entries if e.handler is not handler]
            removed = before != len(self._entries)
            if removed:
                to_close = handler if hasattr(handler, "close") else None
        if to_close is not None:
            try:
                to_close.close()
            except Exception:
                # Best-effort close; avoid raising during detach.
                pass

    def emit(self, event: Event) -> None:
        """Route an event to matching handlers.

        Args:
            event (Event): Event to route.

        Raises:
            TypeError: If `event` has invalid types.

        """
        if not isinstance(event.kind, str) or event.kind not in {
            "log",
            "metric",
            "image",
            "artifact",
        }:
            raise TypeError(
                "event.kind must be one of 'log','metric','image','artifact'"
            )
        if not isinstance(event.namespace, str) or not event.namespace:
            raise TypeError("event.namespace must be a non-empty string")

        with self._lock:
            active_run = self._run_active
            run_id = self._current_run_id
            # Snapshot entries to avoid holding the lock during handler calls.
            entries = tuple(self._entries)
        ev = event.with_defaults()
        if ev.run_id is None and active_run:
            # Enrich with current run id for convenience.
            ev = Event(
                kind=ev.kind,
                namespace=ev.namespace,
                payload=ev.payload,
                level=ev.level,
                run_id=run_id,
                ts=ev.ts,
            )
        for entry in entries:
            if entry.scope == "run" and not active_run:
                continue
            if not entry.filters.matches(ev):
                continue
            entry.handler.handle(ev)


# Singleton factory ---------------------------------------------------------
_singleton: Optional[CoreEventBus] = None
_lock_singleton = RLock()


def get_bus() -> EventBus:
    """Return the process-wide EventBus singleton.

    Returns:
        EventBus: Singleton instance.

    """
    global _singleton
    if _singleton is None:
        with _lock_singleton:
            if _singleton is None:
                _singleton = CoreEventBus()
    return _singleton


__all__ = ["Event", "CoreEventBus", "get_bus", "Handler", "Scope", "Kind"]
