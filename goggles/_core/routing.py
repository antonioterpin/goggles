"""Events routing across files, processes, and machines.

This module encapsulates the multi-machine, multi-process routing of events
via the EventBus class. It uses a client-server model where one process
acts as the host (server) and others connect to it (clients).

Example:
>>> bus = get_bus()

"""

from __future__ import annotations

import socket
import time
from concurrent.futures import Future
from typing import Any, cast

import netifaces
import portal

from goggles import (
    GOGGLES_HOST,
    GOGGLES_PORT,
    GOGGLES_SUPPRESS_CONNECTIVITY_LOGS,
    Event,
    EventBus,
)


class GogglesClient:
    """Client for the Goggles EventBus.

    Wraps a portal.Client to provide event emission with automatic
    future management to prevent memory leaks.

    Attributes:
        futures: List of pending futures from emitted events.
    """

    _client: portal.Client
    futures: list[Future[Any]]
    _pruning_threshold: int

    def __init__(
        self,
        addr: str = f"{GOGGLES_HOST}:{GOGGLES_PORT}",
        name: str = f"EventBus-Client@{socket.gethostname()}",
        pruning_threshold: int = 100,
        suppress_connectivity_logs: bool = GOGGLES_SUPPRESS_CONNECTIVITY_LOGS,
    ) -> None:
        """Initialize the client.

        Args:
            addr: Address of the EventBus server.
            name: Name of this client.
            pruning_threshold: Maximum number of futures to track before
                triggering cleanup of completed ones; cleanup occurs when the
                number of futures exceeds this threshold. Defaults to 100.
            suppress_connectivity_logs: If True, suppress connectivity logs.

        """
        self.futures = []
        self._pruning_threshold = pruning_threshold
        # Increase maxinflight to avoid stalling the main thread on
        # high-throughput logging.
        self._client = portal.Client(
            addr=addr,
            name=name,
            maxinflight=1024,
            max_send_queue=1024,
            logging=not suppress_connectivity_logs,
        )

    def emit(self, event: Event) -> Future[Any]:
        """Emit an event via the EventBus.

        Args:
            event: The event to emit.

        Returns:
            A Future that will complete when the event is delivered.
        """
        # Periodic cleanup of finished futures to avoid memory leak
        if len(self.futures) > self._pruning_threshold:
            self.futures = [f for f in self.futures if not f.done()]

        future = cast(Future[Any], self._client.emit(event.to_dict()))
        self.futures.append(future)
        # Future type is not fully specified in portal.Client.
        return future

    def shutdown(self, timeout: float | None = None) -> Future[Any]:
        """Shutdown the EventBus client.

        Args:
            timeout: Optional timeout in seconds to wait for pending futures
                before issuing client shutdown. If None or <= 0, waits
                indefinitely.

        Returns:
            A Future that will complete when the client is fully shut down.
        """
        deadline: float | None = None
        if timeout is not None and timeout > 0:
            deadline = time.monotonic() + timeout

        for future in self.futures:
            try:
                if deadline is None:
                    future.result()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    future.result(timeout=remaining)
            except TimeoutError:
                break
            except Exception:
                # Ignore failed/dropped futures during shutdown.
                pass

        # Keep only unfinished futures (if any) for introspection after
        # shutdown.
        self.futures = [
            future
            for future in self.futures
            if not getattr(future, "done", lambda: False)()
        ]
        return cast(Future[Any], self._client.shutdown())

    def attach(self, handlers: list[dict], scopes: list[str]) -> None:
        """Attach a handler under the given scope.

        Args:
            handlers:
                The serialized handlers to attach to the scopes.
            scopes: The scopes under which to attach.

        """
        self._client.attach(handlers, scopes)

    def detach(self, handler_name: str, scope: str) -> None:
        """Detach a handler from the given scope.

        Args:
            handler_name: The name of the handler to detach.
            scope: The scope from which to detach.
        """
        self._client.detach(handler_name, scope)


# Singleton factory ---------------------------------------------------------
__singleton_client: GogglesClient | None = None
__singleton_server: portal.Server | None = None


def __i_am_host() -> bool:
    """Return whether this process is the goggles event bus host.

    Returns:
        True if this process is the host, False otherwise.

    """
    # If GOGGLES_HOST is localhost/127.0.0.1, we are always the host
    if GOGGLES_HOST in ("localhost", "127.0.0.1", "::1"):
        return True

    # Get all local IP addresses
    hostname = socket.gethostname()
    local_ips = set()

    # Add hostname resolution
    try:
        local_ips.add(socket.gethostbyname(hostname))
    except socket.gaierror:
        pass

    # Add all interface IPs
    for interface in netifaces.interfaces():
        addrs = netifaces.ifaddresses(interface)
        for addr_family in [netifaces.AF_INET, netifaces.AF_INET6]:
            if addr_family in addrs:
                for addr_info in addrs[addr_family]:
                    if "addr" in addr_info:
                        local_ips.add(addr_info["addr"])

    # Check if GOGGLES_HOST matches any local IP
    return GOGGLES_HOST in local_ips


def __is_port_in_use(host: str, port: int) -> bool:
    """Check if a port is already in use.

    Args:
        host: The host to check.
        port: The port to check.

    Returns:
        True if the port is in use, False otherwise.

    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)  # 1 second timeout
            result = sock.connect_ex((host, port))
            return result == 0  # 0 means connection successful (port in use)
    except Exception:
        return False


def get_bus() -> GogglesClient:
    """Return the process-wide EventBus singleton.

    This function ensures that there is a single instance of the
    EventBus for the entire application, even if distributed across machines.

    It uses a client-server model where one process acts as the host
    (server) and others connect to it (clients). The host is determined
    based on the GOGGLES_HOST configuration. The methods of EventBus are
    exposed via a portal server for remote invocation.

    NOTE: It is not thread-safe. It works on multiple machines and multiple
    processes, but it is not guaranteed to work consistently for multiple
    threads within the same process.

    Returns:
        The singleton EventBus client.

    """
    if __i_am_host() and not __is_port_in_use(GOGGLES_HOST, int(GOGGLES_PORT)):
        global __singleton_server  # noqa: PLW0603
        try:
            event_bus = EventBus()
            # TODO(issue #116): Introduce GogglesServer wrapper and remove
            # direct portal.Server construction.
            # https://github.com/antonioterpin/goggles/issues/116
            server = portal.Server(
                GOGGLES_PORT,
                name=f"EventBus-Server@{socket.gethostname()}",
                logging=not GOGGLES_SUPPRESS_CONNECTIVITY_LOGS,
            )
            server.bind("attach", event_bus.attach)
            server.bind("detach", event_bus.detach)
            server.bind("emit", event_bus.emit)
            server.bind("shutdown", event_bus.shutdown)
            server.start(block=False)
            __singleton_server = server
        except OSError:
            # Fallback: Server creation failed for other reasons
            # (e.g. concurrency), no further need
            pass

    global __singleton_client  # noqa: PLW0603
    if __singleton_client is None:
        __singleton_client = GogglesClient(
            addr=f"{GOGGLES_HOST}:{GOGGLES_PORT}",
            name=f"EventBus-Client@{socket.gethostname()}",
            suppress_connectivity_logs=GOGGLES_SUPPRESS_CONNECTIVITY_LOGS,
        )

    return __singleton_client


__all__ = ["Event", "get_bus"]
