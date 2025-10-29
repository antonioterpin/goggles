"""Events routing across files, processes, and machines.

This module encapsulates the multi-machine, multi-process routing of events
via the EventBus class. It uses a client-server model where one process
acts as the host (server) and others connect to it (clients).

Example:
>>> bus = get_bus()

"""

from __future__ import annotations

from typing import Optional
import portal
import socket
import netifaces

from goggles import EventBus, Event, GOGGLES_HOST, GOGGLES_PORT

# Singleton factory ---------------------------------------------------------
__singleton_client: Optional[portal.Client] = None
__singleton_server: Optional[portal.Server] = None
__singleton_core_event_bus: Optional[EventBus] = None


def __i_am_host() -> bool:
    """Return whether this process is the goggles event bus host.

    Returns:
        bool: True if this process is the host, False otherwise.

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


def get_bus() -> EventBus:
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
        EventBus: Singleton instance.

    """
    global __singleton_core_event_bus
    if __i_am_host():
        global __singleton_server
        if __singleton_core_event_bus is None:
            __singleton_core_event_bus = EventBus()
            __singleton_server = portal.Server(
                GOGGLES_PORT, name=f"EventBus-Server@{GOGGLES_HOST}"
            )
            __singleton_server.bind("attach", __singleton_core_event_bus.attach)
            __singleton_server.bind("detach", __singleton_core_event_bus.detach)
            __singleton_server.bind("emit", __singleton_core_event_bus.emit)
            __singleton_server.start(block=False)

    global __singleton_client
    if __singleton_client is None:
        __singleton_client = portal.Client(
            f"{GOGGLES_HOST}:{GOGGLES_PORT}",
            name=f"EventBus-Client@{socket.gethostname()}",
        )

    return __singleton_client


__all__ = ["Event", "CoreEventBus", "get_bus", "Handler"]
