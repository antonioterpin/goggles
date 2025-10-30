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


def __is_port_in_use(host: str, port: int) -> bool:
    """Check if a port is already in use.

    Args:
        host (str): The host to check.
        port (int): The port to check.

    Returns:
        bool: True if the port is in use, False otherwise.

    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)  # 1 second timeout
            result = sock.connect_ex((host, port))
            return result == 0  # 0 means connection successful (port in use)
    except Exception:
        return False


def get_bus() -> portal.Client:
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
        portal.Client: The singleton EventBus client.

    """
    if __i_am_host() and not __is_port_in_use(GOGGLES_HOST, int(GOGGLES_PORT)):
        global __singleton_server
        try:
            event_bus = EventBus()
            server = portal.Server(
                GOGGLES_PORT, name=f"EventBus-Server@{socket.gethostname()}"
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

    global __singleton_client
    if __singleton_client is None:
        __singleton_client = portal.Client(
            f"{GOGGLES_HOST}:{GOGGLES_PORT}",
            name=f"EventBus-Client@{socket.gethostname()}",
        )

    return __singleton_client


__all__ = ["Event", "CoreEventBus", "get_bus", "Handler"]
