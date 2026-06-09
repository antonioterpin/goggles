"""Example: handlers run in a dedicated host subprocess (the default).

By default goggles spawns a dedicated host subprocess to own the EventBus and
run the handlers, so *this* process -- and every other -- connects as a
*client*. That keeps heavy or blocking handlers (notably the W&B uploader) off
the application's interpreter, where they could otherwise starve its
latency-critical paths (RPC servers, control/sim loops) when logging
back-pressures.

Nothing special is needed -- it is the default. Set ``GOGGLES_DEDICATED_HOST=0``
to opt out and host in-process instead (the first process to bind the socket
becomes the host).

Run:
    python examples/08_dedicated_host.py
"""

import os
from pathlib import Path

import goggles as gg


def main() -> None:
    """Log a few events; the handler runs in the dedicated host subprocess."""
    # The dedicated host is already the default -- no GOGGLES_DEDICATED_HOST
    # needed. (To host in-process instead, set GOGGLES_DEDICATED_HOST=0 before
    # the first goggles call.) Bound the host's graceful-shutdown budget
    # (drain + handler close, e.g. finishing W&B runs) for the demo.
    os.environ.setdefault("GOGGLES_SHUTDOWN_TIMEOUT", "30")

    gg.attach(
        gg.LocalStorageHandler(
            path=Path("examples/logs/dedicated"), name="local"
        ),
        scopes=["global"],
    )
    # Swap the handler above for a WandBHandler to keep W&B uploads entirely
    # off this process:
    #     gg.attach(gg.WandBHandler(project="my-project"), scopes=["global"])

    bus = gg.get_bus()
    print(f"this process is the host: {bus.is_host}  (False => dedicated host)")

    log = gg.get_logger(__name__, scope="global")
    for i in range(5):
        log.info(f"event {i}")

    # finish() flushes this client, then drains + terminates the host
    # subprocess (closing handlers, e.g. finishing W&B runs).
    gg.finish()
    print("done; see examples/logs/dedicated/log.jsonl")


if __name__ == "__main__":
    main()
