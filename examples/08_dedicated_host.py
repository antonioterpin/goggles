"""Example: run the goggles host in a dedicated subprocess.

By default the goggles *host* -- the process that owns the EventBus and runs
the handlers -- is whichever process first binds the socket, which is usually
the application itself. Heavy or blocking handlers (notably the W&B uploader)
then run on the application's interpreter and can starve its latency-critical
paths (RPC servers, control/sim loops) when logging back-pressures.

Setting ``GOGGLES_DEDICATED_HOST=1`` makes goggles spawn a dedicated host
subprocess instead: this process (and every other) connects as a *client*, the
handlers run in the subprocess, and ``gg.finish()`` drains and terminates it.
Handlers are unchanged -- ``attach(...)`` ships them to the host over the wire,
so a ``WandBHandler`` attached here uploads from the subprocess, not from here.

Run:
    python examples/08_dedicated_host.py
"""

import os
from pathlib import Path

import goggles as gg


def main() -> None:
    """Attach a handler (it runs in the dedicated host) and log a few events."""
    # Opt in BEFORE the first goggles call: the host is spawned lazily on the
    # first ``get_bus()`` / ``attach()``. Bound the host's graceful-shutdown
    # budget (drain + handler close, e.g. finishing W&B runs) for the demo.
    os.environ.setdefault("GOGGLES_DEDICATED_HOST", "1")
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
