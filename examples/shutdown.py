"""Example of graceful shutdown using Goggles."""

import time
from goggles import Goggles, Severity, GracefulShutdown

Goggles.set_config(to_file=False, to_terminal=True, level=Severity.DEBUG)
Goggles.info("Starting main loop. Press Ctrl-C to stop.")
with GracefulShutdown("Received interrupt, shutting down...") as gs:
    iteration = 0
    while not gs.stop:
        iteration += 1
        Goggles.debug(f"Iteration {iteration}: still running")
        # Simulate work
        time.sleep(1)

Goggles.info("Cleanup complete. Exiting.")
