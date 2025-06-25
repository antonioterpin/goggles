"""Example of graceful shutdown using Goggles."""

import time
import goggles

goggles.info("Starting main loop. Press Ctrl-C to stop.")
with goggles.GracefulShutdown("Received interrupt, shutting down...") as gs:
    iteration = 0
    while not gs.stop:
        iteration += 1
        goggles.debug(f"Iteration {iteration}: still running")
        # Simulate work
        time.sleep(1)

goggles.info("Cleanup complete. Exiting.")
