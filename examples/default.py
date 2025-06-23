"""Example of using Goggles with default settings."""

from goggles import Goggles

Goggles.cleanup()  # Make sure to clean up any previous runs
Goggles.info("You can log with the specified defauls.")
Goggles.warning(
    "If you do not see this on the terminal, you did not set .goggles-default.yaml."
)
Goggles.cleanup()  # Clean up again to ensure no artifacts are left
