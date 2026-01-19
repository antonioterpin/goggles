"""Example of debugging utils: enabling socket logging."""

# Disable suppression of connectivity logs via environment variable
import os

os.environ["GOGGLES_SUPPRESS_CONNECTIVITY_LOGS"] = "0"

import goggles as gg

# All that follows is as in 01_basic_run.py,
# but now with socket connectivity logs enabled.
logger = gg.get_logger("examples.basic")
gg.attach(
    gg.ConsoleHandler(name="examples.basic.console", level=gg.INFO),
    scopes=["global"],
)

# Because the logging level is set to INFO, the debug message will not be shown.
logger.info("Hello, world!")
logger.debug("you won't see this at INFO")

# There are different colors
logger.warning("This is a warning!")
logger.error("This is an error!")
logger.critical("This is critical!")

# If we attach a handler with a lower logging level, debug messages will be shown.
gg.attach(
    gg.ConsoleHandler(name="examples.basic.debug_console", level=gg.DEBUG),
    scopes=["global"],
)

logger.debug("Now you will see this debug message!")

gg.finish()
