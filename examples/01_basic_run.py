import goggles as gg
import logging

# In this basic example, we set up a logger that outputs to the console.
logger = gg.get_logger("examples.basic")
gg.attach(
    gg.ConsoleHandler(name="examples.basic.console", level=logging.INFO),
    scopes=["global"],
)

# Because the logging level is set to INFO, the debug message will not be shown.
logger.info("Hello, world!")
logger.debug("you won't see this at INFO")
