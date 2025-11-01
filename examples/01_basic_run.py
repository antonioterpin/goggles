import goggles as gg

# In this basic example, we set up a logger that outputs to the console.
logger = gg.get_logger("examples.basic")
gg.attach(
    gg.ConsoleHandler(name="examples.basic.console", level=gg.INFO),
    scopes=["global"],
)

# Because the logging level is set to INFO, the debug message will not be shown.
logger.info("Hello, world!")
logger.debug("you won't see this at INFO")

# If we attach a handler with a lower logging level, debug messages will be shown.
gg.attach(
    gg.ConsoleHandler(name="examples.basic.debug_console", level=gg.DEBUG),
    scopes=["global"],
)

logger.debug("Now you will see this debug message!")
