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

# The default is async mode, but one can change the environment variable
# GOGGLES_ASYNC to "0" to disable it globally.
# Alternatively, one can disable it per-call:
logger.debug("This is a synchronous debug message.", async_mode=False)

# Note that having attached two handlers will result in duplicated outputs
# if both are eligible.
logger.info("This message will be logged by both handlers.")

gg.finish()
