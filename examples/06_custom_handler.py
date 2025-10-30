import goggles as gg
import logging


class CustomConsoleHandler(gg.ConsoleHandler):
    """A custom console handler that adds a prefix to each log message."""

    def handle(self, event: gg.Event) -> None:
        dict = event.to_dict()

        dict["payload"] = f"[CUSTOM PREFIX] {dict['payload']}"

        event = gg.Event.from_dict(dict)
        super().handle(event)


# Register the custom handler so it can be serialized/deserialized
gg.register_handler(CustomConsoleHandler)

# In this basic example, we set up a customized logger that outputs to console.
logger = gg.get_logger("examples.custom_handler")


gg.attach(
    CustomConsoleHandler(name="examples.custom.console", level=logging.INFO),
    scopes=["global"],
)
# Because the logging level is set to INFO, the debug message will not be shown.
logger.info("Hello, world!")
logger.debug("you won't see this at INFO")
