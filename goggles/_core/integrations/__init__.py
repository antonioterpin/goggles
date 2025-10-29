"""Integration module for Goggles core.

This module defines the handlers to be attached to the EventBus to dispatch
events to the appropriate integration modules.

Example:
    class PrintHandler(TextHandler):
        def emit(self, record: LogRecord) -> None:
            print(self.format(record))

"""

# TODO: actually write the docs here

from .console import ConsoleHandler

# from .jsonl import JsonlHandler

__all__ = [
    "ConsoleHandler",
    # "JsonlHandler",
]

# try:
#     from .wandb import WandBHandler

#     integrations.append(WandBHandler)
# except ImportError:
#     pass
