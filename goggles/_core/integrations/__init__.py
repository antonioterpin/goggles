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
from .storage import LocalStorageHandler

from .jsonl import JsonlHandler

try:
    from .wandb import WandBHandler
except ImportError:
    WandBHandler = None  # type: ignore
__all__ = [
    "ConsoleHandler",
    "JsonlHandler",
    "WandBHandler",
    "LocalStorageHandler",
]
