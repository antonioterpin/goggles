"""Integration module for Goggles core.

This module defines the handlers to be attached to the EventBus to dispatch
events to the appropriate integration modules.

Example:
    class PrintHandler(TextHandler):
        def emit(self, record: LogRecord) -> None:
            print(self.format(record))

"""

from .console import ConsoleHandler
from .storage import LocalStorageHandler

__all__ = [
    "ConsoleHandler",
    "LocalStorageHandler",
]

try:
    from .wandb import WandBHandler

    __all__ = [
        *__all__,
        "WandBHandler",
    ]
except Exception:
    pass
