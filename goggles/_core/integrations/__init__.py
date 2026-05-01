"""Integration module for Goggles core.

This module defines the handlers to be attached to the EventBus to dispatch
events to the appropriate integration modules.

Example:
    class PrintHandler(TextHandler):
        def emit(self, record: LogRecord) -> None:
            print(self.format(record))

"""

from importlib import import_module
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any

from .console import ConsoleHandler
from .storage import LocalStorageHandler

if TYPE_CHECKING:
    # Static analyzers can't see the runtime __getattr__ + find_spec dance,
    # so import the symbol unconditionally for type checking. At runtime,
    # the import is gated on the wandb extra being installed.
    from .wandb import WandBHandler  # noqa: F401

__all__: list[str] = [
    "ConsoleHandler",
    "LocalStorageHandler",
]

if find_spec("wandb") is not None:
    __all__.append("WandBHandler")


def __getattr__(name: str) -> Any:
    if name == "WandBHandler":
        module = import_module(".wandb", __name__)
        return module.WandBHandler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
