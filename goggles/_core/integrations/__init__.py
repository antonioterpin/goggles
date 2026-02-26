"""Integrations module."""

from importlib import import_module
from importlib.util import find_spec
from typing import Any

from .console import ConsoleHandler

__all__: list[str] = [
    "ConsoleHandler",
]

if find_spec("wandb") is not None:
    __all__.append("WandBHandler")


def __getattr__(name: str) -> Any:
    if name == "WandBHandler":
        module = import_module(".wandb", __name__)
        return module.WandBHandler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
