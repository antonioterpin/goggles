"""Utilities for loading and pretty-printing configuration files."""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from typing import Any, TypeVar

from rich.console import Console
from rich.pretty import Pretty
from ruamel.yaml import YAML

Cfg = TypeVar("Cfg", bound="PrettyConfig")


class PrettyConfig(dict):
    """Dictionary subclass with pretty-printing and serialization methods."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Automatically wrap `__post_init__` in subclasses to sync dict.

        Args:
            **kwargs: Additional subclass initialization options.

        """
        super().__init_subclass__(**kwargs)

        orig_post_init = cls.__dict__.get("__post_init__")
        if orig_post_init is None:
            return

        # If subclass defines __post_init__,
        # wrap it so we always sync dict after it.
        if orig_post_init is PrettyConfig.__post_init__:
            return

        def __post_init__(self, *a: Any, **k: Any) -> None:
            orig_post_init(self, *a, **k)
            PrettyConfig.__post_init__(self)

        cls.__post_init__ = __post_init__  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """If a dataclass instance, sync dict contents with dataclass fields."""
        if not is_dataclass(self):
            return

        data: dict[str, Any] = {}
        for f in fields(self):
            if f.name.startswith("_"):
                continue
            data[f.name] = getattr(self, f.name)

        dict.__init__(self, data)

    def __str__(self) -> str:
        """Return a pretty-printed string of the configuration.

        Returns:
            Pretty-printed configuration text.

        """
        console = Console()
        plain = dict(self)
        with console.capture() as capture:
            console.print(Pretty(plain))
        return capture.get()

    __repr__ = __str__

    def to_dict(self) -> dict[str, Any]:
        """Return a plain Python dict.

        Returns:
            A shallow dictionary view of this configuration.

        """
        return dict(self)

    @classmethod
    def from_config(cls: type[Cfg], config: dict[str, Any]) -> Cfg:
        """Create an instance from a config dict.

        For dataclass subclasses:
            - Only dataclass fields are accepted.
            - Missing values fall back to the dataclass defaults.
            - Extra keys are ignored.
            - Keys starting with '_' are ignored.

        For non-dataclass usage:
            - Behaves like `cls(config)`.

        Args:
            config: Source configuration mapping.

        Returns:
            A configuration instance of `cls`.

        """
        if not is_dataclass(cls):
            return cls(config)

        # Keep only known dataclass fields.
        allowed = {f.name for f in fields(cls) if not f.name.startswith("_")}

        filtered = {k: v for k, v in config.items() if k in allowed}

        # Instantiating the dataclass will apply defaults automatically.
        return cls(**filtered)  # type: ignore[misc]

    def to_yaml(self, file_path: str) -> None:
        """Save configuration to a YAML file.

        Args:
            file_path: Destination YAML file path.

        """
        yaml = YAML(typ="safe", pure=True)
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(dict(self), f)

    def to_json(self, file_path: str, *, indent: int = 2) -> None:
        """Save configuration to a JSON file.

        Args:
            file_path: Destination JSON file path.
            indent: JSON indentation level.

        """
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(dict(self), f, indent=indent)


def load_configuration(file_path: str) -> PrettyConfig:
    """Load YAML configuration from file and return as PrettyConfig.

    Args:
        file_path: Path to the YAML configuration file.

    Returns:
        PrettyConfig: A PrettyConfig object containing the loaded configuration.

    """
    yaml = YAML(typ="safe", pure=True)

    with open(file_path, encoding="utf-8") as f:
        data = yaml.load(f) or {}
        # Wrap the loaded dict in our PrettyConfig
        return PrettyConfig(data)


def save_configuration(config: PrettyConfig, file_path: str) -> None:
    """Dump PrettyConfig to a YAML file.

    Args:
        config: The configuration to dump.
        file_path: Path to the output YAML file.

    """
    yaml = YAML(typ="safe", pure=True)
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(dict(config), f)
