"""Utilities for loading and pretty-printing configuration files."""

from __future__ import annotations
import json
from ruamel.yaml import YAML
from rich.console import Console
from rich.pretty import Pretty
from yaml.representer import SafeRepresenter
from typing import Any, TypeVar
from dataclasses import fields, is_dataclass

Cfg = TypeVar("Cfg", bound="PrettyConfig")


class PrettyConfig(dict):
    """Dictionary subclass with pretty-printing and serialization methods."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Automatically wrap __post_init__ in subclasses to sync dict."""
        super().__init_subclass__(**kwargs)

        orig_post_init = cls.__dict__.get("__post_init__")
        if orig_post_init is None:
            return

        # If subclass defines __post_init__, wrap it so we always sync dict after it.
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

    def __str__(self):
        """Return a pretty-printed string of the configuration."""
        console = Console()
        plain = dict(self)
        with console.capture() as capture:
            console.print(Pretty(plain))
        return capture.get()

    __repr__ = __str__

    def to_dict(self) -> dict[str, Any]:
        """Return a plain Python dict."""
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
        """
        if not is_dataclass(cls):
            return cls(config)

        # Keep only known dataclass fields.
        allowed = {f.name for f in fields(cls) if not f.name.startswith("_")}

        filtered = {k: v for k, v in config.items() if k in allowed}

        # Instantiating the dataclass will apply defaults automatically.
        return cls(**filtered)  # type: ignore[misc]

    def to_yaml(self, file_path: str) -> None:
        """Save configuration to a YAML file."""
        yaml = YAML(typ="safe", pure=True)
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(dict(self), f)

    def to_json(self, file_path: str, *, indent: int = 2) -> None:
        """Save configuration to a JSON file."""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(dict(self), f, indent=indent)


def load_configuration(file_path: str) -> PrettyConfig:
    """Load YAML configuration from file and return as PrettyConfig.

    Args:
        file_path: Path to the YAML configuration file.

    Returns:
        PrettyConfig: A PrettyConfig object containing the loaded configuration.

    Raises:
        FileNotFoundError: If the specified file does not exist.

    """
    yaml = YAML(typ="safe", pure=True)

    with open(file_path, encoding="utf-8") as f:
        data = yaml.load(f) or {}
        # Wrap the loaded dict in our PrettyConfig
        return PrettyConfig(data)


def represent_prettyconfig(dumper, data):
    """Represent PrettyConfig as a YAML mapping.

    Args:
        dumper: The YAML dumper.
        data: The PrettyConfig instance.

    """
    return dumper.represent_mapping("tag:yaml.org,2002:map", dict(data))


SafeRepresenter.add_representer(PrettyConfig, represent_prettyconfig)


def save_configuration(config: PrettyConfig, file_path: str):
    """Dump PrettyConfig to a YAML file.

    Args:
        config: The configuration to dump.
        file_path: Path to the output YAML file.

    """
    yaml = YAML(typ="safe", pure=True)
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(dict(config), f)
