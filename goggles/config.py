"""Utilities for loading and pretty-printing configuration files."""

from ruamel.yaml import YAML
from rich.console import Console
from rich.pretty import Pretty


class PrettyConfig(dict):
    """Dictionary subclass with pretty-printing using ruamel.yaml."""

    def __str__(self):
        """Return a pretty-printed string representation of the configuration."""
        console = Console()
        plain = dict(self)
        with console.capture() as capture:
            console.print(Pretty(plain))
        return capture.get()

    __repr__ = __str__


def load_configuration(file_path: str) -> PrettyConfig:
    """Load YAML configuration from file and return as PrettyConfig.

    Args:
        file_path (str): Path to the YAML configuration file.

    Returns:
        PrettyConfig: A PrettyConfig object containing the loaded configuration.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    yaml = YAML(typ="safe", pure=True)

    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.load(f) or {}
        # Wrap the loaded dict in our PrettyConfig
        return PrettyConfig(data)
