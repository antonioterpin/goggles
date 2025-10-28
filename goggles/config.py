"""Utilities for loading and pretty-printing configuration files."""

from ruamel.yaml import YAML
from rich.console import Console
from rich.pretty import Pretty
from yaml.representer import SafeRepresenter


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


def represent_prettyconfig(dumper, data):
    """Represent PrettyConfig as a YAML mapping."""
    return dumper.represent_mapping("tag:yaml.org,2002:map", dict(data))


SafeRepresenter.add_representer(PrettyConfig, represent_prettyconfig)


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


# TODO: to fix, maybe put inside the PrettyConfig class?
def _write_json(path: Path, data: Dict[str, Any]) -> None:
    """Atomically write JSON data to disk.

    Args:
        path: The file path to write the JSON data to.
        data: The JSON data to write.

    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
    tmp_path.replace(path)
