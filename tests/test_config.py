"""Tests for PrettyConfig and a typed config that inherits it."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from goggles.config import PrettyConfig, load_configuration


@dataclass(frozen=True)
class DummyTypedConfig(PrettyConfig):
    """Example typed config that inherits PrettyConfig.

    Attributes:
        name: Logical config name.
        port: Serial port path.
        baud_rate: Supported serial speed.
    """

    name: str = "goggles.dummy"
    port: str = "/dev/ttyUSB0"
    baud_rate: int = 115200
    _secret: str = "dont-serialize"

    def __post_init__(self) -> None:
        """Validate baud_rate after initialization.

        Raises:
            ValueError: If `baud_rate` is not the supported value.
        """
        if self.baud_rate != 115200:
            raise ValueError("baud_rate must be 115200.")


def test_typed_config_defaults_and_post_init_dict_shape() -> None:
    """Build typed config and validate defaults + post-init dict sync."""
    cfg = DummyTypedConfig()

    assert cfg.name == "goggles.dummy", "Default name should apply."
    assert cfg.port == "/dev/ttyUSB0", "Default port should apply."
    assert cfg.baud_rate == 115200, "Default baud_rate should apply."
    assert cfg._secret == "dont-serialize", (
        "Private field default should apply."
    )
    assert "_secret" not in dict(cfg), (
        "Private field should not be in the dict."
    )

    # Dict payload is exactly the fields (post-init mirrored)
    assert dict(cfg) == {
        "name": "goggles.dummy",
        "port": "/dev/ttyUSB0",
        "baud_rate": 115200,
    }, "Dict payload should match dataclass fields."


def test_from_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError) as exc_info:
        DummyTypedConfig.from_config(
            {
                "name": "custom.name",
                "port": "/dev/ttyUSB9",
                "extra_key": 123,
                "another": "ignored",
            }
        )
    msg = str(exc_info.value)
    assert "Unknown config keys" in msg
    assert "DummyTypedConfig" in msg
    assert "extra_key" in msg
    assert "another" in msg


def test_to_dict_and_from_config_roundtrip_same() -> None:
    """to_dict + from_config returns an equivalent config."""
    cfg1 = DummyTypedConfig.from_config(
        {"name": "a", "port": "b", "baud_rate": 115200}
    )

    d = cfg1.to_dict()
    cfg2 = DummyTypedConfig.from_config(d)

    assert cfg2.to_dict() == cfg1.to_dict(), (
        "to_dict should produce equivalent dicts."
    )
    assert dict(cfg2) == dict(cfg1), (
        "dict() conversion should produce equivalent dicts."
    )


def test_yaml_and_json_roundtrip_via_prettyconfig_loaders(
    tmp_path: Path,
) -> None:
    """Save to YAML/JSON and reconstruct the typed config back.

    Args:
        tmp_path: Temporary directory for serialized config artifacts.
    """
    cfg = DummyTypedConfig.from_config(
        {
            "name": "n",
            "port": "/dev/ttyUSB0",
            "baud_rate": 115200,
        }
    )

    # --- YAML roundtrip (save via method, load via load_configuration) ---
    yaml_path = tmp_path / "cfg.yaml"
    cfg.to_yaml(str(yaml_path))

    loaded_yaml = load_configuration(str(yaml_path))  # returns PrettyConfig
    cfg_from_yaml = DummyTypedConfig.from_config(dict(loaded_yaml))

    assert cfg_from_yaml.to_dict() == cfg.to_dict(), (
        "YAML roundtrip should preserve config."
    )

    # --- JSON roundtrip (save via method, load via json.load) ---
    json_path = tmp_path / "cfg.json"
    cfg.to_json(str(json_path))

    with open(json_path, encoding="utf-8") as f:
        loaded_json = json.load(f)

    cfg_from_json = DummyTypedConfig.from_config(loaded_json)
    assert cfg_from_json.to_dict() == cfg.to_dict(), (
        "JSON roundtrip should preserve config."
    )


def test_private_fields_cannot_be_overridden_by_from_config() -> None:
    """Attempting to set a `_`-prefixed declared field raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        DummyTypedConfig.from_config(
            {
                "_secret": "hacked",
                "name": "x",
                "port": "y",
            }
        )
    msg = str(exc_info.value)
    assert "Private fields cannot be overridden" in msg
    assert "_secret" in msg


def test_from_config_reports_both_categories_distinctly() -> None:
    """Undeclared `_`-prefixed keys are unknown-keys, not private-field hits."""
    with pytest.raises(ValueError) as exc_info:
        DummyTypedConfig.from_config({"_not_declared": 1})
    msg = str(exc_info.value)
    assert "Unknown config keys" in msg
    assert "_not_declared" in msg


def test_private_fields_not_serialized_yaml_json_roundtrip(
    tmp_path: Path,
) -> None:
    """Private fields stay excluded from YAML/JSON after reload.

    Args:
        tmp_path: Temporary directory for serialized config artifacts.
    """
    cfg = DummyTypedConfig()

    yaml_path = tmp_path / "cfg.yaml"
    cfg.to_yaml(str(yaml_path))
    loaded_yaml = load_configuration(str(yaml_path))
    cfg_from_yaml = DummyTypedConfig.from_config(dict(loaded_yaml))
    assert cfg_from_yaml._secret == "dont-serialize", (
        "YAML output should not contain private fields."
    )
    assert "_secret" not in loaded_yaml, (
        "YAML output should not contain private fields."
    )

    json_path = tmp_path / "cfg.json"
    cfg.to_json(str(json_path))
    with open(json_path, encoding="utf-8") as f:
        loaded_json = json.load(f)
    cfg_from_json = DummyTypedConfig.from_config(loaded_json)
    assert cfg_from_json._secret == "dont-serialize", (
        "JSON output should not contain private fields."
    )
    assert "_secret" not in loaded_json, (
        "JSON output should not contain private fields."
    )
