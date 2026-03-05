"""Tests for PrettyConfig and a typed config that inherits it."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from goggles.config import (
    PrettyConfig,
    apply_overrides,
    load_configuration,
    parse_value,
)


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


def test_from_config_ignores_unknown_keys() -> None:
    """Extra keys in input to from_config are ignored for typed configs."""
    cfg = DummyTypedConfig.from_config(
        {
            "name": "custom.name",
            "port": "/dev/ttyUSB9",
            "extra_key": 123,
            "another": "ignored",
        }
    )

    # Known fields set
    assert cfg.name == "custom.name", "Name should be set from config."
    assert cfg.port == "/dev/ttyUSB9", "Port should be set from config."
    assert cfg.baud_rate == 115200, "Baud rate should be set from config."
    assert cfg._secret == "dont-serialize", (
        "Private field should keep default value."
    )

    # Unknown keys are NOT stored in the dict
    assert "extra_key" not in cfg, "Extra keys should not be in the dict."
    assert "another" not in cfg, "Extra keys should not be in the dict."

    # Unknown keys are NOT attributes
    assert not hasattr(cfg, "extra_key"), "Extra keys should not be attributes."
    assert not hasattr(cfg, "another"), "Extra keys should not be attributes."


def test_to_dict_and_from_config_roundtrip_same() -> None:
    """to_dict + from_config returns an equivalent config."""
    cfg1 = DummyTypedConfig.from_config(
        {"name": "a", "port": "b", "baud_rate": 115200, "ignored": True}
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
            "junk": "nope",
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


def test_private_fields_are_not_overwritten_by_from_config() -> None:
    """Keys starting with '_' must be ignored by from_config."""
    cfg = DummyTypedConfig.from_config(
        {
            "_secret": "hacked",
            "name": "x",
            "port": "y",
        }
    )

    assert cfg.name == "x", "Public field must be set from config."
    assert cfg.port == "y", "Public field must be set from config."
    assert cfg._secret == "dont-serialize", (
        "Private field must keep default value."
    )
    assert "_secret" not in cfg, (
        "Private field must not appear in dict payload."
    )


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


# =========================================================
# CLI override utilities
# =========================================================


def test_parse_value_bool() -> None:
    """Boolean strings should convert to bool."""
    # Arrange
    raw_true = "true"
    raw_false = "false"

    # Act
    result_true = parse_value(raw_true)
    result_false = parse_value(raw_false)

    # Assert
    assert result_true is True
    assert result_false is False


def test_parse_value_none() -> None:
    """String 'none' should convert to None."""
    # Arrange
    raw = "none"

    # Act
    result = parse_value(raw)

    # Assert
    assert result is None


def test_parse_value_int() -> None:
    """Integer strings should convert to int."""
    # Arrange
    raw = "42"

    # Act
    result = parse_value(raw)

    # Assert
    assert result == 42
    assert isinstance(result, int)


def test_parse_value_float() -> None:
    """Float strings should convert to float."""
    # Arrange
    raw = "1e-3"

    # Act
    result = parse_value(raw)

    # Assert
    assert isinstance(result, float)
    assert result == pytest.approx(1e-3)


def test_parse_value_string() -> None:
    """Non-numeric strings should remain strings."""
    # Arrange
    raw = "experiment"

    # Act
    result = parse_value(raw)

    # Assert
    assert result == "experiment"


def test_apply_overrides() -> None:
    """Apply multiple overrides to a configuration."""
    # Arrange
    config = PrettyConfig(
        {
            "training": {
                "lr": 0.001,
                "epochs": 10,
            },
            "seed": 0,
        }
    )

    overrides = [
        "training.lr=0.02",
        "training.epochs=20",
        "seed=42",
    ]

    # Act
    result = apply_overrides(config, overrides)

    # Assert
    assert result["training"]["lr"] == pytest.approx(0.02)
    assert result["training"]["epochs"] == 20
    assert result["seed"] == 42


def test_override_invalid_format() -> None:
    """Overrides without '=' should raise ValueError."""
    # Arrange
    config = PrettyConfig({"training": {"lr": 0.001}})
    overrides = ["training.lr"]

    # Act / Assert
    with pytest.raises(ValueError):
        apply_overrides(config, overrides)


def test_override_invalid_key_path() -> None:
    """Overriding a missing key path should raise KeyError."""
    # Arrange
    config = PrettyConfig({"training": {"lr": 0.001}})
    overrides = ["model.lr=0.01"]

    # Act / Assert
    with pytest.raises(KeyError):
        apply_overrides(config, overrides)


def test_override_dict_node_forbidden() -> None:
    """Overwriting a dictionary node should raise KeyError."""
    # Arrange
    config = PrettyConfig({"training": {"lr": 0.001}})
    overrides = ["training=5"]

    # Act / Assert
    with pytest.raises(KeyError):
        apply_overrides(config, overrides)


def test_override_non_dict_intermediate() -> None:
    """Traversing through a non-dict node should raise KeyError."""
    # Arrange
    config = PrettyConfig({"training": {"lr": 0.001}})
    overrides = ["training.lr.value=1"]

    # Act / Assert
    with pytest.raises(KeyError):
        apply_overrides(config, overrides)


def test_override_type_parsing() -> None:
    """Overrides should parse values into correct Python types."""
    # Arrange
    config = PrettyConfig(
        {
            "training": {"lr": 0.001},
            "seed": 0,
            "use_gpu": False,
        }
    )

    overrides = [
        "training.lr=0.1",
        "seed=42",
        "use_gpu=true",
    ]

    # Act
    result = apply_overrides(config, overrides)

    # Assert
    assert isinstance(result["training"]["lr"], float)
    assert isinstance(result["seed"], int)
    assert isinstance(result["use_gpu"], bool)
