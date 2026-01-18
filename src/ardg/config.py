"""Configuration loading and validation."""

from __future__ import annotations

from typing import Any, Dict

DEFAULT_CONFIG_PATH = "configs/default.yaml"
REQUIRED_TOP_LEVEL_KEYS = (
    "experiment",
    "dataset",
    "model",
    "train",
    "attack",
    "logging",
)


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML config file.

    Args:
        path: Path to a YAML configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config path does not exist.
        ImportError: If PyYAML is not installed.
        ValueError: If the loaded config is invalid.
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load configs.") from exc

    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

    if not isinstance(cfg, dict):
        raise ValueError("Config must be a mapping.")

    validate_config(cfg)
    return cfg


def validate_config(cfg: Dict[str, Any]) -> None:
    """Validate required configuration keys.

    Args:
        cfg: Configuration dictionary to validate.

    Raises:
        ValueError: If required keys are missing.
    """
    missing = [key for key in REQUIRED_TOP_LEVEL_KEYS if key not in cfg]
    if missing:
        raise ValueError(f"Missing config keys: {', '.join(missing)}")


def merge_overrides(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Merge override values into a base config.

    Args:
        cfg: Base configuration dictionary.
        overrides: Override dictionary to merge into cfg.

    Returns:
        A new configuration dictionary with overrides applied.
    """
    merged = dict(cfg)
    _deep_update(merged, overrides)
    return merged


def _deep_update(target: Dict[str, Any], updates: Dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
