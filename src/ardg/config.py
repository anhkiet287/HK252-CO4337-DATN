"""Configuration loading and validation."""

from __future__ import annotations

from typing import Any, Dict

DEFAULT_CONFIG_PATH = "configs/default.yaml"
def load_config(path: str) -> Dict[str, Any]:
    """Load and validate a YAML config file.

    Args:
        path: Path to a YAML configuration file.

    Returns:
        Parsed and validated configuration dictionary.

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
    return cfg


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
