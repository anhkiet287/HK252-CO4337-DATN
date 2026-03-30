"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

DEFAULT_CONFIG_PATH = "configs/default.yaml"


def load_config(path: str, extra_paths: Iterable[str] | None = None) -> Dict[str, Any]:
    """Load a config tree and merge optional overlay configs."""
    resolved_cfg, sources = _load_config_tree(Path(path).resolve(), stack=[])
    merged = resolved_cfg

    for extra_path in extra_paths or ():
        overlay_cfg, overlay_sources = _load_config_tree(Path(extra_path).resolve(), stack=[])
        merged = merge_overrides(merged, overlay_cfg)
        sources.extend(overlay_sources)

    merged.setdefault("_meta", {})["config_sources"] = _dedupe_preserve_order(sources)
    return merged


def merge_overrides(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Merge override values into a base config."""
    merged = dict(cfg)
    _deep_update(merged, overrides)
    return merged


def _deep_update(target: Dict[str, Any], updates: Dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


def _load_config_tree(path: Path, stack: List[Path]) -> Tuple[Dict[str, Any], List[str]]:
    if path in stack:
        cycle = " -> ".join(str(item) for item in [*stack, path])
        raise ValueError(f"Config include cycle detected: {cycle}")
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw_cfg = _read_yaml_mapping(path)
    base_value = raw_cfg.pop("_base_", raw_cfg.pop("defaults", None))

    merged: Dict[str, Any] = {}
    sources: List[str] = []
    for base_path in _resolve_base_paths(path, base_value):
        base_cfg, base_sources = _load_config_tree(base_path, stack=[*stack, path])
        merged = merge_overrides(merged, base_cfg)
        sources.extend(base_sources)

    merged = merge_overrides(merged, raw_cfg)
    sources.append(str(path))
    return merged, sources


def _read_yaml_mapping(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load configs.") from exc

    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {path} must be a mapping.")
    return cfg


def _resolve_base_paths(path: Path, base_value: Any) -> List[Path]:
    if base_value is None:
        return []
    if isinstance(base_value, str):
        base_entries = [base_value]
    elif isinstance(base_value, list):
        base_entries = [str(item) for item in base_value]
    else:
        raise ValueError(f"Unsupported _base_ value in {path}: {base_value!r}")

    base_paths: List[Path] = []
    for entry in base_entries:
        candidate = Path(entry)
        if not candidate.is_absolute():
            candidate = (path.parent / candidate).resolve()
        base_paths.append(candidate)
    return base_paths


def _dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered
