"""Run metadata helpers for manifests, naming, and experiment tracing."""

from __future__ import annotations

import json
import os
import socket
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch

from ardg.config import merge_overrides
from ardg.utils.paths import project_root


def canonical_backbone_name(model_name: str) -> str:
    name = str(model_name or "").lower()
    aliases = {
        "resnet18_cifar": "resnet18",
        "resnet50_cifar": "resnet50",
        "vit_b16_cifar": "vit_b16",
        "vit_b16": "vit_b16",
        "vit_b_16": "vit_b16",
        "vit": "vit_b16",
    }
    return aliases.get(name, name or "model")


def canonical_profile_name(profile_name: str) -> str:
    name = str(profile_name or "local_gpu").strip().lower()
    aliases = {
        "dev_fast": "dev",
        "local_gpu": "local",
        "colab_gpu": "colab",
        "h100": "h100",
    }
    return aliases.get(name, name)


def get_runtime_profile(cfg: Dict[str, Any]) -> str:
    runtime_cfg = cfg.get("runtime", {})
    experiment_cfg = cfg.get("experiment", {})
    return str(
        runtime_cfg.get("profile")
        or experiment_cfg.get("runtime_profile")
        or "local_gpu"
    )


def derive_run_name(cfg: Dict[str, Any], suffix: Optional[str] = None) -> str:
    logging_cfg = cfg.get("logging", {})
    explicit = str(logging_cfg.get("run_name", "") or "").strip()
    if explicit:
        return _append_suffix(explicit, suffix)

    backbone = canonical_backbone_name(cfg.get("model", {}).get("name", "model"))
    profile = canonical_profile_name(get_runtime_profile(cfg))
    experiment_name = str(
        cfg.get("experiment", {}).get("name")
        or cfg.get("experiment", {}).get("track_name")
        or cfg.get("train", {}).get("mode")
        or "run"
    ).strip().lower().replace(" ", "_")
    seed = int(cfg.get("experiment", {}).get("seed", 42))
    base_name = f"{backbone}_{profile}_{experiment_name}_seed{seed}"
    return _append_suffix(base_name, suffix)


def build_run_metadata(
    cfg: Dict[str, Any],
    *,
    stage: str,
    run_dir: str,
    config_sources: Iterable[str],
    checkpoint_path: Optional[str] = None,
) -> Dict[str, Any]:
    train_cfg = cfg.get("train", {})
    experiment_cfg = cfg.get("experiment", {})
    logging_cfg = cfg.get("logging", {})
    device = str(experiment_cfg.get("device", "cpu"))

    return {
        "stage": str(stage),
        "run_name": str(logging_cfg.get("run_name", "")),
        "backbone": canonical_backbone_name(cfg.get("model", {}).get("name", "")),
        "dataset": str(cfg.get("dataset", {}).get("name", "")),
        "train_mode": str(train_cfg.get("mode", "")),
        "experiment_name": str(experiment_cfg.get("name", train_cfg.get("mode", ""))),
        "runtime_profile": str(get_runtime_profile(cfg)),
        "platform": str(experiment_cfg.get("platform", "")),
        "seed": int(experiment_cfg.get("seed", 42)),
        "deterministic": bool(experiment_cfg.get("deterministic", True)),
        "precision": str(experiment_cfg.get("precision", "fp32")),
        "device": device,
        "device_name": get_device_name(device),
        "hostname": socket.gethostname(),
        "machine": os.getenv("COMPUTERNAME") or os.getenv("HOSTNAME") or socket.gethostname(),
        "git_commit": get_git_commit(),
        "run_dir": str(run_dir),
        "output_dir": str(logging_cfg.get("output_dir", "")),
        "config_sources": list(config_sources),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "batch_size": _resolve_batch_size(cfg),
    }


def write_resolved_config_snapshot(run_dir: str, cfg: Dict[str, Any]) -> str:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to write resolved configs.") from exc

    path = Path(run_dir) / "resolved_config.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(cfg), handle, sort_keys=False)
    return str(path)


def write_run_manifest(run_dir: str, manifest: Dict[str, Any]) -> str:
    path = Path(run_dir) / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


def update_run_manifest(run_dir: str, updates: Dict[str, Any]) -> str:
    path = Path(run_dir) / "run_manifest.json"
    current: Dict[str, Any] = {}
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            current = loaded
    merged = merge_overrides(current, updates)
    path.write_text(json.dumps(merged, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


def get_git_commit() -> Optional[str]:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=project_root(),
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
            or None
        )
    except Exception:
        return None


def get_device_name(device: str) -> Optional[str]:
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return None
    try:
        return str(torch.cuda.get_device_name(torch.device(device)))
    except Exception:
        return None


def build_wandb_tags(metadata: Dict[str, Any]) -> list[str]:
    ordered = [
        metadata.get("backbone"),
        metadata.get("dataset"),
        metadata.get("train_mode"),
        metadata.get("runtime_profile"),
        metadata.get("stage"),
    ]
    tags: list[str] = []
    for item in ordered:
        value = str(item or "").strip()
        if value and value not in tags:
            tags.append(value)
    return tags


def build_wandb_group(group_template: str, metadata: Dict[str, Any]) -> str:
    context = {
        "model": metadata.get("backbone", ""),
        "backbone": metadata.get("backbone", ""),
        "stage": metadata.get("stage", ""),
        "dataset": metadata.get("dataset", ""),
        "profile": metadata.get("runtime_profile", ""),
        "runtime_profile": metadata.get("runtime_profile", ""),
        "experiment": metadata.get("experiment_name", ""),
        "train_mode": metadata.get("train_mode", ""),
    }
    return group_template.format(**context)


def _append_suffix(name: str, suffix: Optional[str]) -> str:
    clean_suffix = str(suffix or "").strip()
    if not clean_suffix:
        return name
    if name.endswith(f"_{clean_suffix}"):
        return name
    return f"{name}_{clean_suffix}"


def _resolve_batch_size(cfg: Dict[str, Any]) -> int:
    eval_cfg = cfg.get("eval", {})
    train_cfg = cfg.get("train", {})
    dataset_cfg = cfg.get("dataset", {})
    return int(
        train_cfg.get(
            "batch_size",
            eval_cfg.get("batch_size", dataset_cfg.get("batch_size", 0)),
        )
        or 0
    )
