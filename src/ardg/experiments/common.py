"""Shared experiment wiring utilities."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from ardg.config import load_config
from ardg.data.datasets import get_dataloaders
from ardg.evaluation.evaluator import evaluate_clean
from ardg.models.factory import build_model
from ardg.utils.logging import init_wandb, log_metrics, setup_logging
from ardg.utils.artifacts import update_artifact_manifest
from ardg.utils.paths import (
    default_output_dir,
    ensure_dir,
    get_run_dir,
    get_wandb_run_id_path,
    get_wandb_run_url_path,
    get_legacy_wandb_run_id_path,
    get_legacy_wandb_run_url_path,
    initialize_run_layout,
    platform_workspace_root,
)
from ardg.utils.platform import resolve_platform
from ardg.utils.run_metadata import (
    build_run_metadata,
    derive_run_name,
    write_resolved_config_snapshot,
    write_run_manifest,
)
from ardg.utils.seed import set_seed


def _resolve_device(requested: str, logger: Any) -> str:
    """Resolve runtime device; try CUDA first, fallback to CPU on failure."""
    device = str(requested or "cpu")
    if device.startswith("cuda"):
        try:
            _ = torch.empty(1, device=torch.device(device))
            return device
        except Exception as exc:
            logger.warning(
                "Failed to initialize %s (%s). Falling back to cpu.",
                device,
                exc,
            )
            return "cpu"
    return device


def load_runtime_config(
    cfg_path: str,
    profile_path: Optional[str] = None,
    platform_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Load config and apply runtime-only platform path overrides."""
    extra_paths = [profile_path] if profile_path else None
    cfg = load_config(cfg_path, extra_paths=extra_paths)
    platform = resolve_platform(platform_override or cfg.get("experiment", {}).get("platform"))
    cfg.setdefault("experiment", {})["platform"] = platform
    runtime_cfg = cfg.setdefault("runtime", {})
    if profile_path:
        runtime_cfg["profile_path"] = str(Path(profile_path).resolve())
    runtime_cfg.setdefault(
        "profile",
        cfg.get("experiment", {}).get("runtime_profile", runtime_cfg.get("profile", "local_gpu")),
    )
    cfg.setdefault("experiment", {})["runtime_profile"] = str(runtime_cfg.get("profile", "local_gpu"))

    logging_cfg = cfg.setdefault("logging", {})
    if platform_override:
        workspace_root = platform_workspace_root(platform)
        cfg.setdefault("dataset", {})["data_dir"] = str(workspace_root / "data")
        logging_cfg["output_dir"] = str(workspace_root / "outputs")
    elif str(logging_cfg.get("output_dir", "")).strip() == "":
        logging_cfg["output_dir"] = default_output_dir(cfg)
    logging_cfg["run_name"] = derive_run_name(cfg)
    return cfg


def setup_run(
    cfg_path: str,
    *,
    stage: str = "train",
    run_name_suffix: Optional[str] = None,
    wandb_run_id: Optional[str] = None,
    wandb_resume: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    profile_path: Optional[str] = None,
    require_wandb: bool = False,
    platform_override: Optional[str] = None,
) -> Tuple[Dict[str, Any], Any, Any, str]:
    """Load config and initialize run essentials.

    Args:
        cfg_path: Path to the YAML config file.
        stage: Run stage such as train/eval/preflight.
        run_name_suffix: Optional suffix to append to logging.run_name (e.g., "eval").
        wandb_run_id: Optional W&B run id to resume.
        wandb_resume: Optional W&B resume mode ("allow"/"must"/"never").
        checkpoint_path: Optional checkpoint path attached to run metadata.
        profile_path: Optional runtime profile overlay path.
        require_wandb: Whether W&B must be enabled for this run.
        platform_override: Optional runtime platform override for data/output roots.

    Returns:
        Tuple of (cfg, logger, wandb_run, device).
    """
    cfg = load_runtime_config(
        cfg_path,
        profile_path=profile_path,
        platform_override=platform_override,
    )
    deterministic = cfg.get("experiment", {}).get("deterministic", True)
    set_seed(cfg["experiment"]["seed"], deterministic=deterministic)
    cfg.setdefault("logging", {})["run_name"] = derive_run_name(cfg, suffix=run_name_suffix)

    run_dir = Path(get_run_dir(cfg))
    ensure_dir(str(run_dir))
    layout = initialize_run_layout(str(run_dir))
    log_path = Path(layout["logs_dir"]) / f"{stage}.log"
    logger = setup_logging(name=cfg.get("logging", {}).get("run_name"), log_file=str(log_path))
    device = _resolve_device(cfg.get("experiment", {}).get("device", "cpu"), logger)
    cfg.setdefault("experiment", {})["device"] = device
    cfg.setdefault("_meta", {})["run_dir"] = str(run_dir)
    cfg["_meta"]["stage"] = str(stage)

    config_sources = cfg.get("_meta", {}).get("config_sources", [str(Path(cfg_path).resolve())])
    metadata = build_run_metadata(
        cfg,
        stage=stage,
        run_dir=str(run_dir),
        config_sources=config_sources,
        checkpoint_path=checkpoint_path,
    )
    cfg["_meta"]["run_metadata"] = metadata
    cfg["_meta"]["resolved_config_path"] = write_resolved_config_snapshot(str(run_dir), cfg)

    wandb_cfg = cfg.setdefault("logging", {}).setdefault("wandb", {})
    if wandb_run_id:
        wandb_cfg["run_id"] = str(wandb_run_id)
    if wandb_resume:
        wandb_cfg["resume"] = str(wandb_resume)
    run = init_wandb(cfg, metadata=metadata, require_enabled=require_wandb)
    if run is not None and getattr(run, "id", None):
        wandb_cfg["run_id"] = str(run.id)
        get_wandb_run_id_path(str(run_dir)).write_text(str(run.id), encoding="utf-8")
        get_legacy_wandb_run_id_path(str(run_dir)).write_text(str(run.id), encoding="utf-8")
        metadata["wandb_run_id"] = str(run.id)
        if getattr(run, "url", None):
            metadata["wandb_url"] = str(run.url)
            get_wandb_run_url_path(str(run_dir)).write_text(str(run.url), encoding="utf-8")
            get_legacy_wandb_run_url_path(str(run_dir)).write_text(str(run.url), encoding="utf-8")

    manifest = {
        "stage": stage,
        "cfg_path": str(Path(cfg_path).resolve()),
        "profile_path": str(Path(profile_path).resolve()) if profile_path else None,
        "platform_override": platform_override,
        "resolved_config_path": cfg["_meta"]["resolved_config_path"],
        "run_log_path": str(log_path),
        "paths": layout,
        "run_metadata": metadata,
        "wandb": {
            "enabled": bool(wandb_cfg.get("enabled", False)),
            "mode": str(wandb_cfg.get("mode", "online") or "online"),
            "run_id": wandb_cfg.get("run_id"),
        },
    }
    cfg["_meta"]["run_manifest_path"] = write_run_manifest(str(run_dir), manifest)
    update_artifact_manifest(
        {
            "latest": {
                str(metadata.get("backbone") or "model"): {
                    stage: {
                        "run_name": metadata.get("run_name"),
                        "run_dir": str(run_dir),
                        "resolved_config": cfg["_meta"]["resolved_config_path"],
                        "run_manifest": cfg["_meta"]["run_manifest_path"],
                        "config_path": str(Path(cfg_path).resolve()),
                        "profile_path": str(Path(profile_path).resolve()) if profile_path else None,
                        "wandb_run_id": metadata.get("wandb_run_id"),
                        "wandb_url": metadata.get("wandb_url"),
                    }
                }
            }
        }
    )

    logger.info(
        "Initialized run name=%s stage=%s backbone=%s dataset=%s mode=%s profile=%s seed=%s device=%s precision=%s run_dir=%s",
        metadata.get("run_name"),
        metadata.get("stage"),
        metadata.get("backbone"),
        metadata.get("dataset"),
        metadata.get("train_mode"),
        metadata.get("runtime_profile"),
        metadata.get("seed"),
        metadata.get("device"),
        metadata.get("precision"),
        metadata.get("run_dir"),
    )
    if metadata.get("git_commit"):
        logger.info("Source commit=%s", metadata["git_commit"])
    if metadata.get("wandb_run_id"):
        logger.info(
            "W&B attached run_id=%s url=%s",
            metadata.get("wandb_run_id"),
            metadata.get("wandb_url"),
        )
    return cfg, logger, run, device


def build_loaders(cfg: Dict[str, Any]) -> Tuple[Any, Any, Any]:
    """Build train/val/test data loaders from config.

    Args:
        cfg: Configuration dictionary.

    Returns:
        Tuple of train, val, and test loaders.
    """
    return get_dataloaders(cfg)


def load_model_from_checkpoint(cfg: Dict[str, Any], ckpt_path: str, device: str) -> Any:
    """Build a model and load weights from a checkpoint.

    Args:
        cfg: Configuration dictionary.
        ckpt_path: Path to the checkpoint file.
        device: Device string such as "cpu" or "cuda".

    Returns:
        Model loaded with checkpoint weights.

    Raises:
        ValueError: If the checkpoint does not contain a state dict.
    """
    model = build_model(cfg)
    checkpoint = torch.load(ckpt_path, map_location=torch.device(device))
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint at {ckpt_path} does not contain a valid state_dict.")
    model.load_state_dict(state_dict)
    model.to(device)
    return model


def run_clean_eval(
    model: Any,
    val_loader: Any,
    test_loader: Any,
    device: str,
    logger: Any,
    step: int,
    precision: Any | None = None,
) -> None:
    """Run clean evaluation on val/test splits and log metrics.

    Args:
        model: Trained model.
        val_loader: Validation data loader.
        test_loader: Test data loader.
        device: Device string such as "cpu" or "cuda".
        logger: Logger instance.
        step: Starting step for logging.

    Returns:
        None.
    """
    eval_step = step
    for split_name, loader in (("val", val_loader), ("test", test_loader)):
        metrics = evaluate_clean(model, loader, device, precision=precision)
        metrics["device"] = str(device)
        log_metrics(logger, metrics, step=eval_step, split=f"{split_name}_clean")
        eval_step += 1
