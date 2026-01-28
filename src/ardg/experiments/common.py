"""Shared experiment wiring utilities."""

from typing import Any, Dict, Optional, Tuple

import torch

from ardg.config import load_config
from ardg.utils.platform import resolve_platform
from ardg.data.datasets import get_dataloaders
from ardg.evaluation.evaluator import evaluate_clean
from ardg.models.factory import build_model
from ardg.utils.logging import init_wandb, log_metrics, setup_logging
from ardg.utils.seed import set_seed


def setup_run(cfg_path: str, run_name_suffix: Optional[str] = None) -> Tuple[Dict[str, Any], Any, Any, str]:
    """Load config and initialize run essentials.

    Args:
        cfg_path: Path to the YAML config file.
        run_name_suffix: Optional suffix to append to logging.run_name (e.g., "eval").

    Returns:
        Tuple of (cfg, logger, wandb_run, device).
    """
    cfg = load_config(cfg_path)
    platform = resolve_platform(cfg.get("experiment", {}).get("platform"))
    cfg.setdefault("experiment", {})["platform"] = platform
    deterministic = cfg.get("experiment", {}).get("deterministic", True)
    set_seed(cfg["experiment"]["seed"], deterministic=deterministic)

    if run_name_suffix:
        log_cfg = cfg.setdefault("logging", {}).setdefault("run_name", "")
        if log_cfg:
            cfg["logging"]["run_name"] = f"{log_cfg}_{run_name_suffix}"
        else:
            cfg["logging"]["run_name"] = run_name_suffix

    logger = setup_logging(name=cfg.get("logging", {}).get("run_name"))
    run = init_wandb(cfg)
    device = cfg["experiment"].get("device", "cpu")
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
        metrics = evaluate_clean(model, loader, device)
        metrics["device"] = str(device)
        log_metrics(logger, metrics, step=eval_step, split=f"{split_name}_clean")
        eval_step += 1
