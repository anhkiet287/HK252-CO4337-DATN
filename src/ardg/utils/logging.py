"""Logging helpers for console and wandb."""

import logging
from typing import Any, Dict, Optional


def setup_logging(level: int = logging.INFO, name: Optional[str] = None) -> logging.Logger:
    """Initialize a standard console logger.

    Args:
        level: Logging level.
        name: Logger name.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def init_wandb(cfg: Dict[str, Any]) -> Optional[Any]:
    """Initialize a Weights & Biases run.

    Args:
        cfg: Configuration dictionary with logging.wandb settings.

    Returns:
        Active wandb run.

    Raises:
        ImportError: When wandb is enabled but not installed.
    """
    wandb_cfg = cfg.get("logging", {}).get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return None

    try:
        import wandb  # type: ignore
    except ImportError as exc:
        raise ImportError("wandb is enabled but not installed.") from exc

    run_id = wandb_cfg.get("run_id") or None
    resume = wandb_cfg.get("resume") or None
    if run_id and resume is None:
        resume = "allow"

    run = wandb.init(
        project=wandb_cfg.get("project"),
        entity=wandb_cfg.get("entity") or None,
        name=cfg.get("logging", {}).get("run_name"),
        config=cfg,
        id=run_id,
        resume=resume,
    )
    if run is not None:
        try:
            run.define_metric("epoch")
            run.define_metric("epoch/*", step_metric="epoch")
        except Exception:
            pass
    return run


def log_metrics(
    logger: logging.Logger,
    metrics: Dict[str, Any],
    step: int,
    split: str,
    *,
    epoch: Optional[int] = None,
    log_by_epoch: bool = False,
) -> None:
    """Log metrics to console and wandb.

    Args:
        logger: Logger instance.
        metrics: Metric values such as loss, acc_clean, acc_pgd.
        step: Global step or epoch index.
        split: Split name (train/val/test).
        epoch: Optional epoch index for additional epoch-based W&B logging.
        log_by_epoch: Whether to also emit ``epoch/...`` metrics tied to ``epoch``.

    Side effects:
        Logs metrics to the logger and to wandb if a run is active.
    """
    prefixed = {f"{split}/{key}": value for key, value in metrics.items()}
    logger.info("step=%s metrics=%s", step, prefixed)

    try:
        import wandb  # type: ignore
    except ImportError:
        return

    if getattr(wandb, "run", None) is not None:
        payload: Dict[str, Any] = dict(prefixed)
        if log_by_epoch and epoch is not None:
            payload["epoch"] = int(epoch)
            payload.update({f"epoch/{split}/{key}": value for key, value in metrics.items()})
        wandb.log(payload, step=step)
