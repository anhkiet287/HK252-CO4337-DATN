"""Logging helpers for console and wandb."""

import logging
import math
import numbers
from pathlib import Path
from typing import Any, Dict, Optional

from ardg.utils.run_metadata import build_wandb_group, build_wandb_tags


def setup_logging(
    level: int = logging.INFO,
    name: Optional[str] = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Initialize a standard console logger.

    Args:
        level: Logging level.
        name: Logger name.
        log_file: Optional file path for persistent logs.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.propagate = False

    if not any(
        isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)
        for handler in logger.handlers
    ):
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if log_file:
        resolved_log = str(Path(log_file).resolve())
        has_file_handler = any(
            isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", "") == resolved_log
            for handler in logger.handlers
        )
        if not has_file_handler:
            file_handler = logging.FileHandler(resolved_log, encoding="utf-8")
            file_handler.setFormatter(
                logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
            )
            logger.addHandler(file_handler)
    logger.setLevel(level)
    return logger


def init_wandb(
    cfg: Dict[str, Any],
    *,
    metadata: Optional[Dict[str, Any]] = None,
    require_enabled: bool = False,
) -> Optional[Any]:
    """Initialize a Weights & Biases run.

    Args:
        cfg: Configuration dictionary with logging.wandb settings.
        metadata: Run metadata to attach to the run.
        require_enabled: Fail when W&B is required but disabled.

    Returns:
        Active wandb run.

    Raises:
        ImportError: When wandb is enabled but not installed.
        ValueError: When W&B is required but disabled or misconfigured.
    """
    wandb_cfg = cfg.get("logging", {}).get("wandb", {})
    enabled = bool(wandb_cfg.get("enabled", False))
    if require_enabled and not enabled:
        raise ValueError(
            "W&B logging is mandatory for canonical train/eval runs. "
            "Set logging.wandb.enabled=true and choose logging.wandb.mode=online|offline."
        )
    if not enabled:
        return None

    mode = str(wandb_cfg.get("mode", "online") or "online").strip().lower()
    if mode not in {"online", "offline"}:
        raise ValueError(f"Unsupported logging.wandb.mode={mode!r}. Use 'online' or 'offline'.")

    try:
        import wandb  # type: ignore
    except ImportError as exc:
        raise ImportError("wandb is enabled but not installed.") from exc

    run_id = wandb_cfg.get("run_id") or None
    resume = wandb_cfg.get("resume") or None
    if run_id and resume is None:
        resume = "allow"

    resolved_metadata = metadata or {}
    tags = list(wandb_cfg.get("tags", []) or [])
    if bool(wandb_cfg.get("add_default_tags", False)):
        for tag in build_wandb_tags(resolved_metadata):
            if tag not in tags:
                tags.append(tag)

    group = None
    group_template = str(wandb_cfg.get("group_template", "") or "").strip()
    if group_template:
        group = build_wandb_group(group_template, resolved_metadata)

    try:
        run = wandb.init(
            project=wandb_cfg.get("project"),
            entity=wandb_cfg.get("entity") or None,
            name=cfg.get("logging", {}).get("run_name"),
            mode=mode,
            config=cfg,
            id=run_id,
            resume=resume,
            tags=tags or None,
            group=group or None,
            dir=cfg.get("_meta", {}).get("run_dir") or cfg.get("logging", {}).get("output_dir"),
        )
    except Exception as exc:
        raise RuntimeError(_format_wandb_init_error(exc, mode)) from exc
    if run is not None:
        try:
            run.define_metric("epoch")
            run.define_metric("epoch/*", step_metric="epoch")
        except Exception:
            pass
        try:
            run.summary.update({k: v for k, v in resolved_metadata.items() if v is not None})
        except Exception:
            pass
    return run


def _format_wandb_init_error(exc: Exception, mode: str) -> str:
    message = str(exc)
    if mode == "online" and ("not logged in" in message.lower() or "401" in message):
        return (
            "W&B online initialization failed because the current environment is not logged in. "
            "Run `wandb login` before using `configs/profiles/local_gpu.yaml`, `configs/profiles/colab_gpu.yaml`, "
            "or `configs/profiles/h100.yaml`. For local smoke/debug only, use "
            "`configs/profiles/dev_fast.yaml` or set `logging.wandb.mode: offline`."
        )
    return f"W&B initialization failed in mode={mode!r}: {message}"


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
    logger.info("step=%s metrics=%s", step, _format_metrics_for_console(prefixed))

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


def _format_metrics_for_console(metrics: Dict[str, Any], precision: int = 3) -> str:
    """Render metrics with stable key order and fixed decimal places for numerics."""
    parts = [f"{key}={_format_metric_value(value, precision)}" for key, value in sorted(metrics.items())]
    return "{" + ", ".join(parts) + "}"


def _format_metric_value(value: Any, precision: int) -> str:
    """Format one metric value for aligned console logging."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, numbers.Real):
        number = float(value)
        if math.isfinite(number):
            return f"{number:.{precision}f}"
        return str(number)
    return str(value)
