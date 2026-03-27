"""Smoke-test runner for quick train/eval cycles."""

from typing import Sequence

from ardg.experiments.common import build_loaders, load_model_from_checkpoint, run_clean_eval, setup_run
from ardg.models.factory import build_model
from ardg.training.trainer import Trainer


def run_smoke(cfg_paths: Sequence[str], profile_path: str | None = None) -> None:
    """Run smoke-test training and clean evaluation for each config.

    Args:
        cfg_paths: Iterable of config file paths.

    Returns:
        None.

    Raises:
        RuntimeError: If a smoke run fails to produce a checkpoint.
    """
    for cfg_path in cfg_paths:
        _run_single(cfg_path, profile_path=profile_path)


def _run_single(cfg_path: str, profile_path: str | None = None) -> None:
    """Run a single smoke-test train/eval cycle.

    Args:
        cfg_path: Path to the YAML config file.

    Returns:
        None.

    Raises:
        RuntimeError: If no checkpoint is produced during training.
    """
    cfg, logger, run, device = setup_run(
        cfg_path,
        stage="smoke",
        run_name_suffix="smoke",
        profile_path=profile_path,
        require_wandb=False,
    )
    train_loader, val_loader, test_loader = build_loaders(cfg)
    model = build_model(cfg)
    trainer = Trainer(
        cfg,
        model,
        train_loader,
        val_loader,
        device=device,
        logger=logger,
    )
    checkpoints = trainer.train()
    ckpt_path = checkpoints.get("best") or checkpoints.get("last")
    if not ckpt_path:
        raise RuntimeError("Smoke test did not produce a checkpoint.")
    model = load_model_from_checkpoint(cfg, ckpt_path, device)
    run_clean_eval(model, val_loader, test_loader, device, logger, trainer.global_step + 1)

    if run is not None:
        run.finish()
