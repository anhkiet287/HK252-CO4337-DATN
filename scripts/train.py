"""Train a model with the ARDG pipeline.

Loads config, sets seeds, builds data/model/trainer components, and writes
checkpoints and run summaries to outputs/.
"""

import argparse
from pathlib import Path
from typing import Optional

import torch

from ardg.config import DEFAULT_CONFIG_PATH
from ardg.experiments.common import build_loaders, load_runtime_config, setup_run
from ardg.models.factory import build_model
from ardg.training.trainer import Trainer
from ardg.utils.logging import validate_wandb_policy
from ardg.utils.paths import find_checkpoint, get_run_dir, get_run_root_from_artifact, get_wandb_run_id_path
from ardg.utils.run_metadata import update_run_manifest


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to YAML config.")
    parser.add_argument(
        "--profile",
        default=None,
        help="Optional runtime profile overlay YAML (e.g. configs/profiles/local_gpu.yaml).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint in the run directory (or from --checkpoint).",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path to resume from. Defaults to <run_dir>/last.pt when --resume is set.",
    )
    parser.add_argument(
        "--wandb_run_id",
        default=None,
        help="Optional W&B run id to resume into (useful for legacy checkpoints).",
    )
    parser.add_argument(
        "--platform",
        choices=("local", "colab"),
        default=None,
        help=(
            "Legacy platform override for runtime roots. "
            "Prefer --profile for thesis workflows."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print resolved config summary before training starts.",
    )
    return parser.parse_args()


def _resolve_resume_checkpoint(
    cfg_path: str,
    profile_path: Optional[str],
    resume: bool,
    checkpoint: Optional[str],
    platform_override: Optional[str],
) -> Optional[str]:
    if checkpoint:
        return checkpoint
    if not resume:
        return None

    cfg = load_runtime_config(
        cfg_path,
        profile_path=profile_path,
        platform_override=platform_override,
    )
    run_dir = Path(get_run_dir(cfg))
    candidate = find_checkpoint(str(run_dir), names=("last", "best"))
    if candidate:
        return candidate
    raise FileNotFoundError(
        f"Resume requested but no checkpoint found in {run_dir}. "
        "Expected checkpoints/last.pt or checkpoints/best.pt."
    )


def _extract_wandb_run_id(ckpt_path: Optional[str]) -> Optional[str]:
    if not ckpt_path:
        return None
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
    except Exception:
        checkpoint = None
    if isinstance(checkpoint, dict):
        run_id = checkpoint.get("wandb_run_id")
        if run_id:
            return str(run_id)

    run_dir = get_run_root_from_artifact(ckpt_path)
    run_id_file = get_wandb_run_id_path(str(run_dir))
    if not run_id_file.exists():
        run_id_file = Path(ckpt_path).parent / "wandb_run_id.txt"
    if run_id_file.exists():
        value = run_id_file.read_text(encoding="utf-8").strip()
        if value:
            return value
    return None


def main() -> None:
    """Run end-to-end training.

    Args:
        None.

    Returns:
        None.

    Side effects:
        Writes outputs/checkpoints and may initialize wandb logging.
    """
    # Parsing arguments and loading config.
    args = parse_args()
    resume_ckpt = _resolve_resume_checkpoint(
        args.config,
        args.profile,
        args.resume,
        args.checkpoint,
        args.platform,
    )
    preview_cfg = load_runtime_config(
        args.config,
        profile_path=args.profile,
        platform_override=args.platform,
    )
    validate_wandb_policy(preview_cfg, require_enabled=True)
    resume_run_id = args.wandb_run_id or _extract_wandb_run_id(resume_ckpt)
    resume_mode = "must" if resume_run_id else None
    cfg, logger, run, device = setup_run(
        args.config,
        stage="train",
        wandb_run_id=resume_run_id,
        wandb_resume=resume_mode,
        checkpoint_path=resume_ckpt,
        profile_path=args.profile,
        require_wandb=True,
        platform_override=args.platform,
    )
    if resume_ckpt and not resume_run_id:
        logger.warning(
            "Resuming from checkpoint without wandb run id. A new wandb run will be created."
        )
    update_run_manifest(
        cfg["_meta"]["run_dir"],
        {
            "resume": {
                "requested": bool(args.resume),
                "checkpoint_path": resume_ckpt,
                "wandb_run_id": resume_run_id,
            }
        },
    )

    if args.verbose:
        train_cfg = cfg.get("train", {})
        ds_cfg = cfg.get("dataset", {})
        model_cfg = cfg.get("model", {})
        meta = cfg.get("_meta", {}).get("run_metadata", {})
        logger.info(
            "Resolved run: name=%s mode=%s model=%s batch_size=%s epochs=%s device=%s deterministic=%s profile=%s",
            meta.get("run_name"),
            train_cfg.get("mode"),
            model_cfg.get("name"),
            train_cfg.get("batch_size"),
            train_cfg.get("epochs"),
            device,
            cfg.get("experiment", {}).get("deterministic", True),
            meta.get("runtime_profile"),
        )
        logger.info(
            "Dataset: name=%s augmentation=%s num_workers=%s val_ratio=%s data_dir=%s",
            ds_cfg.get("name"),
            ds_cfg.get("augmentation"),
            ds_cfg.get("num_workers"),
            ds_cfg.get("val_ratio"),
            ds_cfg.get("data_dir"),
        )
        logger.info(
            "Paths: config_sources=%s output_dir=%s run_dir=%s resolved_config=%s",
            cfg.get("_meta", {}).get("config_sources"),
            cfg.get("logging", {}).get("output_dir"),
            cfg.get("_meta", {}).get("run_dir"),
            cfg.get("_meta", {}).get("resolved_config_path"),
        )
        logger.info(
            "Runtime: platform=%s precision=%s wandb_run_id=%s wandb_url=%s",
            cfg.get("experiment", {}).get("platform"),
            cfg.get("experiment", {}).get("precision"),
            meta.get("wandb_run_id"),
            meta.get("wandb_url"),
        )
        if train_cfg.get("mode") == "multi_attack_erm":
            ma_cfg = train_cfg.get("multi_attack", {})
            logger.info(
                "Multi-attack: strategy=%s aggregation=%s include_clean=%s",
                ma_cfg.get("strategy"),
                ma_cfg.get("aggregation"),
                ma_cfg.get("include_clean"),
            )

    # Building data, model, and trainer components.
    train_loader, val_loader, _ = build_loaders(cfg)
    model = build_model(cfg)
    trainer = Trainer(
        cfg,
        model,
        train_loader,
        val_loader,
        device=device,
        logger=logger,
    )
    if resume_ckpt:
        trainer.load_checkpoint(resume_ckpt)
        update_run_manifest(
            cfg["_meta"]["run_dir"],
            {
                "resume": {
                    "loaded_checkpoint_path": resume_ckpt,
                    "wandb_run_id": trainer.wandb_run_id,
                }
            },
        )
    trainer.train()
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
