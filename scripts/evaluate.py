"""Evaluate a trained model on clean and adversarial inputs."""

import argparse
import time
from pathlib import Path

from ardg.attacks.attack_suite import build_eval_attacks
from ardg.attacks.autoattack import run_autoattack
from ardg.config import DEFAULT_CONFIG_PATH
from ardg.evaluation.evaluator import evaluate_suite, evaluate_clean
from ardg.experiments.common import (
    build_loaders,
    load_model_from_checkpoint,
    setup_run,
)
from ardg.utils.paths import get_run_dir
from ardg.utils.logging import log_metrics


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to YAML config.")
    parser.add_argument("--checkpoint", required=False, help="Path to model checkpoint. If omitted, will try best.pt then last.pt in the run directory.")
    parser.add_argument(
        "--splits",
        default="val,test",
        help="Comma-separated list of splits to evaluate (choose from val,test). Default: val,test",
    )
    return parser.parse_args()


def main() -> None:
    """Run evaluation on clean and adversarial inputs."""
    # Parsing arguments and loading config.
    args = parse_args()
    cfg, logger, _, device = setup_run(args.config, run_name_suffix="eval")

    # Building model and loading checkpoint.
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        run_dir = Path(get_run_dir(cfg))
        cand = [run_dir / "best.pt", run_dir / "last.pt"]
        ckpt_path = next((str(p) for p in cand if p.exists()), None)
        if ckpt_path is None:
            raise FileNotFoundError(f"No checkpoint provided and none found in {run_dir} (looked for best.pt / last.pt).")
    model = load_model_from_checkpoint(cfg, ckpt_path, device)

    # Building data loaders and attacks.
    _, val_loader, test_loader = build_loaders(cfg)
    has_attack_block = "attack" in cfg
    eval_enabled = cfg.get("attack", {}).get("eval", {}).get("enabled", has_attack_block)
    attacks = build_eval_attacks(cfg, model) if (has_attack_block and eval_enabled) else {}

    requested = {s.strip() for s in args.splits.split(",") if s.strip()}
    split_loaders = {"val": val_loader, "test": test_loader}

    # Running evaluation.
    start = time.perf_counter()
    step = 0
    for split_name in ("val", "test"):
        if split_name not in requested:
            continue
        loader = split_loaders[split_name]

        # Clean eval
        clean_metrics = evaluate_clean(model, loader, device)
        clean_metrics["device"] = str(device)
        log_metrics(logger, clean_metrics, step=step, split=f"{split_name}_clean")
        step += 1

        # Adversarial eval
        if attacks:
            attack_metrics = evaluate_suite(model, loader, attacks, device)
            for name, metrics in attack_metrics.items():
                metrics["device"] = str(device)
                log_metrics(logger, metrics, step=step, split=f"{split_name}_{name}")
            step += 1

        # Optional AutoAttack eval
        if cfg.get("attack", {}).get("autoattack", {}).get("enabled", False):
            autoattack_metrics = run_autoattack(
                model,
                loader,
                cfg["attack"]["autoattack"]["eps"],
                device,
            )
            autoattack_metrics["device"] = str(device)
            log_metrics(logger, autoattack_metrics, step=step, split=f"{split_name}_autoattack")
            step += 1

    elapsed = time.perf_counter() - start
    log_metrics(logger, {"time_sec": elapsed, "device": str(device)}, step=step, split="eval_runtime")


if __name__ == "__main__":
    main()
