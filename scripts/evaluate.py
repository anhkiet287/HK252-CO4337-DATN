"""Evaluate a trained model on clean and adversarial inputs."""

import argparse
import time

from ardg.attacks.attack_suite import build_eval_attacks
from ardg.attacks.autoattack import run_autoattack
from ardg.config import DEFAULT_CONFIG_PATH
from ardg.evaluation.evaluator import evaluate_suite
from ardg.experiments.common import (
    build_loaders,
    load_model_from_checkpoint,
    run_clean_eval,
    setup_run,
)
from ardg.utils.logging import log_metrics


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    return parser.parse_args()


def main() -> None:
    """Run evaluation on clean and adversarial inputs."""
    # Parsing arguments and loading config.
    args = parse_args()
    cfg, logger, _, device = setup_run(args.config)

    # Building model and loading checkpoint.
    model = load_model_from_checkpoint(cfg, args.checkpoint, device)

    # Building data loaders and attacks.
    _, val_loader, test_loader = build_loaders(cfg)
    eval_enabled = cfg.get("attack", {}).get("eval", {}).get("enabled", True)
    attacks = build_eval_attacks(cfg, model) if eval_enabled else {}

    # Running evaluation.
    start = time.perf_counter()
    base_step = 0
    step_map = {"val": base_step, "test": base_step + 1}
    run_clean_eval(model, val_loader, test_loader, device, logger, step=base_step)
    for split_name, loader in (("val", val_loader), ("test", test_loader)):
        
        # Evaluate on adversarial inputs
        if attacks:
            attack_metrics = evaluate_suite(model, loader, attacks, device)
            for name, metrics in attack_metrics.items():
                metrics["device"] = str(device)
                log_metrics(logger, metrics, step=step_map[split_name], split=f"{split_name}_{name}")

        # Optional AutoAttack evaluation
        if cfg.get("attack", {}).get("autoattack", {}).get("enabled", False):
            autoattack_metrics = run_autoattack(
                model,
                loader,
                cfg["attack"]["autoattack"]["eps"],
                device,
            )
            autoattack_metrics["device"] = str(device)
            log_metrics(logger, autoattack_metrics, step=step_map[split_name], split=f"{split_name}_autoattack")

    elapsed = time.perf_counter() - start
    log_metrics(logger, {"time_sec": elapsed, "device": str(device)}, step=base_step + 2, split="eval_runtime")


if __name__ == "__main__":
    main()
