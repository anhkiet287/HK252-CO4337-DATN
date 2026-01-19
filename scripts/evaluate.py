"""Evaluate a trained model on clean and adversarial inputs."""

import argparse
import time

import torch

from ardg.attacks.attack_suite import build_eval_attacks
from ardg.attacks.autoattack import run_autoattack
from ardg.config import DEFAULT_CONFIG_PATH, load_config
from ardg.data.cifar import get_dataloaders
from ardg.evaluation.evaluator import evaluate_clean, evaluate_suite
from ardg.models.resnet import resnet18_cifar
from ardg.utils.logging import init_wandb, log_metrics, setup_logging
from ardg.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    return parser.parse_args()


def main() -> None:
    """Run evaluation on clean and adversarial inputs."""
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["experiment"]["seed"])
    logger = setup_logging()
    init_wandb(cfg)
    device = cfg["experiment"].get("device", "cpu")

    model = resnet18_cifar(cfg["model"]["num_classes"])
    checkpoint = torch.load(args.checkpoint, map_location=torch.device(device))
    state_dict = checkpoint.get("model") if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)

    _, val_loader, test_loader = get_dataloaders(cfg)
    attacks = build_eval_attacks(cfg, model)

    start = time.perf_counter()
    for split_name, loader in (("val", val_loader), ("test", test_loader)):
        clean_metrics = evaluate_clean(model, loader, device)
        clean_metrics["device"] = str(device)
        log_metrics(logger, clean_metrics, step=0, split=f"{split_name}_clean")

        attack_metrics = evaluate_suite(model, loader, attacks, device)
        for name, metrics in attack_metrics.items():
            metrics["device"] = str(device)
            log_metrics(logger, metrics, step=0, split=f"{split_name}_{name}")

        if cfg.get("attack", {}).get("autoattack", {}).get("enabled", False):
            autoattack_metrics = run_autoattack(
                model,
                loader,
                cfg["attack"]["autoattack"]["eps"],
                device,
            )
            autoattack_metrics["device"] = str(device)
            log_metrics(logger, autoattack_metrics, step=0, split=f"{split_name}_autoattack")

    elapsed = time.perf_counter() - start
    log_metrics(logger, {"time_sec": elapsed, "device": str(device)}, step=0, split="eval_runtime")


if __name__ == "__main__":
    main()
