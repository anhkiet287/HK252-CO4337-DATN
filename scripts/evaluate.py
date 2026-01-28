"""Evaluate a trained model on clean and adversarial inputs (config-driven)."""

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
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to YAML config.")
    parser.add_argument(
        "--checkpoint",
        required=False,
        help="Path to model checkpoint. If omitted, tries best.pt then last.pt in run dir.",
    )
    parser.add_argument(
        "--splits",
        default="test",
        help="Comma-separated splits to evaluate (subset of val,test). Default: test",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg, logger, run, device = setup_run(args.config, run_name_suffix="eval")

    # Resolve checkpoint
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        run_dir = Path(get_run_dir(cfg))
        cand_dirs = [run_dir]
        run_name = cfg.get("logging", {}).get("run_name", "")
        if run_name.endswith("_eval"):
            base_name = run_name[: -len("_eval")]
            cand_dirs.append(run_dir.parent / base_name)
        tried = []
        for d in cand_dirs:
            for cand in (d / "best.pt", d / "last.pt"):
                tried.append(str(cand))
                if cand.exists():
                    ckpt_path = str(cand)
                    break
            if ckpt_path is not None:
                break
        if ckpt_path is None:
            raise FileNotFoundError(
                f"No checkpoint provided and none found. Tried paths: {tried}"
            )

    model = load_model_from_checkpoint(cfg, ckpt_path, device)

    # Data + attacks
    _, val_loader, test_loader = build_loaders(cfg)
    has_attack = "attack" in cfg
    eval_enabled = cfg.get("attack", {}).get("eval", {}).get("enabled", has_attack)
    attacks = build_eval_attacks(cfg, model) if (has_attack and eval_enabled) else {}

    requested = {s.strip() for s in args.splits.split(",") if s.strip()}
    split_loaders = {"val": val_loader, "test": test_loader}

    # Eval loop
    start = time.perf_counter()
    step = 0
    summary_rows = []
    for split_name in ("val", "test"):
        if split_name not in requested:
            continue
        loader = split_loaders[split_name]
        row = {"split": split_name}

        # Clean
        clean_metrics = evaluate_clean(model, loader, device)
        clean_metrics["device"] = str(device)
        log_metrics(logger, clean_metrics, step=step, split=f"{split_name}_clean")
        row["clean_acc"] = clean_metrics.get("acc")
        row["clean_loss"] = clean_metrics.get("loss")
        step += 1

        # PGD / eval attacks
        if attacks:
            attack_metrics = evaluate_suite(model, loader, attacks, device)
            for name, metrics in attack_metrics.items():
                metrics["device"] = str(device)
                log_metrics(logger, metrics, step=step, split=f"{split_name}_{name}")
                row[f"{name}_acc"] = metrics.get("acc")
                row[f"{name}_loss"] = metrics.get("loss")
            step += 1

        # AutoAttack
        if cfg.get("attack", {}).get("autoattack", {}).get("enabled", False):
            autoattack_metrics = run_autoattack(
                model,
                loader,
                cfg["attack"]["autoattack"]["eps"],
                device,
            )
            autoattack_metrics["device"] = str(device)
            log_metrics(logger, autoattack_metrics, step=step, split=f"{split_name}_autoattack")
            row["autoattack_acc"] = autoattack_metrics.get("acc")
            step += 1

        summary_rows.append(row)

    elapsed = time.perf_counter() - start
    log_metrics(logger, {"time_sec": elapsed, "device": str(device)}, step=step, split="eval_runtime")

    # W&B summary/table
    try:
        import wandb  # type: ignore

        if run is not None and getattr(wandb, "run", None) is not None:
            cols = ["split", "clean_acc", "clean_loss", "pgd_acc", "pgd_loss", "autoattack_acc", "time_sec"]
            table = wandb.Table(columns=cols)
            for row in summary_rows:
                table.add_data(
                    row.get("split"),
                    row.get("clean_acc"),
                    row.get("clean_loss"),
                    row.get("pgd_acc"),
                    row.get("pgd_loss"),
                    row.get("autoattack_acc"),
                    elapsed,
                )
            wandb.log({"eval/summary_table": table}, step=step)
            for row in summary_rows:
                if row.get("split") == "test":
                    if "pgd_acc" in row:
                        wandb.run.summary["test_pgd_acc"] = row["pgd_acc"]
                    if "clean_acc" in row:
                        wandb.run.summary["test_clean_acc"] = row["clean_acc"]
                    if "autoattack_acc" in row:
                        wandb.run.summary["test_autoattack_acc"] = row["autoattack_acc"]
            wandb.run.summary["eval_time_sec"] = elapsed
    except ImportError:
        pass


if __name__ == "__main__":
    main()
