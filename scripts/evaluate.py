"""Evaluate a trained model on clean/PGD20/AutoAttack metrics."""

import argparse
import time
from pathlib import Path

import torch

from ardg.attacks.attack_suite import build_eval_attacks
from ardg.attacks.autoattack import run_autoattack
from ardg.config import DEFAULT_CONFIG_PATH
from ardg.evaluation.evaluator import Evaluator
from ardg.experiments.common import build_loaders, load_model_from_checkpoint, setup_run
from ardg.utils.logging import log_metrics
from ardg.utils.paths import get_run_dir


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to YAML config.")
    parser.add_argument(
        "--checkpoint",
        required=False,
        help="Path to checkpoint. If omitted, tries best.pt then last.pt in run dir.",
    )
    parser.add_argument(
        "--splits",
        default="test",
        help="Comma-separated splits to evaluate (subset of val,test). Default: test",
    )
    return parser.parse_args()


def _resolve_checkpoint(cfg: dict, ckpt_arg: str | None) -> str:
    if ckpt_arg:
        return ckpt_arg

    run_dir = Path(get_run_dir(cfg))
    candidates = [run_dir]
    run_name = str(cfg.get("logging", {}).get("run_name", ""))
    if run_name.endswith("_eval"):
        candidates.append(run_dir.parent / run_name[: -len("_eval")])

    tried = []
    for base in candidates:
        for cand in (base / "best.pt", base / "last.pt"):
            tried.append(str(cand))
            if cand.exists():
                return str(cand)
    raise FileNotFoundError(f"No checkpoint provided and none found. Tried: {tried}")


def _resolve_aa_eps(cfg: dict) -> float:
    aa_cfg = cfg.get("attack", {}).get("autoattack", {})
    eval_cfg = cfg.get("attack", {}).get("eval", {})
    pgd20_cfg = eval_cfg.get("pgd20", {})
    return float(
        aa_cfg.get(
            "eps",
            pgd20_cfg.get("eps", eval_cfg.get("eps", cfg.get("attack", {}).get("val", {}).get("eps", 8.0 / 255.0))),
        )
    )


def main() -> None:
    args = parse_args()
    cfg, logger, run, device = setup_run(args.config, run_name_suffix="eval")
    ckpt_path = _resolve_checkpoint(cfg, args.checkpoint)
    model = load_model_from_checkpoint(cfg, ckpt_path, device)

    _, val_loader, test_loader = build_loaders(cfg)
    requested = {s.strip() for s in str(args.splits).split(",") if s.strip()}
    split_loaders = {"val": val_loader, "test": test_loader}

    has_attack = "attack" in cfg
    eval_enabled = cfg.get("attack", {}).get("eval", {}).get("enabled", has_attack)
    attacks = build_eval_attacks(cfg, model) if (has_attack and eval_enabled) else {}
    if "pgd20" not in attacks:
        raise RuntimeError(
            "PGD20 evaluation is required. Set attack.eval.enabled=true and provide attack.eval.pgd20 config."
        )

    aa_cfg = cfg.get("attack", {}).get("autoattack", {})
    aa_enabled = bool(aa_cfg.get("enabled", False))
    aa_eps = _resolve_aa_eps(cfg)
    aa_norm = str(aa_cfg.get("norm", "Linf"))
    aa_version = str(aa_cfg.get("version", "standard"))
    aa_max_batches = aa_cfg.get("max_batches")

    step = 0
    start = time.perf_counter()
    test_n_samples = 0
    test_aa_runtime = None
    test_aa_samples = None
    for split_name in ("val", "test"):
        if split_name not in requested:
            continue

        loader = split_loaders[split_name]
        evaluator = Evaluator(model, loader, device=device, attack_suite=attacks)
        clean = evaluator.evaluate_clean()
        acc_clean = float(clean.get("acc", 0.0))
        n_samples = int(clean.get("n_samples", 0))
        test_n_samples = n_samples if split_name == "test" else test_n_samples

        acc_pgd20 = None
        if attacks:
            pgd20 = evaluator.evaluate_suite({"pgd20": attacks["pgd20"]})["pgd20"]
            acc_pgd20 = float(pgd20.get("acc", 0.0))

        acc_aa = None
        if aa_enabled:
            aa = run_autoattack(
                model,
                loader,
                aa_eps,
                device,
                dataset_name=cfg["dataset"]["name"],
                norm=aa_norm,
                version=aa_version,
                max_batches=aa_max_batches,
            )
            acc_aa = float(aa.get("acc", 0.0))
            if split_name == "test":
                test_aa_runtime = float(aa.get("aa_runtime_sec", 0.0))
                test_aa_samples = int(aa.get("aa_n_samples", aa.get("n_samples", 0)))

        worst_acc = None
        if acc_pgd20 is not None and acc_aa is not None:
            worst_acc = min(acc_pgd20, acc_aa)
        elif acc_pgd20 is not None:
            worst_acc = acc_pgd20

        metrics = {"acc_clean": acc_clean}
        if acc_pgd20 is not None:
            metrics["acc_pgd20"] = acc_pgd20
        if acc_aa is not None:
            metrics["acc_aa"] = acc_aa
        if worst_acc is not None:
            metrics["worst_acc"] = worst_acc
        log_metrics(logger, metrics, step=step, split=split_name)
        step += 1

    runtime = time.perf_counter() - start
    sys_metrics = {
        "runtime_sec": runtime,
        "n_samples_eval": test_n_samples,
        "torch_version": str(torch.__version__),
    }
    if str(device).startswith("cuda") and torch.cuda.is_available():
        try:
            sys_metrics["gpu_name"] = str(torch.cuda.get_device_name(torch.device(device)))
        except Exception:
            pass
    if test_aa_runtime is not None:
        sys_metrics["aa_runtime_sec"] = test_aa_runtime
    if test_aa_samples is not None:
        sys_metrics["aa_n_samples"] = test_aa_samples
    log_metrics(logger, sys_metrics, step=step, split="sys")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
