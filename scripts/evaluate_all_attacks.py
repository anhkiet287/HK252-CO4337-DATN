"""Evaluate a checkpoint on clean + multiple attacks + optional AutoAttack.

This script is useful for checking robust accuracy of a clean-trained model.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ardg.attacks.attack_suite import build_attack
from ardg.config import DEFAULT_CONFIG_PATH
from ardg.evaluation.evaluator import Evaluator
from ardg.experiments.common import build_loaders, load_model_from_checkpoint, setup_run
from ardg.utils.logging import log_metrics
from ardg.utils.paths import get_run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate robust accuracy on many attacks.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to YAML config.")
    parser.add_argument(
        "--checkpoint",
        required=False,
        help="Path to checkpoint. If omitted, tries best.pt then last.pt in run dir.",
    )
    parser.add_argument("--split", default="test", choices=["val", "test"], help="Data split to evaluate.")
    parser.add_argument("--max-batches", type=int, default=0, help="0 means full split.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Optional DataLoader num_workers override (e.g., 0 for constrained environments).",
    )
    parser.add_argument("--fast", action="store_true", help="Use lighter attack settings for quick checks.")
    parser.add_argument("--no-autoattack", action="store_true", help="Skip AutoAttack.")
    parser.add_argument(
        "--save-json",
        default=None,
        help="Optional output JSON path. Default: <run_dir>/all_attack_eval_<split>.json",
    )
    return parser.parse_args()


def _resolve_checkpoint(cfg: Dict[str, Any], ckpt_arg: str | None) -> str:
    if ckpt_arg:
        return ckpt_arg
    run_dir = Path(get_run_dir(cfg))
    candidates = [run_dir]
    run_name = str(cfg.get("logging", {}).get("run_name", ""))
    if run_name.endswith("_eval"):
        candidates.append(run_dir.parent / run_name[: -len("_eval")])
    tried: List[str] = []
    for base in candidates:
        for cand in (base / "best.pt", base / "last.pt"):
            tried.append(str(cand))
            if cand.exists():
                return str(cand)
    raise FileNotFoundError(f"No checkpoint provided and none found. Tried: {tried}")


def _resolve_eps_alpha(cfg: Dict[str, Any]) -> Tuple[float, float]:
    attack_cfg = cfg.get("attack", {})
    eval_cfg = attack_cfg.get("eval", {})
    pgd20_cfg = eval_cfg.get("pgd20", {})
    val_cfg = attack_cfg.get("val", {})
    eps = float(pgd20_cfg.get("eps", eval_cfg.get("eps", val_cfg.get("eps", 8.0 / 255.0))))
    alpha = float(
        pgd20_cfg.get(
            "step_size",
            pgd20_cfg.get("alpha", eval_cfg.get("step_size", val_cfg.get("step_size", 2.0 / 255.0))),
        )
    )
    return eps, alpha


def _build_attack_specs(eps: float, alpha: float, *, fast: bool) -> List[Dict[str, Any]]:
    # Keep all attacks in pixel-space eps/alpha. Attack wrappers handle normalization bridge.
    deepfool_steps = 20 if fast else 50
    cw_steps = 20 if fast else 50
    fab_steps = 20 if fast else 30
    square_queries = 1000 if fast else 5000

    return [
        {"label": "fgsm", "type": "fgsm", "eps": eps},
        {"label": "fgsm_rs", "type": "fgsm_rs", "eps": eps, "alpha": eps},
        {"label": "pgd10_ce", "type": "pgd", "eps": eps, "alpha": alpha, "steps": 10, "loss": "ce", "random_start": True},
        {"label": "pgd20_ce", "type": "pgd", "eps": eps, "alpha": alpha, "steps": 20, "loss": "ce", "random_start": True},
        {"label": "pgd10_dlr", "type": "pgd", "eps": eps, "alpha": alpha, "steps": 10, "loss": "dlr", "random_start": True},
        {"label": "cw_l2", "type": "cw", "steps": cw_steps, "lr": 0.01},
        {"label": "deepfool_l2", "type": "deepfool", "steps": deepfool_steps, "overshoot": 0.02},
        {"label": "fab_linf", "type": "fab", "norm": "Linf", "eps": eps, "steps": fab_steps, "restarts": 1},
        {"label": "square_linf", "type": "square", "norm": "Linf", "eps": eps, "n_queries": square_queries, "restarts": 1},
    ]


def _build_autoattack_spec(cfg: Dict[str, Any], default_eps: float) -> Dict[str, Any]:
    aa_cfg = cfg.get("attack", {}).get("autoattack", {})
    return {
        "label": "autoattack",
        "type": "autoattack",
        "norm": str(aa_cfg.get("norm", "Linf")),
        "eps": float(aa_cfg.get("eps", default_eps)),
        "version": str(aa_cfg.get("version", "standard")),
        "n_classes": int(cfg.get("model", {}).get("num_classes", 10)),
        "verbose": bool(aa_cfg.get("verbose", False)),
    }


def main() -> None:
    args = parse_args()
    cfg, logger, run, device = setup_run(args.config, run_name_suffix="all_attacks_eval")
    if args.num_workers is not None:
        cfg.setdefault("dataset", {})["num_workers"] = int(args.num_workers)
    ckpt_path = _resolve_checkpoint(cfg, args.checkpoint)
    model = load_model_from_checkpoint(cfg, ckpt_path, device)

    _, val_loader, test_loader = build_loaders(cfg)
    loader = val_loader if args.split == "val" else test_loader
    eps, alpha = _resolve_eps_alpha(cfg)

    evaluator = Evaluator(model, loader, device=device, max_batches=int(args.max_batches))
    start = time.perf_counter()
    clean = evaluator.evaluate_clean()

    dataset_name = str(cfg["dataset"]["name"])
    attack_specs = _build_attack_specs(eps, alpha, fast=bool(args.fast))
    if not args.no_autoattack:
        attack_specs.append(_build_autoattack_spec(cfg, eps))
    robust: Dict[str, Dict[str, float]] = {}
    failures: Dict[str, str] = {}

    print(f"[INFO] checkpoint={ckpt_path}")
    print(f"[INFO] split={args.split} clean_acc={float(clean['acc']):.6f} n={int(clean['n_samples'])}")
    for spec in attack_specs:
        label = str(spec["label"])
        spec_for_build = dict(spec)
        spec_for_build["name"] = str(spec_for_build.get("type", "pgd"))
        try:
            attack = build_attack(spec_for_build, model, dataset_name=dataset_name)
            attack_start = time.perf_counter()
            metrics = evaluator.evaluate_under_attack(attack)
            metrics["runtime_sec"] = float(time.perf_counter() - attack_start)
            robust[label] = metrics
            print(
                f"[ATTACK] {label}: acc={float(metrics['acc']):.6f} "
                f"loss={float(metrics['loss']):.6f} n={int(metrics['n_samples'])} "
                f"runtime={float(metrics['runtime_sec']):.2f}s"
            )
        except Exception as exc:
            failures[label] = str(exc)
            print(f"[ERROR] {label}: {exc}")

    elapsed = time.perf_counter() - start
    worst_robust = min((float(v["acc"]) for v in robust.values()), default=float(clean["acc"]))
    print(f"[SUMMARY] clean_acc={float(clean['acc']):.6f} worst_robust_acc={worst_robust:.6f}")

    out_payload = {
        "config": args.config,
        "checkpoint": ckpt_path,
        "split": args.split,
        "max_batches": int(args.max_batches),
        "fast": bool(args.fast),
        "clean": clean,
        "robust": robust,
        "failures": failures,
        "eps": eps,
        "alpha": alpha,
        "runtime_sec": elapsed,
        "worst_robust_acc": worst_robust,
    }
    out_path = (
        Path(args.save_json)
        if args.save_json
        else Path(get_run_dir(cfg)) / f"all_attack_eval_{args.split}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
    print(f"[INFO] saved={out_path}")

    metric_row: Dict[str, float] = {
        "acc_clean": float(clean["acc"]),
        "worst_robust_acc": float(worst_robust),
        "runtime_sec": float(elapsed),
    }
    for name, met in robust.items():
        metric_row[f"acc_{name}"] = float(met["acc"])
    log_metrics(logger, metric_row, step=0, split=f"{args.split}_all_attacks")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
