"""Evaluate a trained model on test split with configurable attack suite."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import torch

from ardg.attacks.attack_suite import build_eval_suite
from ardg.config import DEFAULT_CONFIG_PATH
from ardg.evaluation.evaluator import Evaluator
from ardg.evaluation.summary import summarize_suite
from ardg.experiments.common import build_loaders, load_model_from_checkpoint, setup_run
from ardg.utils.logging import log_metrics
from ardg.utils.paths import get_run_dir
from ardg.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained model (test split only).")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to YAML config.")
    parser.add_argument(
        "--checkpoint",
        required=False,
        help="Path to checkpoint. If omitted, tries best.pt then last.pt in run dir.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional cap on number of test batches. If omitted, uses config or full test set.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Optional DataLoader num_workers override (e.g., 0 for constrained environments).",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        help="Optional cap on number of test samples loaded from dataset.",
    )
    parser.add_argument(
        "--smoke-one-sample",
        action="store_true",
        help="Smoke mode: force evaluation on exactly 1 test sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional eval seed override. Defaults to experiment.seed from config.",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force deterministic eval behavior. Defaults to experiment.deterministic from config.",
    )
    parser.add_argument(
        "--save-json",
        default=None,
        help="Optional output JSON path. Default: <run_dir>/eval_test_summary.json",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print resolved evaluation setup and attack suite details.",
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

    tried = []
    for base in candidates:
        for cand in (base / "best.pt", base / "last.pt"):
            tried.append(str(cand))
            if cand.exists():
                return str(cand)
    raise FileNotFoundError(f"No checkpoint provided and none found. Tried: {tried}")


def _resolve_max_batches(cfg: Dict[str, Any], cli_max_batches: int | None) -> int:
    if cli_max_batches is not None:
        return max(0, int(cli_max_batches))

    attack_cfg = cfg.get("attack", {})
    eval_suite_cfg = attack_cfg.get("eval_suite", {})
    if isinstance(eval_suite_cfg, dict) and eval_suite_cfg.get("max_batches") is not None:
        return max(0, int(eval_suite_cfg["max_batches"]))

    eval_cfg = attack_cfg.get("eval", {})
    if isinstance(eval_cfg, dict) and eval_cfg.get("max_batches") is not None:
        return max(0, int(eval_cfg["max_batches"]))

    return 0


def _resolve_eval_seed(cfg: Dict[str, Any], cli_seed: int | None) -> int:
    if cli_seed is not None:
        return int(cli_seed)
    return int(cfg.get("experiment", {}).get("seed", 42))


def _resolve_eval_deterministic(cfg: Dict[str, Any], cli_det: bool | None) -> bool:
    if cli_det is not None:
        return bool(cli_det)
    return bool(cfg.get("experiment", {}).get("deterministic", True))


def _log_attack_comparison_chart(clean: Dict[str, float], robust: Dict[str, Dict[str, float]], step: int) -> None:
    """Log one W&B chart/table to compare model performance across attacks."""
    try:
        import wandb  # type: ignore
    except ImportError:
        return
    if getattr(wandb, "run", None) is None:
        return

    if not robust:
        return

    table = wandb.Table(columns=["attack", "acc", "loss", "runtime_sec"])
    table.add_data("clean", float(clean.get("acc_clean", 0.0)), float(clean.get("loss_clean", 0.0)), 0.0)
    for name, metrics in robust.items():
        table.add_data(
            str(name),
            float(metrics.get("acc_adv", 0.0)),
            float(metrics.get("loss_adv", 0.0)),
            float(metrics.get("runtime_sec", 0.0)),
        )

    wandb.log(
        {
            "test/attack_compare_table": table,
            "test/attack_acc_compare": wandb.plot.bar(
                table,
                "attack",
                "acc",
                title="Model Accuracy by Attack",
            ),
            "test/attack_loss_compare": wandb.plot.bar(
                table,
                "attack",
                "loss",
                title="Model Loss by Attack",
            ),
        },
        step=step,
    )


def main() -> None:
    args = parse_args()
    cfg, logger, run, device = setup_run(args.config, run_name_suffix="eval")

    eval_seed = _resolve_eval_seed(cfg, args.seed)
    eval_deterministic = _resolve_eval_deterministic(cfg, args.deterministic)
    # Re-assert deterministic policy explicitly for evaluation runs.
    set_seed(eval_seed, deterministic=eval_deterministic)

    if args.num_workers is not None:
        cfg.setdefault("dataset", {})["num_workers"] = int(args.num_workers)
    elif eval_deterministic:
        # Conservative default for repeatable loader behavior across environments.
        cfg.setdefault("dataset", {})["num_workers"] = 0

    if args.smoke_one_sample:
        cfg.setdefault("dataset", {})["max_test_samples"] = 1
    elif args.max_test_samples is not None:
        cfg.setdefault("dataset", {})["max_test_samples"] = max(1, int(args.max_test_samples))

    ckpt_path = _resolve_checkpoint(cfg, args.checkpoint)
    model = load_model_from_checkpoint(cfg, ckpt_path, device)

    _, _, test_loader = build_loaders(cfg)
    cli_max_batches = args.max_batches
    if args.smoke_one_sample and cli_max_batches is None:
        cli_max_batches = 1
    max_batches = _resolve_max_batches(cfg, cli_max_batches)
    evaluator = Evaluator(model, test_loader, device=device, max_batches=max_batches)

    start = time.perf_counter()
    clean = evaluator.evaluate_clean()

    attacks = build_eval_suite(cfg, model)
    robust: Dict[str, Dict[str, float]] = {}
    failures: Dict[str, str] = {}

    print(f"[INFO] checkpoint={ckpt_path}")
    print(f"[INFO] split=test clean_acc={float(clean['acc_clean']):.6f} n={int(clean['n_samples'])}")
    print(
        f"[INFO] deterministic={eval_deterministic} seed={eval_seed} "
        f"num_workers={int(cfg.get('dataset', {}).get('num_workers', 0))}"
    )
    print(f"[INFO] max_test_samples={cfg.get('dataset', {}).get('max_test_samples', 'full')}")
    if args.verbose:
        print(f"[INFO] max_batches={max_batches}")
        print(f"[INFO] eval_attacks={list(attacks.keys()) if attacks else []}")
        print(f"[INFO] model={cfg.get('model', {}).get('name')} mode={cfg.get('train', {}).get('mode')}")
    for label, attack in attacks.items():
        try:
            attack_start = time.perf_counter()
            metrics = evaluator.evaluate_under_attack(attack)
            metrics["runtime_sec"] = float(time.perf_counter() - attack_start)
            robust[label] = metrics
            print(
                f"[ATTACK] {label}: acc={float(metrics['acc_adv']):.6f} "
                f"loss={float(metrics['loss_adv']):.6f} n={int(metrics['n_samples'])} "
                f"runtime={float(metrics['runtime_sec']):.2f}s"
            )
        except Exception as exc:
            failures[label] = str(exc)
            print(f"[ERROR] {label}: {exc}")

    elapsed = time.perf_counter() - start
    worst_robust = min((float(v["acc_adv"]) for v in robust.values()), default=float(clean["acc_clean"]))

    per_domain = {"clean": clean, **robust}
    summary = summarize_suite(per_domain, prefix="test")
\n+    metric_row = {k.replace(\"test/\", \"\"): v for k, v in summary.items() if k.startswith(\"test/\")}\n+    metric_row[\"runtime_sec\"] = float(elapsed)\n+    log_metrics(logger, metric_row, step=0, split=\"test\")\n     _log_attack_comparison_chart(clean, robust, step=0)
    _log_attack_comparison_chart(clean, robust, step=0)

    sys_metrics: Dict[str, float | str] = {
        "runtime_sec": float(elapsed),
        "n_samples_eval": int(clean.get("n_samples", 0)),
        "torch_version": str(torch.__version__),
        "num_attacks": int(len(attacks)),
    }
    if str(device).startswith("cuda") and torch.cuda.is_available():
        try:
            sys_metrics["gpu_name"] = str(torch.cuda.get_device_name(torch.device(device)))
        except Exception:
            pass
    log_metrics(logger, sys_metrics, step=1, split="sys")

    payload = {
        "config": args.config,
        "checkpoint": ckpt_path,
        "split": "test",
        "seed": int(eval_seed),
        "deterministic": bool(eval_deterministic),
        "max_batches": int(max_batches),
        "clean": clean,
        "robust": robust,
        "summary": summary,
        "failures": failures,
        "runtime_sec": float(elapsed),
        "worst_robust_acc": float(worst_robust),
    }
    out_path = (
        Path(args.save_json)
        if args.save_json
        else Path(get_run_dir(cfg)) / "eval_test_summary.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[INFO] saved={out_path}")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
