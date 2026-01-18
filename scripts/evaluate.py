"""Evaluate a trained model on clean and adversarial inputs.

Loads a YAML config, restores a model checkpoint, and writes evaluation
reports under outputs/reports/.
"""

import argparse
import json

from ardg.attacks.attack_suite import build_eval_attacks
from ardg.attacks.autoattack import run_autoattack
from ardg.config import DEFAULT_CONFIG_PATH, load_config
from ardg.data.cifar import get_dataloaders
from ardg.evaluation.evaluator import evaluate_clean, evaluate_suite
from ardg.models.resnet import resnet18_cifar
from ardg.utils.paths import ensure_dir, project_root


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to YAML config.")
    parser.add_argument("--checkpoint", required=False, help="Path to model checkpoint.")
    return parser.parse_args()


def _write_report(cfg: dict, report: dict) -> str:
    """Write evaluation report to outputs/reports.

    Args:
        cfg: Configuration dictionary.
        report: Evaluation metrics to serialize.

    Returns:
        Path to the written report JSON file.

    Side effects:
        Creates directories and writes a JSON file to disk.
    """
    output_dir = cfg.get("logging", {}).get("output_dir", "outputs")
    run_name = cfg.get("logging", {}).get("run_name", "run")
    report_dir = project_root() / output_dir / "reports"
    ensure_dir(str(report_dir))
    report_path = report_dir / f"{run_name}_eval.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return str(report_path)


def main() -> None:
    """Run evaluation on clean and adversarial inputs.

    Args:
        None.

    Returns:
        None.

    Side effects:
        Writes JSON reports to outputs/reports/ and may run long attacks.
    """
    args = parse_args()
    cfg = load_config(args.config)

    model = resnet18_cifar(cfg["model"]["num_classes"])
    # TODO: load checkpoint into model when implemented.

    _, val_loader = get_dataloaders(cfg)
    device = cfg["experiment"].get("device", "cpu")

    report = {
        "clean": evaluate_clean(model, val_loader, device),
    }

    attacks = build_eval_attacks(cfg, model)
    report["attack_suite"] = evaluate_suite(model, val_loader, attacks, device)

    if cfg.get("attack", {}).get("autoattack", {}).get("enabled", False):
        report["autoattack"] = run_autoattack(
            model,
            val_loader,
            cfg["attack"]["autoattack"]["eps"],
            device,
        )

    report_path = _write_report(cfg, report)
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
