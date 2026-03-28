"""Collect evaluation summaries into one export file."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from ardg.utils.artifacts import update_artifact_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export eval summaries from run directories.")
    parser.add_argument(
        "--root",
        default="outputs",
        help="Root directory to scan for run manifests and eval summaries.",
    )
    parser.add_argument(
        "--json-out",
        default="artifacts/exports/latest_results.json",
        help="Path to aggregated JSON export.",
    )
    parser.add_argument(
        "--csv-out",
        default="artifacts/exports/latest_results.csv",
        help="Path to aggregated CSV export.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    summary_paths = _collect_summary_paths(root)
    rows = [_to_row(path) for path in summary_paths]

    json_out = Path(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")

    csv_out = Path(args.csv_out)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_dir",
        "checkpoint",
        "seed",
        "clean_acc",
        "worst_acc",
        "avg_acc",
        "runtime_sec",
        "num_attacks",
    ]
    with csv_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    print(f"[INFO] exported_json={json_out}")
    print(f"[INFO] exported_csv={csv_out}")
    print(f"[INFO] runs={len(rows)}")
    update_artifact_manifest(
        {
            "latest": {
                "exports": {
                    "json": str(json_out),
                    "csv": str(csv_out),
                }
            }
        }
    )


def _collect_summary_paths(root: Path) -> List[Path]:
    seen: set[Path] = set()
    collected: List[Path] = []

    for manifest_path in sorted(root.glob("**/run_manifest.json")):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary_path = payload.get("evaluation", {}).get("summary_json")
        if summary_path:
            candidate = Path(str(summary_path))
            if candidate.exists():
                resolved = candidate.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    collected.append(candidate)

    for pattern in ("**/eval/summary.json", "**/eval_test_summary.json"):
        for candidate in sorted(root.glob(pattern)):
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                collected.append(candidate)

    return collected


def _to_row(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    robust = payload.get("robust", {})
    run_dir = path.parent.parent if path.parent.name == "eval" else path.parent
    return {
        "run_dir": str(run_dir),
        "checkpoint": payload.get("checkpoint"),
        "seed": payload.get("seed"),
        "clean_acc": summary.get("test/acc_clean"),
        "worst_acc": summary.get("test/acc_worst"),
        "avg_acc": summary.get("test/acc_avg"),
        "runtime_sec": payload.get("runtime_sec"),
        "num_attacks": len(robust),
    }


if __name__ == "__main__":
    main()
