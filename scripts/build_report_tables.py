"""Build compact Markdown tables from exported evaluation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from ardg.utils.artifacts import update_artifact_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Markdown tables from exported eval results.")
    parser.add_argument(
        "--input",
        default="artifacts/exports/latest_results.json",
        help="Path to aggregated JSON export from scripts/export_results.py.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/reports/report_table.md",
        help="Path to the generated Markdown table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: List[Dict[str, Any]] = json.loads(Path(args.input).read_text(encoding="utf-8"))
    lines = [
        "| Run Dir | Clean Acc | Worst Acc | Avg Acc | Runtime (s) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {run_dir} | {clean_acc:.4f} | {worst_acc:.4f} | {avg_acc:.4f} | {runtime_sec:.2f} |".format(
                run_dir=row.get("run_dir", ""),
                clean_acc=float(row.get("clean_acc") or 0.0),
                worst_acc=float(row.get("worst_acc") or 0.0),
                avg_acc=float(row.get("avg_acc") or 0.0),
                runtime_sec=float(row.get("runtime_sec") or 0.0),
            )
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[INFO] report_table={output}")
    update_artifact_manifest(
        {
            "latest": {
                "reports": {
                    "table_md": str(output),
                }
            }
        }
    )


if __name__ == "__main__":
    main()
