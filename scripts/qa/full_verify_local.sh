#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
. "$SCRIPT_DIR/common.sh"

cd "$REPO_ROOT"

qa_section "Full local verification"
qa_run sh "$SCRIPT_DIR/preflight_local.sh"
qa_run sh "$SCRIPT_DIR/test_wandb_policy.sh"
qa_run sh "$SCRIPT_DIR/smoke_resnet18_dev_fast.sh"
qa_run sh "$SCRIPT_DIR/test_resume_resnet18.sh"
qa_run sh "$SCRIPT_DIR/smoke_resnet50.sh"

qa_section "Export and report validation"
qa_run "$PYTHON_BIN" scripts/export_results.py --root outputs
qa_run "$PYTHON_BIN" scripts/build_report_tables.py
qa_assert_exists artifacts/exports/latest_results.json
qa_assert_exists artifacts/exports/latest_results.csv
qa_assert_exists artifacts/reports/report_table.md

echo "[PASS] full local verification completed"
