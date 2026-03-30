#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
. "$SCRIPT_DIR/common.sh"

cd "$REPO_ROOT"

qa_section "Local preflight"
qa_run "$PYTHON_BIN" scripts/preflight_check.py \
  --config configs/default.yaml \
  --profile configs/profiles/local_gpu.yaml \
  --io_mode normalized \
  --check-wandb

echo "[PASS] local preflight completed"
