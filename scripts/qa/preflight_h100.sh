#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
. "$SCRIPT_DIR/common.sh"

cd "$REPO_ROOT"

qa_section "H100 preflight"
qa_run "$PYTHON_BIN" scripts/preflight_check.py \
  --config configs/experiments/cifar10/vit_b16/baselines/erm.yaml \
  --profile configs/profiles/h100.yaml \
  --io_mode normalized \
  --check-wandb

echo "[PASS] h100 preflight completed"
