#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
. "$SCRIPT_DIR/common.sh"

cd "$REPO_ROOT"

TRAIN_CFG=configs/experiments/cifar10/resnet50/baselines/erm.yaml
PROFILE=configs/profiles/dev_fast.yaml
RUN_DIR=outputs/dev_fast/resnet50_dev_baseline_erm_seed42

qa_section "ResNet-50 smoke train"
qa_run "$PYTHON_BIN" scripts/train.py \
  --config "$TRAIN_CFG" \
  --profile "$PROFILE"

qa_assert_exists "$RUN_DIR/checkpoints/best.pt"
qa_assert_exists "$RUN_DIR/train/summary.json"

echo "[PASS] ResNet-50 smoke completed"
