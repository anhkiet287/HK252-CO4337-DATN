#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
. "$SCRIPT_DIR/common.sh"

cd "$REPO_ROOT"

TRAIN_CFG=configs/experiments/cifar10/resnet18/baselines/erm.yaml
EVAL_CFG=configs/experiments/cifar10/resnet18/eval/baseline_erm_all_attacks.yaml
PROFILE=configs/profiles/dev_fast.yaml
TRAIN_RUN_DIR=outputs/dev_fast/resnet18_dev_baseline_erm_seed42
EVAL_RUN_DIR=outputs/dev_fast/resnet18_dev_baseline_erm_seed42_eval
BEST_CKPT="$TRAIN_RUN_DIR/checkpoints/best.pt"

qa_section "ResNet-18 dev_fast preflight"
qa_run "$PYTHON_BIN" scripts/preflight_check.py \
  --config "$TRAIN_CFG" \
  --profile "$PROFILE" \
  --io_mode normalized \
  --check-wandb

qa_section "ResNet-18 dev_fast train"
qa_run "$PYTHON_BIN" scripts/train.py \
  --config "$TRAIN_CFG" \
  --profile "$PROFILE" \
  --verbose

qa_assert_exists "$TRAIN_RUN_DIR"
qa_assert_exists "$TRAIN_RUN_DIR/run_manifest.json"
qa_assert_exists "$TRAIN_RUN_DIR/resolved_config.yaml"
qa_assert_exists "$TRAIN_RUN_DIR/train/summary.json"
qa_assert_exists "$TRAIN_RUN_DIR/checkpoints/last.pt"
qa_assert_exists "$BEST_CKPT"

qa_section "ResNet-18 dev_fast eval"
qa_run "$PYTHON_BIN" scripts/evaluate.py \
  --config "$EVAL_CFG" \
  --profile "$PROFILE" \
  --checkpoint "$BEST_CKPT" \
  --smoke-one-sample \
  --max-batches 1 \
  --verbose

qa_assert_exists "$EVAL_RUN_DIR"
qa_assert_exists "$EVAL_RUN_DIR/run_manifest.json"
qa_assert_exists "$EVAL_RUN_DIR/eval/summary.json"

echo "[PASS] ResNet-18 dev_fast smoke completed"
