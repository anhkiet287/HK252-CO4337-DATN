#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
. "$SCRIPT_DIR/common.sh"

cd "$REPO_ROOT"

TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/ardg-vit-h100.XXXXXX")
trap 'rm -rf "$TMP_DIR"' EXIT HUP INT TERM

PROFILE="$TMP_DIR/h100_smoke.yaml"
cat >"$PROFILE" <<EOF
_base_:
  - $REPO_ROOT/configs/profiles/h100.yaml
dataset:
  num_workers: 4
train:
  epochs: 1
  batch_size: 8
  max_batches: 2
  max_val_batches: 1
  log_interval: 1
eval:
  batch_size: 8
logging:
  output_dir: outputs/qa_h100_smoke
  wandb:
    mode: offline
EOF

TRAIN_CFG=configs/experiments/cifar10/vit_b16/baselines/erm.yaml
RUN_DIR=outputs/qa_h100_smoke/vit_b16_h100_baseline_erm_seed42

qa_section "ViT-B/16 H100 smoke"
echo "[INFO] This script assumes a strong CUDA GPU and the H100 runtime profile."
qa_run "$PYTHON_BIN" scripts/train.py \
  --config "$TRAIN_CFG" \
  --profile "$PROFILE" \
  --verbose

qa_assert_exists "$RUN_DIR/checkpoints/best.pt"
qa_assert_exists "$RUN_DIR/train/summary.json"

echo "[PASS] ViT-B/16 H100 smoke completed"
