#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
. "$SCRIPT_DIR/common.sh"

cd "$REPO_ROOT"

TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/ardg-resume.XXXXXX")
trap 'rm -rf "$TMP_DIR"' EXIT HUP INT TERM

PROFILE="$TMP_DIR/resume_profile.yaml"
cat >"$PROFILE" <<EOF
_base_:
  - $REPO_ROOT/configs/profiles/dev_fast.yaml
experiment:
  seed: 84
train:
  epochs: 1
  max_batches: 1
  max_val_batches: 1
logging:
  output_dir: outputs/qa_resume
EOF

RUN_DIR=outputs/qa_resume/resnet18_dev_baseline_erm_seed84
TRAIN_CFG=configs/experiments/cifar10/resnet18/baselines/erm.yaml

qa_section "Resume setup run"
qa_run "$PYTHON_BIN" scripts/train.py \
  --config "$TRAIN_CFG" \
  --profile "$PROFILE"

qa_assert_exists "$RUN_DIR/checkpoints/last.pt"
qa_assert_exists "$RUN_DIR/wandb/run_id.txt"

qa_section "Resume from last checkpoint"
qa_run "$PYTHON_BIN" scripts/train.py \
  --config "$TRAIN_CFG" \
  --profile "$PROFILE" \
  --resume

qa_assert_exists "$RUN_DIR/run_manifest.json"
qa_assert_file_contains "$RUN_DIR/run_manifest.json" "loaded_checkpoint_path"
qa_assert_file_contains "$RUN_DIR/run_manifest.json" "wandb_run_id"

echo "[PASS] ResNet-18 resume path verified"
