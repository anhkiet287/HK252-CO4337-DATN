#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
. "$SCRIPT_DIR/common.sh"

cd "$REPO_ROOT"

TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/ardg-wandb.XXXXXX")
trap 'rm -rf "$TMP_DIR"' EXIT HUP INT TERM

DISABLE_PROFILE="$TMP_DIR/disable_wandb.yaml"
cat >"$DISABLE_PROFILE" <<EOF
_base_:
  - $REPO_ROOT/configs/profiles/dev_fast.yaml
logging:
  output_dir: outputs/qa_wandb_policy
  wandb:
    enabled: false
EOF

TRAIN_CFG=configs/experiments/cifar10/resnet18/baselines/erm.yaml
EVAL_CFG=configs/experiments/cifar10/resnet18/eval/baseline_erm_all_attacks.yaml

qa_section "W&B offline policy"
qa_run "$PYTHON_BIN" scripts/preflight_check.py \
  --config "$TRAIN_CFG" \
  --profile configs/profiles/dev_fast.yaml \
  --io_mode normalized \
  --check-wandb

qa_section "W&B online policy"
qa_run "$PYTHON_BIN" scripts/preflight_check.py \
  --config "$TRAIN_CFG" \
  --profile configs/profiles/local_gpu.yaml \
  --io_mode normalized \
  --check-wandb

qa_section "W&B disabled must fail for train"
qa_expect_fail_contains \
  "$TMP_DIR/train_disabled.log" \
  "W&B logging is mandatory" \
  "$PYTHON_BIN" scripts/train.py \
  --config "$TRAIN_CFG" \
  --profile "$DISABLE_PROFILE"

qa_section "W&B disabled must fail for eval"
qa_expect_fail_contains \
  "$TMP_DIR/eval_disabled.log" \
  "W&B logging is mandatory" \
  "$PYTHON_BIN" scripts/evaluate.py \
  --config "$EVAL_CFG" \
  --profile "$DISABLE_PROFILE" \
  --checkpoint "$TMP_DIR/nonexistent.pt"

echo "[PASS] W&B policy checks completed"
