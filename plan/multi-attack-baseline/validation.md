# Validation

## Evidence Scope (Only 6 Checks)
1. Normalization is correct.
2. Attack space is correct.
3. Data splits (`train/val/test`) are correct.
4. Batch size behavior is correct.
5. Domain generation per batch matches strategy logic.
6. Deterministic behavior is correct.

## Commands Run
```bash
python3 -m compileall -q src scripts
```

## Results
- Compile/syntax check passed for modified Python files.

## Minimal Evidence Commands (To Run)
```bash
# 1) Normalization
PYTHONPATH=src python3 scripts/preflight_check.py \
  --config configs/local/resnet50/train/multi_attack_erm.yaml --io_mode normalized \
  | tee plan/multi-attack-baseline/evidence/01_preflight/preflight_normalized.txt

# 2) Smoke train log (used for checks 4 and 5)
PYTHONPATH=src python3 scripts/train.py \
  --config configs/smoke_test/resnet50_local_multi_attack_erm_smoke.yaml \
  2>&1 | tee plan/multi-attack-baseline/evidence/03_training/smoke_train.log

# 3) Attack space visualization
PYTHONPATH=src python3 scripts/attack_visual_check.py \
  --config configs/local/resnet50/train/multi_attack_erm.yaml \
  --checkpoint outputs/smoke_resnet50_local_multi_attack_erm/best.pt \
  --split val --attack-source train --attack from_source --num-samples 8 \
  --strict-eps --output-dir plan/multi-attack-baseline/evidence/02_attack_space

# 4) Split sizes (train/val/test)
PYTHONPATH=src python3 - <<'PY' > plan/multi-attack-baseline/evidence/03_split/split_sizes.txt
from ardg.config import load_config
from ardg.experiments.common import build_loaders
cfg = load_config("configs/local/resnet50/train/multi_attack_erm.yaml")
tr, va, te = build_loaders(cfg)
print({"train_batches": len(tr), "val_batches": len(va), "test_batches": len(te)})
print({"train_samples": len(tr.dataset), "val_samples": len(va.dataset), "test_samples": len(te.dataset)})
PY

# 5) Deterministic check (same command twice, compare json)
WANDB_MODE=disabled PYTHONPATH=src python3 scripts/evaluate.py --config configs/local/resnet18/at/groupdro_plus.yaml --checkpoint models/ta50.tar --smoke-one-sample --seed 42 --deterministic --save-json plan/multi-attack-baseline/evidence/06_deterministic/eval_run1.json
WANDB_MODE=disabled PYTHONPATH=src python3 scripts/evaluate.py --config configs/local/resnet18/at/groupdro_plus.yaml --checkpoint models/ta50.tar --smoke-one-sample --seed 42 --deterministic --save-json plan/multi-attack-baseline/evidence/06_deterministic/eval_run2.json
```

## How To Read Evidence
1. Normalization:
- `01_preflight/preflight_normalized.txt` must show roundtrip/clamp checks pass.

2. Attack space:
- `02_attack_space/stats.json` must show `linf_pixel_max <= eps` for Linf attacks.

3. Split correctness:
- `03_split/split_sizes.txt` must contain valid non-empty `train/val/test` sizes.

4. Batch size correctness:
- `03_training/smoke_train.log` should show consistent step-level batch behavior with configured smoke batch size.

5. Domain-per-batch logic:
- `03_training/smoke_train.log` should include `train/domain_name` and `train/domain_count/*` consistent with chosen strategy.

6. Determinism:
- `06_deterministic/eval_run1.json` and `eval_run2.json` should match on key metrics (`clean.acc`, `robust[*].acc`).
```

## Expected Artifacts
- `plan/multi-attack-baseline/evidence/01_preflight/preflight_normalized.txt`
- `plan/multi-attack-baseline/evidence/02_attack_space/*`
- `plan/multi-attack-baseline/evidence/03_split/split_sizes.txt`
- `plan/multi-attack-baseline/evidence/03_training/smoke_train.log`
- `plan/multi-attack-baseline/evidence/06_deterministic/eval_run1.json`
- `plan/multi-attack-baseline/evidence/06_deterministic/eval_run2.json`

## Gaps / Risks
- Runtime commands still need to be executed in your environment to populate artifacts.
