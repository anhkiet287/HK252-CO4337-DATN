# Validation

## Commands Run
```bash
# Syntax sanity for training pipeline modules
python3 -m py_compile \
  scripts/train.py \
  src/ardg/training/trainer.py \
  src/ardg/training/objectives/__init__.py \
  src/ardg/training/objectives/erm.py \
  src/ardg/training/objectives/pgd_at.py \
  src/ardg/training/objectives/rex.py \
  src/ardg/training/objectives/groupdro.py \
  src/ardg/training/objectives/groupdro_plus.py \
  src/ardg/training/objectives/multi_attack_erm.py
```

## Recommended Runtime Validation Commands
```bash
# 1) Data/attack IO correctness
PYTHONPATH=src python3 scripts/preflight_check.py \
  --config configs/local/resnet50/at/multi_attack_erm.yaml --io_mode normalized \
  | tee plan/training-pipeline/evidence/01_data_io/preflight_multi_attack_normalized.txt

# 2) PGD-AT smoke
PYTHONPATH=src python3 scripts/train.py \
  --config configs/smoke_test/resnet50_local_pgd_at_smoke.yaml \
  2>&1 | tee plan/training-pipeline/evidence/02_mode_smoke/pgd_at_smoke.log

# 3) Multi-attack smoke
PYTHONPATH=src python3 scripts/train.py \
  --config configs/smoke_test/resnet50_local_multi_attack_erm_smoke.yaml \
  2>&1 | tee plan/training-pipeline/evidence/02_mode_smoke/multi_attack_smoke.log
```

## Results
- Training pipeline structure documented and mapped to code.
- Mode coverage confirmed in objective registry.

## Gaps / Risks
- Runtime verification for `rex`, `groupdro`, and `groupdro_plus` still pending in this folder.
- Need standardized short-run configs for fair smoke comparison across all modes.
