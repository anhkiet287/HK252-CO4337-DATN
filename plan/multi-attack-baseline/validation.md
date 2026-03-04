# Validation

## Commands Run
```bash
python3 -m compileall -q src scripts
```

## Results
- Compile/syntax check passed for modified Python files.

## Minimal Evidence Plan (To Run)
```bash
PYTHONPATH=src python3 scripts/preflight_check.py \
  --config configs/local/resnet50/at/multi_attack_erm.yaml --io_mode normalized \
  | tee plan/multi-attack-baseline/evidence/01_preflight/preflight_normalized.txt

PYTHONPATH=src python3 scripts/train.py \
  --config configs/smoke_test/resnet50_local_multi_attack_erm_smoke.yaml \
  2>&1 | tee plan/multi-attack-baseline/evidence/03_training/smoke_train.log
```

## Expected Artifacts
- `plan/multi-attack-baseline/evidence/01_preflight/preflight_normalized.txt`
- `plan/multi-attack-baseline/evidence/02_attack_space/*`
- `plan/multi-attack-baseline/evidence/03_training/smoke_train.log`
- `plan/multi-attack-baseline/evidence/04_checkpoint_rule/best_checkpoint_metrics.json`
- `plan/multi-attack-baseline/evidence/05_eval/eval_log.txt`

## Gaps / Risks
- Runtime training validation still required in full environment.
- `torchattacks` and dataset/runtime dependencies must be installed where training is executed.
