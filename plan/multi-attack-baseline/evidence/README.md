# Evidence Folder Guide (Minimal)

## Layout
- `01_preflight/`: IO/epsilon sanity logs.
- `02_attack_space/`: clean/adv/perturbation images + `stats.json`.
- `03_training/`: training logs (smoke/full).
- `04_checkpoint_rule/`: extracted checkpoint metrics.
- `05_eval/`: evaluation logs/metrics.
- `06_regression/`: ERM/PGD-AT regression run logs.

## Minimal Commands
```bash
# 1) Preflight
PYTHONPATH=src python3 scripts/preflight_check.py \
  --config configs/local/resnet50/at/multi_attack_erm.yaml --io_mode normalized \
  | tee plan/multi-attack-baseline/evidence/01_preflight/preflight_normalized.txt

# 2) Smoke train
PYTHONPATH=src python3 scripts/train.py \
  --config configs/smoke_test/resnet50_local_multi_attack_erm_smoke.yaml \
  2>&1 | tee plan/multi-attack-baseline/evidence/03_training/smoke_train.log

# 3) Attack visualization (after training)
PYTHONPATH=src python3 scripts/attack_visual_check.py \
  --config configs/local/resnet50/at/multi_attack_erm.yaml \
  --checkpoint outputs/smoke_resnet50_local_multi_attack_erm/best.pt \
  --split val --attack-source train --attack from_source \
  --num-samples 16 --strict-eps \
  --output-dir plan/multi-attack-baseline/evidence/02_attack_space

# 4) Checkpoint metric extraction
PYTHONPATH=src python3 - <<'PY' \
> plan/multi-attack-baseline/evidence/04_checkpoint_rule/best_checkpoint_metrics.json
import json, torch
ckpt = torch.load("outputs/smoke_resnet50_local_multi_attack_erm/best.pt", map_location="cpu")
print(json.dumps(ckpt.get("metrics", {}), indent=2))
PY

# 5) Evaluate
PYTHONPATH=src python3 scripts/evaluate.py \
  --config configs/local/resnet50/at/multi_attack_erm.yaml --splits val,test \
  | tee plan/multi-attack-baseline/evidence/05_eval/eval_log.txt
```
