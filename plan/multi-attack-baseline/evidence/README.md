# Evidence Folder Guide (Minimal)

## Scope
Only keep evidence for these 6 checks:
1. normalized pipeline
2. attack space
3. split correctness (`train/val/test`)
4. batch size correctness
5. domain generation logic per batch
6. deterministic behavior

## Layout
- `01_preflight/`: normalization sanity logs.
- `02_attack_space/`: clean/adv/perturbation + `stats.json`.
- `03_split/`: split size/batch count snapshot.
- `03_training/`: smoke training log (batch size + domain logic evidence).
- `06_deterministic/`: repeated eval jsons for reproducibility check.

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

# 4) Split snapshot
PYTHONPATH=src python3 - <<'PY' > plan/multi-attack-baseline/evidence/03_split/split_sizes.txt
from ardg.config import load_config
from ardg.experiments.common import build_loaders
cfg = load_config("configs/local/resnet50/at/multi_attack_erm.yaml")
tr, va, te = build_loaders(cfg)
print({"train_batches": len(tr), "val_batches": len(va), "test_batches": len(te)})
print({"train_samples": len(tr.dataset), "val_samples": len(va.dataset), "test_samples": len(te.dataset)})
PY

# 5) Deterministic check (same run twice)
WANDB_MODE=disabled PYTHONPATH=src python3 scripts/evaluate.py --config configs/local/resnet18/at/groupdro_plus.yaml --checkpoint models/ta50.tar --smoke-one-sample --seed 42 --deterministic --save-json plan/multi-attack-baseline/evidence/06_deterministic/eval_run1.json
WANDB_MODE=disabled PYTHONPATH=src python3 scripts/evaluate.py --config configs/local/resnet18/at/groupdro_plus.yaml --checkpoint models/ta50.tar --smoke-one-sample --seed 42 --deterministic --save-json plan/multi-attack-baseline/evidence/06_deterministic/eval_run2.json
```
