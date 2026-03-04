# Progress Log

## 2026-03-04 — In Progress
- Work completed:
  - Added minimal report template: `plan/multi-attack-baseline/report.md`.
  - Added evidence folder structure under `plan/multi-attack-baseline/evidence/`.
  - Added evidence collection guide: `plan/multi-attack-baseline/evidence/README.md`.
- Files touched:
  - `plan/multi-attack-baseline/report.md`
  - `plan/multi-attack-baseline/evidence/README.md`
  - `plan/multi-attack-baseline/evidence/*/.gitkeep`
- Next steps:
  - Run minimal command set in evidence guide and fill report table values.
- Blockers:
  - None.

## 2026-03-04 — In Progress
- Work completed:
  - Added `multi_attack_erm` objective and registry hook.
  - Added `build_attack(...)` API in attack suite.
  - Extended FGSM (RS-FGSM path) and PGD (`ce`/`dlr`) wrappers.
  - Added objective validation hook + trainer probe metric merge/checkpoint selection.
  - Added local/colab/smoke configs for ResNet50 multi-attack.
- Files touched:
  - `src/ardg/training/objectives/multi_attack_erm.py`
  - `src/ardg/training/objectives/__init__.py`
  - `src/ardg/training/objectives/base.py`
  - `src/ardg/training/trainer.py`
  - `src/ardg/attacks/attack_suite.py`
  - `src/ardg/attacks/fgsm.py`
  - `src/ardg/attacks/pgd.py`
  - `configs/local/resnet50/at/multi_attack_erm.yaml`
  - `configs/colab/resnet50/at/multi_attack_erm.yaml`
  - `configs/smoke_test/resnet50_local_multi_attack_erm_smoke.yaml`
- Next steps:
  - Run end-to-end smoke and full train on target environment.
  - Validate robust metrics trend and checkpoint behavior.
- Blockers:
  - Runtime environment must have `torchattacks` and training dependencies installed.
