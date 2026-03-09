# Progress Log

## 2026-03-04 — In Progress
- Work completed:
  - Audited end-to-end training path from `scripts/train.py` to objective classes.
  - Mapped behavior of all implemented modes (`erm`, `pgd_at`, `multi_attack_erm`, `rex`, `groupdro`, `groupdro_plus`).
  - Documented checkpoint selection behavior and multi-attack-specific rule.
  - Added training-pipeline documentation and evidence structure.
- Files touched:
  - `plan/training-pipeline/plan.md`
  - `plan/training-pipeline/decisions.md`
  - `plan/training-pipeline/progress.md`
  - `plan/training-pipeline/validation.md`
  - `plan/training-pipeline/evidence/README.md`
- Next steps:
  - Run mode-by-mode smoke checks and store logs in `evidence/02_mode_smoke/`.
  - Add one balanced ablation table for scheduler/LR/attack strength in `evidence/06_ablation/`.
- Blockers:
  - Missing quick smoke configs for some modes; may need temporary short-epoch config copies.
