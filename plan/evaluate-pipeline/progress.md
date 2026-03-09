# Progress Log

## 2026-03-04 — In Progress
- Work completed:
  - Audited evaluation flow from `scripts/evaluate.py` through `Evaluator` and attack builders.
  - Confirmed unified evaluator class API is now used by the main evaluate script.
  - Documented current requirements and gaps (PGD20 gate, optional AutoAttack).
  - Added evaluate-pipeline planning and evidence structure.
- Files touched:
  - `plan/evaluate-pipeline/plan.md`
  - `plan/evaluate-pipeline/decisions.md`
  - `plan/evaluate-pipeline/progress.md`
  - `plan/evaluate-pipeline/validation.md`
  - `plan/evaluate-pipeline/evidence/README.md`
- Next steps:
  - Collect one clean+PGD20 eval log and one AutoAttack eval log in this folder.
  - Add consistency check artifact comparing repeated eval runs on same checkpoint.
- Blockers:
  - Requires available trained checkpoints for each mode.
