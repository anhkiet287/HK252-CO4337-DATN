# Plan Hub

This folder stores implementation plans and execution tracking per pipeline and method.

## Plan Folders
- `training-pipeline/`
- `evaluate-pipeline/`
- `ablation/`
- `multi-attack-baseline/`
- `group-dro/`
- `group-dro-plus/`

## Required Docs Per Folder
- `plan.md`: full implementation specification.
- `decisions.md`: key design decisions and rationale.
- `progress.md`: chronological implementation log.
- `validation.md`: tests/checks/results.

## Workflow
1. Draft or update `plan.md` before coding.
2. Implement and debug locally.
3. Run local verification gate before any full run:
   - `scripts/preflight_check.py`
   - smoke train (short config/epochs)
   - smoke evaluate (`scripts/evaluate.py` with reduced batches)
4. Record major decisions in `decisions.md`.
5. Update `progress.md` after each implementation session.
6. Add executed checks and outcomes in `validation.md`.
7. Run full report experiment on Colab only, with output dir:
   - `/content/drive/MyDrive/ardg/HK252-CO4337-DATN/outputs`
8. Update `INDEX.md` status.

## Where To Document What
- Correctness verification evidence: `<folder>/evidence/` + `<folder>/validation.md`
- Data exploration notes/figures: `ablation/` (cross-cutting) and method-specific `evidence/`
- Method rationale and expected behavior: `plan.md` + `decisions.md`
- Final summary numbers for report/slides: method `report.md` (for example `multi-attack-baseline/report.md`)
