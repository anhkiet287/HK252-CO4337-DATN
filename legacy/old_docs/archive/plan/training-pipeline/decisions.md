# Decisions

## 2026-03-04
- Decision: Keep one shared `Trainer` and implement mode differences inside objectives.
- Rationale: Reduces duplication and keeps checkpoint/logging behavior consistent.
- Alternatives: Separate trainer per mode.
- Impact: Easier maintenance; objective interfaces become critical.

## 2026-03-04
- Decision: Keep attack-generation inside objective `preprocess_batch`.
- Rationale: Explicitly couples training mode and attack logic while keeping trainer generic.
- Alternatives: Attack generation directly in trainer.
- Impact: Cleaner trainer, mode logic isolated.

## 2026-03-04
- Decision: `multi_attack_erm` best checkpoint uses `val/pgd20_probe_acc` when available.
- Rationale: Better alignment with robust objective than clean-only checkpointing.
- Alternatives: Always `val/acc`.
- Impact: More robust model selection with bounded validation cost.

## 2026-03-04
- Decision: Use plan evidence folders for correctness and regression artifacts.
- Rationale: Team reporting requires reproducible files, not only terminal logs.
- Alternatives: Keep only wandb/console logs.
- Impact: Better traceability for reports and ablation write-up.
