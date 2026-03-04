# Decisions

## 2026-03-04
- Decision: Use class-based `Evaluator` as the reusable evaluation core.
- Rationale: Same interface can be used by scripts and future notebooks/tools.
- Alternatives: Script-local evaluation loops.
- Impact: Better reuse and lower divergence across evaluation entry points.

## 2026-03-04
- Decision: Keep attack suite construction in `build_eval_attacks(...)`.
- Rationale: One source of truth for attack params from config.
- Alternatives: Construct attacks ad hoc inside each script.
- Impact: Easier consistency and maintenance.

## 2026-03-04
- Decision: Keep AutoAttack as optional eval path.
- Rationale: AutoAttack is slower and dependency-heavy; not always needed for every run.
- Alternatives: Always run AutoAttack.
- Impact: Faster routine evals while still supporting stronger robustness checks.
