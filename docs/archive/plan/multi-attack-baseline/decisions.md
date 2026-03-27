# Decisions

## 2026-03-04
- Decision: Use `per_batch` as default domain sampling strategy.
- Rationale: Lowest overhead while preserving multi-domain training behavior.
- Alternatives considered: `split_batch`, `all_domains`.
- Impact: Faster iteration and safer default for Colab/local runs.

## 2026-03-04
- Decision: Use `val/pgd20_probe_acc` as checkpoint metric in `multi_attack_erm`.
- Rationale: Robustness-oriented selection with bounded evaluation cost.
- Alternatives considered: clean `val/acc`, full worst-domain validation.
- Impact: Better alignment with adversarial robustness objective.
