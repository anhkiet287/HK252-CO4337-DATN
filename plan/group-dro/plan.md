# GroupDRO Implementation Plan

## Summary
Track and improve the GroupDRO training path (`train.mode: groupdro`) with clear group semantics, robust validation, and reproducible configs.

## Public Interface Changes
1. Keep `train.mode: groupdro`.
2. Use `train.groupdro.eta` as core GroupDRO hyperparameter.
3. Optionally support explicit group labels (`g`) in batches when available.
4. Keep backward compatibility by falling back to label-as-group when `g` is absent.

## Implementation Scope
1. Review `src/ardg/training/objectives/groupdro.py` for group handling and `q` updates.
2. Define group data source strategy (explicit `g` vs fallback labels).
3. Add/refresh configs under `configs/local/.../groupdro.yaml` and `configs/colab/.../groupdro.yaml`.
4. Add smoke config for GroupDRO behavior sanity.
5. Add robust validation/probe policy if needed (aligned with trainer architecture).

## Runtime Behavior
1. Optional adversarial preprocess when `train.adv_training=true`.
2. Compute per-group loss and update group weights `q` multiplicatively.
3. Optimize weighted robust objective.

## Test Cases and Scenarios
1. GroupDRO runs with and without explicit group IDs.
2. `q` remains normalized and finite.
3. Smoke run completes and logs `q_max`/`q_min`.
4. Regression: ERM/PGD-AT unaffected.

## Pipeline and Workflow Summary
1. Data loader -> batch (`x`,`y`, optional `g`).
2. Objective preprocess (optional attack).
3. Per-group loss + `q` update.
4. Weighted objective optimization.
5. Validation + checkpointing.

## Assumptions and Defaults
1. If no `g`, fallback to labels as groups.
2. `eta` default from config.
3. Attack backend remains `torchattacks` for adversarial branch.
