# ARDG Implementation Progress

Last updated: 2026-03-03 (local)
Branch: `kiet-dev`

## Scope
Delivery tracked against the agreed checklist:
- P0 (must): `T0`..`T4`
- P1 (nice): `T5`, `T6`

## Status Summary

### P0 — MUST
- [x] `T0` Replace AutoAttack placeholder with real integration.
- [x] `T0` Fail-fast error message when AutoAttack is missing.
- [x] `T0` Verified real AutoAttack run on local env (`autoattack` installed, metrics logged).

- [x] `T1` Evaluate logs required W&B metrics only (no JSON/CSV artifacts):
  - `test/acc_clean`
  - `test/acc_pgd20`
  - `test/acc_aa` (if enabled)
  - `test/worst_acc`
  - `sys/runtime_sec`
  - `sys/n_samples_eval`
  - `sys/aa_runtime_sec`, `sys/aa_n_samples` (if AA enabled)
  - optional: `sys/gpu_name`, `sys/torch_version`

- [x] `T2` PGD eval standardized to PGD-Linf `(steps=20, restarts=5)` via `attack.eval.pgd20`.

- [x] `T3` Added `scripts/preflight_check.py` with both modes:
  - `--io_mode pixel`
  - `--io_mode normalized`
  - Includes input range, clamp check, PGD small-steps epsilon-bound check, forward check.
- [x] `T3` Ran both preflight modes successfully.

- [x] `T4` Smoke test switched to `Trainer1` path.
- [x] `T4` Smoke defaults reduced to 2 quick configs.
- [x] `T4` Smoke run completed successfully.

### P1 — NICE
- [x] `T5` Added minimal eval config: `configs/eval/cifar10_stage1_min.yaml`.
- [x] `T6` Repo hygiene (short-term) started:
  - `.gitignore` updated to ignore `runs/` and `models/*.tar`.
  - asset handling note added in `docs/ASSETS.md`.

## Validation Log (Executed)

1. Train (`test_local`)
- Command: `python scripts/train.py --config configs/test_local.yaml`
- Result: completed; W&B run synced; checkpoints produced (`best.pt`, `last.pt`).
- Runtime device: CUDA confirmed (`train/device: cuda` in logs).

2. Evaluate without AA
- Command: `python scripts/evaluate.py --config configs/test_local.yaml`
- Result: completed; logged `test/acc_clean`, `test/acc_pgd20`, `test/worst_acc`, `sys/*`.

3. AutoAttack installation
- Command: `pip install git+https://github.com/fra31/auto-attack.git`
- Result: installed (`autoattack-0.1`).

4. Evaluate with AA enabled
- Command: `python scripts/evaluate.py --config configs/test_local.yaml`
- Result: completed; logged `test/acc_aa`, `sys/aa_runtime_sec`, `sys/aa_n_samples`.

5. Preflight checks
- Commands:
  - `python scripts/preflight_check.py --config configs/test_local.yaml --io_mode pixel`
  - `python scripts/preflight_check.py --config configs/test_local.yaml --io_mode normalized`
- Result: both PASS.

6. Smoke
- Command: `python scripts/smoke_test.py`
- Result: PASS for 2 quick configs on `Trainer1`, checkpoints saved under `outputs/smoke_local/*`.

## Key Code Changes

- `src/ardg/attacks/autoattack.py`
  - Real AutoAttack execution + runtime/sample metrics + fail-fast guidance.
- `src/ardg/attacks/pgd.py`
  - Added restart support, normalized-space clamp to valid pixel bounds, best-of-restarts.
- `src/ardg/attacks/attack_suite.py`
  - Canonical eval attack renamed/standardized to `pgd20`.
- `scripts/evaluate.py`
  - Minimal W&B metric logging with `worst_acc` and `sys/*`.
- `scripts/preflight_check.py` (new)
  - Fast 1-batch IO/attack sanity checks.
- `src/ardg/experiments/smoke.py`
  - Migrated smoke flow to `Trainer1`.
- `scripts/smoke_test.py`
  - Defaults now 2 quick smoke configs.
- `configs/default.yaml`
  - Added `attack.eval.enabled` + `attack.eval.pgd20` section.
- `configs/eval/cifar10_stage1_min.yaml` (new)
  - Minimal stage-1 eval config.
- `configs/smoke_quick_resnet18_erm.yaml` (new)
- `configs/smoke_quick_resnet18_pgd_at.yaml` (new)
- `pyproject.toml`
  - Added optional extra: `autoattack`.
- `.gitignore`
  - Added `runs/`, `models/*.tar`.

## Notes / Known Non-blockers

- `torchvision` emits `VisibleDeprecationWarning` with NumPy 2.4 in CIFAR loader; this does not block train/eval.
- Unimplemented attack wrappers (FGSM/CW/DeepFool/FAB/Square) remain intentionally out-of-scope for this sprint.
