# Training Pipeline Analysis and Plan

## Summary
This document describes the training pipeline in the current repo: what it does, why it is structured this way, what is already implemented, and what remains to improve.

Primary goal: keep one stable training stack for all implemented modes:
- `erm`
- `pgd_at`
- `multi_attack_erm`
- `rex`
- `groupdro`
- `groupdro_plus`

## Why This Pipeline Exists
1. Reuse one trainer loop across methods.
2. Keep method-specific logic isolated in objective classes.
3. Keep experiment behavior mostly config-driven.
4. Preserve reproducibility (seeded split + fixed config + checkpoints).

## Current Architecture (What We Have)

### 1) Entry and Wiring
1. `scripts/train.py`: loads config, sets up runtime, builds loaders/model, runs `Trainer`.
2. `src/ardg/experiments/common.py`: resolves platform/device/output/wandb and creates dataloaders.

### 2) Data Pipeline
1. `src/ardg/data/datasets.py`: dataset loading, stratified split usage, optional max sample caps.
2. `src/ardg/data/transforms.py`: train/eval transforms and normalization stats.

### 3) Model Pipeline
1. `src/ardg/models/factory.py`: config-to-model dispatch (`resnet18_cifar`, `resnet50_cifar`, `vit_b16_cifar`).
2. `src/ardg/models/resnet.py`: CIFAR stem for ResNet-18/50 (3x3 stride-1, no maxpool).

### 4) Training Core
1. `src/ardg/training/trainer.py`: optimizer/scheduler, epoch loop, logging, validation, checkpointing, resume, early stopping.
2. Objective dispatch:
   - `src/ardg/training/objectives/__init__.py`
   - Objective implementations under `src/ardg/training/objectives/`.

### 5) Mode-Specific Logic
1. `erm.py`: clean CE training.
2. `pgd_at.py`: replaces batch with adversarial examples from `attack.train`.
3. `multi_attack_erm.py`: attacks-as-domains (`per_batch`, `split_batch`, `all_domains`) + optional PGD probe validation.
4. `rex.py`: pseudo-env split + variance penalty.
5. `groupdro.py`: multiplicative `q` updates over groups.
6. `groupdro_plus.py`: pseudo-groups via batch-wise clustering + weighted objective.

### 6) Attack Building for Training
1. `src/ardg/attacks/attack_suite.py`: attack factory + normalization/threat-model checks.
2. `src/ardg/attacks/pgd.py`: PGD/UPGD (`ce` or `dlr`) + restarts.
3. `src/ardg/attacks/fgsm.py`: FGSM / RS-FGSM.
4. `src/ardg/attacks/torchattacks_utils.py`: normalization bridge (`set_normalization_used`).

## Selection and Checkpoint Behavior
1. Most modes: best checkpoint selected by `val/acc`.
2. `multi_attack_erm`: best checkpoint selected by `val/pgd20_probe_acc` when present, otherwise fallback `val/acc`.

## What We Want Next (Target State)
1. A documented correctness contract for each mode:
   - attack-space constraints
   - mode-specific metric expectations
   - regression requirements.
2. Comparable smoke settings for all modes (currently only some modes have dedicated smoke configs).
3. Stronger evidence storage discipline per run for report writing.
4. Clear ablation protocol (same data split, same schedule family, single-factor changes).

## Gaps and Risks
1. Not all modes have dedicated smoke configs with equal budget.
2. `split_batch` in `multi_attack_erm` is currently random per sample, not exact-balanced.
3. Group-based modes currently rely on fallback semantics if explicit group labels are absent.
4. Training README and plan docs were previously fragmented; now consolidated in this folder.

## Training Verification Scope
1. Data/normalization correctness: `scripts/preflight_check.py`.
2. Attack budget correctness: `linf(pixel) <= eps` checks and visualization.
3. Mode behavior correctness:
   - expected log keys appear
   - checkpoints use expected selection metric
   - no regression in `erm` and `pgd_at`.
4. Minimal convergence sanity:
   - smoke loss should not diverge to NaN/inf
   - clean accuracy should exceed chance in non-trivial runs.

## Evidence Layout
Use `plan/training-pipeline/evidence/`:
1. `01_data_io/`
2. `02_mode_smoke/`
3. `03_mode_full/`
4. `04_correctness/`
5. `05_regression/`
6. `06_ablation/`

## Pipeline Flow (Text)
1. Config -> `setup_run()` -> seed/device/output/wandb.
2. Config -> dataloaders + model.
3. Config -> objective selection by `train.mode`.
4. Trainer loop:
   - objective preprocess (optional attack/domain transform)
   - forward/loss/backward/step
   - epoch validation (clean + objective hook extras)
   - checkpoint selection and save.
5. Run artifacts -> output run dir + evidence copy into plan folder.
