# Evaluate Pipeline Analysis and Plan

## Summary
This document defines the current evaluation pipeline and the target documentation/testing standard for robust evaluation across all training modes.

Core goal: one consistent evaluation interface for all models/checkpoints using:
- clean metrics
- configured attack suite metrics
- optional AutoAttack metrics

## Why This Pipeline Exists
1. Use one evaluation script and one evaluator class across methods/models.
2. Keep attack construction centralized and config-driven.
3. Ensure robust metrics are reproducible and comparable across experiments.

## Current Architecture (What We Have)

### 1) Entry and Wiring
1. `scripts/evaluate.py`: loads run config/checkpoint and evaluates selected splits (`val`, `test`).
2. `src/ardg/experiments/common.py`: shared setup, dataloaders, checkpoint loading.

### 2) Evaluator Core
1. `src/ardg/evaluation/evaluator.py`:
   - `Evaluator(model, loader, device, attack_suite=None, max_batches=0)`
   - `evaluate_clean()`
   - `evaluate_under_attack()`
   - `evaluate_suite()`
   - `evaluate_all()`
2. Backward-compatible function wrappers still exist for old call sites.

### 3) Attack Suite Integration
1. `src/ardg/attacks/attack_suite.py::build_eval_attacks` builds evaluation attacks from `attack.eval`.
2. `scripts/evaluate.py` currently requires `pgd20` in eval suite.
3. AutoAttack path is separate (`src/ardg/attacks/autoattack.py`) and controlled by `attack.autoattack.enabled`.

## Current Runtime Behavior
1. Resolve checkpoint (`best.pt` then `last.pt`) unless explicit `--checkpoint`.
2. Evaluate clean accuracy on each requested split.
3. Evaluate PGD20 attack from eval suite.
4. Optionally evaluate AutoAttack.
5. Log `acc_clean`, `acc_pgd20`, optional `acc_aa`, and `worst_acc`.

## What We Want Next (Target State)
1. Multi-attack eval suite support beyond mandatory `pgd20` gate.
2. Optional fixed-batch quick eval mode in `scripts/evaluate.py` for faster sanity checks.
3. Unified JSON export schema for report ingestion.
4. Strong consistency checks:
   - attack eps budget correctness
   - clean-vs-robust monotonicity expectations
   - deterministic split/eval protocol.

## Gaps and Risks
1. `scripts/evaluate.py` hard-requires `pgd20`; this is strict but can block some configs.
2. AutoAttack runtime can be expensive and may be skipped in practice.
3. No single script currently produces a full ablation-ready evaluation table directly.

## Evaluation Verification Scope
1. Correctness:
   - clean eval sanity
   - attack eval runs with expected threat model parameters.
2. Stability:
   - no NaN/inf metrics
   - sample counts are consistent across clean and attacked eval.
3. Reproducibility:
   - same checkpoint + config should reproduce metrics within expected variance.

## Evidence Layout
Use `plan/evaluate-pipeline/evidence/`:
1. `01_clean/`: clean evaluation logs.
2. `02_attack_suite/`: PGD suite logs and summaries.
3. `03_autoattack/`: AutoAttack logs and runtime.
4. `04_consistency/`: cross-run or cross-script consistency checks.
5. `05_visualization/`: qualitative attack visuals used in reports.

## Pipeline Flow (Text)
1. Config + checkpoint resolution.
2. Build loaders (`val`, `test`).
3. Build eval attacks from config.
4. Per split:
   - clean metrics via `Evaluator.evaluate_clean()`
   - attack metrics via `Evaluator.evaluate_suite(...)`
   - optional AutoAttack.
5. Log split metrics and system/runtime metadata.
