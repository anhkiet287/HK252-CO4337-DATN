# Ablation Hub

This folder tracks cross-method ablation design and evidence.

## Purpose
1. Define comparable ablation experiments.
2. Keep one matrix for planned/running/completed ablations.
3. Store result artifacts used in report tables and figures.

## Files
- `matrix.md`: master ablation matrix.
- `evidence/`: logs, csv/json summaries, and plots.

## Rules
1. Change one factor at a time per ablation row.
2. Keep fixed:
   - dataset split/seed
   - model backbone
   - eval protocol (clean + PGD20, optional AA)
3. Record:
   - exact config path
   - checkpoint path
   - clean and robust metrics
   - runtime/GPU context
