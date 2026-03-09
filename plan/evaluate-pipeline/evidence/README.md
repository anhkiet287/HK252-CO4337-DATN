# Evaluate Pipeline Evidence Guide

## Layout
- `01_clean/`: clean-only eval logs and summaries.
- `02_attack_suite/`: eval suite logs (`pgd20`, and later extended attacks).
- `03_autoattack/`: AutoAttack logs, runtime, and sample counts.
- `04_consistency/`: repeated-run consistency checks for same checkpoint/config.
- `05_visualization/`: qualitative images (clean/adv/perturbation).

## Minimum Evidence For Report
1. One clean + PGD20 eval log for baseline model.
2. One clean + PGD20 eval log for robust model.
3. Optional but recommended AutoAttack log for final robust comparison.
4. One visualization folder with `stats.json` and image grids.
