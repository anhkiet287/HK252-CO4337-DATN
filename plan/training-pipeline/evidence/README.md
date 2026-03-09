# Training Pipeline Evidence Guide

## Layout
- `01_data_io/`: preflight and normalization/attack-space checks.
- `02_mode_smoke/`: short smoke logs per mode.
- `03_mode_full/`: full run logs or summaries.
- `04_correctness/`: explicit correctness artifacts (metric extraction, checkpoint metadata).
- `05_regression/`: regression logs for baseline modes.
- `06_ablation/`: ablation tables/charts related to training choices.

## Minimum Evidence For Report
1. One preflight log proving IO/epsilon consistency.
2. Smoke logs for at least:
   - `pgd_at`
   - `multi_attack_erm`
3. One regression log for `erm` unchanged behavior.
4. One checkpoint metadata extract proving expected selection metric.

## Naming Convention
- Use `<mode>_<scope>_<date>.log` when possible.
- Keep one JSON summary per run for tables.
