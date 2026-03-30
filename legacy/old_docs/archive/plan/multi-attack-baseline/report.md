# Multi-Attack Baseline Report (Minimal)

## 1) Goal
Show that `multi_attack_erm` is implemented correctly and behaves as intended.

## 2) Correctness Evidence (Required)
1. Attack-space correctness:
   - `linf_pixel_max <= eps + 1e-3`
   - Source: `evidence/01_preflight/`, `evidence/02_attack_space/stats.json`
2. Domain mechanism correctness:
   - `train/domain_count/*` and `train/domain_name` appear in logs
   - Source: `evidence/03_training/smoke_train.log`
3. Checkpoint rule correctness:
   - `selection_metric_name == pgd20_probe_acc` in `best.pt` metrics
   - Source: `evidence/04_checkpoint_rule/best_checkpoint_metrics.json`
4. Robust validation hook correctness:
   - `val/pgd20_probe_acc`, `val/pgd20_probe_loss` logged
   - Source: `evidence/03_training/smoke_train.log`

## 3) Key Numbers
| Metric | Value | Evidence |
|---|---:|---|
| `linf_pixel_max` | TODO | `evidence/02_attack_space/stats.json` |
| `eps_config` | TODO | `evidence/02_attack_space/stats.json` |
| `val/pgd20_probe_acc` (best) | TODO | `evidence/03_training/smoke_train.log` |
| `val/acc` (best) | TODO | `evidence/03_training/smoke_train.log` |
| `selection_metric_name` | TODO | `evidence/04_checkpoint_rule/best_checkpoint_metrics.json` |

## 4) Figures (Images)
Use these in slides/report:
1. `evidence/02_attack_space/clean_pixel.png`
2. `evidence/02_attack_space/adv_pixel.png`
3. `evidence/02_attack_space/perturbation_pixel_signed.png`
4. `evidence/02_attack_space/clean_normalized_scaled.png`
5. `evidence/02_attack_space/adv_normalized_scaled.png`
6. `evidence/02_attack_space/perturbation_normalized_signed.png`

## 5) Regression Check
Run ERM and PGD-AT smoke to confirm no breakage and store logs under:
- `evidence/06_regression/`
