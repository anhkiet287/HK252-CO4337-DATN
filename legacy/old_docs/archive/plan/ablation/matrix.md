# Ablation Matrix

| ID | Purpose | Base Config | Changed Factor | Values | Status | Evidence Path | Notes |
|---|---|---|---|---|---|---|---|
| A1 | Training strategy in multi-attack | `configs/local/resnet50/train/multi_attack_erm.yaml` | `train.multi_attack.strategy` | `per_batch`, `split_batch`, `all_domains` | Planned | `plan/training-pipeline/evidence/06_ablation/` | Compare cost vs robust metrics |
| A2 | Multi-attack domain composition | `configs/local/resnet50/train/multi_attack_erm.yaml` | `attack.multi_train.domains` | with/without `pgd_dlr` | Planned | `plan/training-pipeline/evidence/06_ablation/` | Test domain diversity effect |
| A3 | Probe strength for checkpointing | `configs/local/resnet50/train/multi_attack_erm.yaml` | `val.probe.steps` | `10`, `20` | Planned | `plan/evaluate-pipeline/evidence/04_consistency/` | Trade-off compute vs selection quality |
| A4 | Adversarial training baseline strength | `configs/local/resnet50/train/pgd_at.yaml` | `attack.train.num_steps` | `7`, `10`, `20` | Planned | `plan/training-pipeline/evidence/06_ablation/` | Robustness sensitivity |
| A5 | Objective comparison | mode-specific configs | `train.mode` | `erm`, `pgd_at`, `multi_attack_erm`, `rex`, `groupdro`, `groupdro_plus` | Planned | `plan/ablation/evidence/` | Main report comparison |

## Required Metrics Per Row
1. `val/acc`
2. `val/pgd20_probe_acc` (if available during train)
3. Test `acc_clean`
4. Test `acc_pgd20`
5. Test `acc_aa` (optional)
6. Runtime (train and eval)
