# Latest Paths

This file is the human-readable companion to `artifacts/latest/manifest.yaml`.

## Canonical Configs
- `default`: `configs/default.yaml`
- `resnet18_train`: `configs/experiments/cifar10/resnet18/baselines/erm.yaml`
- `resnet18_eval`: `configs/experiments/cifar10/resnet18/eval/baseline_erm_all_attacks.yaml`
- `resnet50_train`: `configs/experiments/cifar10/resnet50/baselines/erm.yaml`
- `resnet50_eval`: `configs/experiments/cifar10/resnet50/eval/baseline_erm_all_attacks.yaml`
- `vit_b16_train`: `configs/experiments/cifar10/vit_b16/baselines/erm.yaml`
- `vit_b16_eval`: `configs/experiments/cifar10/vit_b16/eval/baseline_erm_all_attacks.yaml`

## Canonical Profiles
- `dev_fast`: `configs/profiles/dev_fast.yaml`
- `local_gpu`: `configs/profiles/local_gpu.yaml`
- `colab_gpu`: `configs/profiles/colab_gpu.yaml`
- `h100`: `configs/profiles/h100.yaml`

## Verification Scripts
- `wandb_policy`: `scripts/qa/test_wandb_policy.sh`
- `resnet18_dev_fast`: `scripts/qa/smoke_resnet18_dev_fast.sh`
- `resume_resnet18`: `scripts/qa/test_resume_resnet18.sh`
- `resnet50_smoke`: `scripts/qa/smoke_resnet50.sh`
- `vit_h100_smoke`: `scripts/qa/smoke_vit_h100.sh`
- `preflight_local`: `scripts/qa/preflight_local.sh`
- `preflight_h100`: `scripts/qa/preflight_h100.sh`
- `full_verify_local`: `scripts/qa/full_verify_local.sh`

## Latest resnet18
### train
- `run_name`: `resnet18_dev_ablation_groupdro_pgd_linf_pgd_l2_seed42`
- `run_dir`: `/mnt/c/Users/ADMIN/Github/HCMUT/HK252-CO4337-DATN/outputs/dev_fast/resnet18_dev_ablation_groupdro_pgd_linf_pgd_l2_seed42`
- `resolved_config`: `/mnt/c/Users/ADMIN/Github/HCMUT/HK252-CO4337-DATN/outputs/dev_fast/resnet18_dev_ablation_groupdro_pgd_linf_pgd_l2_seed42/resolved_config.yaml`
- `run_manifest`: `/mnt/c/Users/ADMIN/Github/HCMUT/HK252-CO4337-DATN/outputs/dev_fast/resnet18_dev_ablation_groupdro_pgd_linf_pgd_l2_seed42/run_manifest.json`
- `config_path`: `/mnt/c/Users/ADMIN/Github/HCMUT/HK252-CO4337-DATN/configs/experiments/cifar10/resnet18/ablations/groupdro_pgd_linf_pgd_l2.yaml`
- `profile_path`: `/mnt/c/Users/ADMIN/Github/HCMUT/HK252-CO4337-DATN/configs/profiles/dev_fast.yaml`
- `wandb_run_id`: `87fs5nec`
- `best_checkpoint`: `/mnt/c/Users/ADMIN/Github/HCMUT/HK252-CO4337-DATN/outputs/dev_fast/resnet18_dev_ablation_groupdro_pgd_linf_pgd_l2_seed42/checkpoints/best.pt`
- `last_checkpoint`: `/mnt/c/Users/ADMIN/Github/HCMUT/HK252-CO4337-DATN/outputs/dev_fast/resnet18_dev_ablation_groupdro_pgd_linf_pgd_l2_seed42/checkpoints/last.pt`
- `train_summary_json`: `/mnt/c/Users/ADMIN/Github/HCMUT/HK252-CO4337-DATN/outputs/dev_fast/resnet18_dev_ablation_groupdro_pgd_linf_pgd_l2_seed42/train/summary.json`
### eval
- `run_name`: `resnet18_dev_ablation_groupdro_pgd_linf_pgd_l2_seed42_eval`
- `run_dir`: `/mnt/c/Users/ADMIN/Github/HCMUT/HK252-CO4337-DATN/outputs/dev_fast/resnet18_dev_ablation_groupdro_pgd_linf_pgd_l2_seed42_eval`
- `resolved_config`: `/mnt/c/Users/ADMIN/Github/HCMUT/HK252-CO4337-DATN/outputs/dev_fast/resnet18_dev_ablation_groupdro_pgd_linf_pgd_l2_seed42_eval/resolved_config.yaml`
- `run_manifest`: `/mnt/c/Users/ADMIN/Github/HCMUT/HK252-CO4337-DATN/outputs/dev_fast/resnet18_dev_ablation_groupdro_pgd_linf_pgd_l2_seed42_eval/run_manifest.json`
- `config_path`: `/mnt/c/Users/ADMIN/Github/HCMUT/HK252-CO4337-DATN/configs/experiments/cifar10/resnet18/eval/groupdro_pgd_linf_pgd_l2.yaml`
- `profile_path`: `/mnt/c/Users/ADMIN/Github/HCMUT/HK252-CO4337-DATN/configs/profiles/dev_fast.yaml`
- `wandb_run_id`: `zjg8uyvx`
- `checkpoint`: `/mnt/c/Users/ADMIN/Github/HCMUT/HK252-CO4337-DATN/outputs/dev_fast/resnet18_dev_ablation_groupdro_pgd_linf_pgd_l2_seed42/checkpoints/best.pt`
- `summary_json`: `/mnt/c/Users/ADMIN/Github/HCMUT/HK252-CO4337-DATN/outputs/dev_fast/resnet18_dev_ablation_groupdro_pgd_linf_pgd_l2_seed42_eval/eval/summary.json`

## Latest resnet50
### train
- `run_name`: `resnet50_dev_baseline_erm_seed42`
- `run_dir`: `/workspaces/HK252-CO4337-DATN/outputs/dev_fast/resnet50_dev_baseline_erm_seed42`
- `resolved_config`: `/workspaces/HK252-CO4337-DATN/outputs/dev_fast/resnet50_dev_baseline_erm_seed42/resolved_config.yaml`
- `run_manifest`: `/workspaces/HK252-CO4337-DATN/outputs/dev_fast/resnet50_dev_baseline_erm_seed42/run_manifest.json`
- `config_path`: `/workspaces/HK252-CO4337-DATN/configs/experiments/cifar10/resnet50/baselines/erm.yaml`
- `profile_path`: `/workspaces/HK252-CO4337-DATN/configs/profiles/dev_fast.yaml`
- `wandb_run_id`: `2y46t4k6`
- `best_checkpoint`: `/workspaces/HK252-CO4337-DATN/outputs/dev_fast/resnet50_dev_baseline_erm_seed42/checkpoints/best.pt`
- `last_checkpoint`: `/workspaces/HK252-CO4337-DATN/outputs/dev_fast/resnet50_dev_baseline_erm_seed42/checkpoints/last.pt`
- `train_summary_json`: `/workspaces/HK252-CO4337-DATN/outputs/dev_fast/resnet50_dev_baseline_erm_seed42/train/summary.json`
### eval
- not recorded yet

## Latest vit_b16
### train
- `run_name`: `vit_b16_h100_baseline_erm_seed42`
- `run_dir`: `/mnt/c/Users/ADMIN/Github/HCMUT/HK252-CO4337-DATN/outputs/qa_h100_smoke/vit_b16_h100_baseline_erm_seed42`
- `resolved_config`: `/mnt/c/Users/ADMIN/Github/HCMUT/HK252-CO4337-DATN/outputs/qa_h100_smoke/vit_b16_h100_baseline_erm_seed42/resolved_config.yaml`
- `run_manifest`: `/mnt/c/Users/ADMIN/Github/HCMUT/HK252-CO4337-DATN/outputs/qa_h100_smoke/vit_b16_h100_baseline_erm_seed42/run_manifest.json`
- `config_path`: `/mnt/c/Users/ADMIN/Github/HCMUT/HK252-CO4337-DATN/configs/experiments/cifar10/vit_b16/baselines/erm.yaml`
- `profile_path`: `/tmp/ardg-vit-h100.8JATWH/h100_smoke.yaml`
- `wandb_run_id`: `g3ri998t`
- `best_checkpoint`: `/mnt/c/Users/ADMIN/Github/HCMUT/HK252-CO4337-DATN/outputs/qa_h100_smoke/vit_b16_h100_baseline_erm_seed42/checkpoints/best.pt`
- `last_checkpoint`: `/mnt/c/Users/ADMIN/Github/HCMUT/HK252-CO4337-DATN/outputs/qa_h100_smoke/vit_b16_h100_baseline_erm_seed42/checkpoints/last.pt`
- `train_summary_json`: `/mnt/c/Users/ADMIN/Github/HCMUT/HK252-CO4337-DATN/outputs/qa_h100_smoke/vit_b16_h100_baseline_erm_seed42/train/summary.json`
### eval
- not recorded yet

## Latest Exports
- `json`: `artifacts/exports/latest_results.json`
- `csv`: `artifacts/exports/latest_results.csv`

## Latest Reports
- `table_md`: `artifacts/reports/report_table.md`
- `slides`: `None`
