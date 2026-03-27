# Thesis Protocol

## Scope

- Primary default: CIFAR-10 + ResNet-18
- Secondary scale-up: CIFAR-10 + ResNet-50
- Optional extension: CIFAR-10 + ViT-B/16 on strong GPUs
- Shared train/eval pipeline preserved through `scripts/train.py`, `scripts/evaluate.py`, and `src/ardg/training/trainer.py`

## Config Policy

- Scientific experiment logic lives under `configs/experiments/`
- Runtime and hardware settings live under `configs/profiles/`
- Reusable config pieces live under `configs/fragments/`
- Switching backbone should mainly change the experiment config
- Switching local/Colab/H100 should mainly change the profile config

## W&B Hard Constraint

- `logging.wandb.enabled` remains in schema
- Canonical train/eval configs require `logging.wandb.enabled: true`
- `scripts/train.py` and `scripts/evaluate.py` now fail immediately if W&B is disabled
- Allowed modes are `online` and `offline`
- `offline` is the correct local smoke/debug path
- Resume continuity is tracked with `wandb_run_id` in checkpoints and run manifests

## Run Artifact Contract

Each run directory should contain:

- `resolved_config.yaml`
- `run_manifest.json`
- stage log file such as `train.log` or `eval.log`
- checkpoint files
- `train_summary.json` after training
- `eval_test_summary.json` after evaluation
- `wandb_run_id.txt` when W&B is attached

## Backbone Ladder

1. ResNet-18 is the clean default and the first path to validate.
2. ResNet-50 uses the same trainer, model factory, and evaluation entrypoint.
3. ViT-B/16 is intentionally optional and expected to run with the H100 profile.

## Legacy Policy

- Previous `configs/local`, `configs/colab`, `configs/smoke_test`, and `configs/eval` are archived into `legacy/old_configs/`
- Archived planning docs and exploratory notebooks live under `legacy/`
- The main repo path documents only the thesis-oriented workflow
