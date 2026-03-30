# Run Guide

## Preflight

Run a fast consistency check before training:

```bash
sh scripts/qa/preflight_local.sh
```

## Local GPU

Primary default:

```bash
python scripts/train.py --config configs/experiments/cifar10/resnet18/baselines/erm.yaml --profile configs/profiles/local_gpu.yaml --verbose
python scripts/evaluate.py --config configs/experiments/cifar10/resnet18/eval/baseline_erm_all_attacks.yaml --profile configs/profiles/local_gpu.yaml --verbose
```

Runtime precision is controlled by `experiment.precision`:

- `fp32`: full precision
- `fp16`: CUDA autocast + GradScaler
- `bf16`: CUDA autocast without GradScaler
- CPU fallback logs a forced `fp32` downgrade

GroupDRO study:

```bash
python scripts/train.py --config configs/experiments/cifar10/resnet18/ablations/groupdro.yaml --profile configs/profiles/local_gpu.yaml --verbose
```

Smoke runs:

```bash
sh scripts/qa/smoke_resnet18_dev_fast.sh
sh scripts/qa/test_resume_resnet18.sh
sh scripts/qa/smoke_resnet50.sh
sh scripts/qa/full_verify_local.sh
```

## Colab GPU

```bash
python scripts/preflight_check.py --config configs/experiments/cifar10/resnet18/baselines/erm.yaml --profile configs/profiles/colab_gpu.yaml --io_mode normalized --check-wandb
python scripts/train.py --config configs/experiments/cifar10/resnet18/baselines/erm.yaml --profile configs/profiles/colab_gpu.yaml --verbose
python scripts/evaluate.py --config configs/experiments/cifar10/resnet18/eval/baseline_erm_all_attacks.yaml --profile configs/profiles/colab_gpu.yaml --verbose
```

## SSH Server / H100

```bash
sh scripts/qa/preflight_h100.sh
python scripts/train.py --config configs/experiments/cifar10/resnet50/baselines/erm.yaml --profile configs/profiles/h100.yaml --verbose
sh scripts/qa/smoke_vit_h100.sh
```

## Resume

Resume from the run directory selected by config plus profile:

```bash
python scripts/train.py --config configs/experiments/cifar10/resnet18/baselines/erm.yaml --profile configs/profiles/local_gpu.yaml --resume
```

Or resume from a specific checkpoint:

```bash
python scripts/train.py --config configs/experiments/cifar10/resnet18/baselines/erm.yaml --profile configs/profiles/local_gpu.yaml --checkpoint outputs/thesis/local/resnet18_local_baseline_erm_seed42/checkpoints/best.pt
```

## Artifact Registry

- `artifacts/latest/manifest.yaml` records canonical config paths and latest important outputs
- `artifacts/latest/paths.md` renders the same information in Markdown

## W&B Policy Verification

```bash
sh scripts/qa/test_wandb_policy.sh
```
