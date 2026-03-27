# Run Guide

## Preflight

Run a fast consistency check before training:

```bash
python scripts/preflight_check.py --config configs/default.yaml --io_mode normalized --check-wandb
```

## Local GPU

Primary default:

```bash
python scripts/train.py --config configs/experiments/cifar10/resnet18/baselines/erm.yaml --profile configs/profiles/local_gpu.yaml --verbose
python scripts/evaluate.py --config configs/experiments/cifar10/resnet18/eval/baseline_erm_all_attacks.yaml --profile configs/profiles/local_gpu.yaml --verbose
```

Proposed method:

```bash
python scripts/train.py --config configs/experiments/cifar10/resnet18/proposed/groupdro_plus.yaml --profile configs/profiles/local_gpu.yaml --verbose
```

Smoke runs:

```bash
python scripts/train.py --config configs/experiments/cifar10/resnet18/baselines/erm.yaml --profile configs/profiles/dev_fast.yaml --verbose
python scripts/evaluate.py --config configs/experiments/cifar10/resnet18/eval/baseline_erm_all_attacks.yaml --profile configs/profiles/dev_fast.yaml --smoke-one-sample --max-batches 1 --verbose
python scripts/train.py --config configs/experiments/cifar10/resnet50/baselines/erm.yaml --profile configs/profiles/dev_fast.yaml --verbose
python scripts/train.py --config configs/experiments/cifar10/vit_b16/baselines/erm.yaml --profile configs/profiles/dev_fast.yaml --verbose
```

## Colab GPU

```bash
python scripts/preflight_check.py --config configs/experiments/cifar10/resnet18/baselines/erm.yaml --profile configs/profiles/colab_gpu.yaml --io_mode normalized --check-wandb
python scripts/train.py --config configs/experiments/cifar10/resnet18/baselines/erm.yaml --profile configs/profiles/colab_gpu.yaml --verbose
python scripts/evaluate.py --config configs/experiments/cifar10/resnet18/eval/baseline_erm_all_attacks.yaml --profile configs/profiles/colab_gpu.yaml --verbose
```

## SSH Server / H100

```bash
python scripts/preflight_check.py --config configs/experiments/cifar10/vit_b16/baselines/erm.yaml --profile configs/profiles/h100.yaml --io_mode normalized --check-wandb
python scripts/train.py --config configs/experiments/cifar10/resnet50/baselines/erm.yaml --profile configs/profiles/h100.yaml --verbose
python scripts/train.py --config configs/experiments/cifar10/vit_b16/baselines/erm.yaml --profile configs/profiles/h100.yaml --verbose
```

## Resume

Resume from the run directory selected by config plus profile:

```bash
python scripts/train.py --config configs/experiments/cifar10/resnet18/baselines/erm.yaml --profile configs/profiles/local_gpu.yaml --resume
```

Or resume from a specific checkpoint:

```bash
python scripts/train.py --config configs/experiments/cifar10/resnet18/baselines/erm.yaml --profile configs/profiles/local_gpu.yaml --checkpoint outputs/thesis/local/resnet18_local_baseline_erm_seed42/best.pt
```
