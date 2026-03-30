# ARDG Thesis Experiments

This repo is now organized as a thesis-focused experiment artifact. The primary workflow is CIFAR-10 with ResNet-18. ResNet-50 remains the scale-up path, and ViT-B/16 remains an optional extension path for stronger GPUs such as H100.

## Canonical Workflow

- Default experiment: `configs/experiments/cifar10/resnet18/baselines/erm.yaml`
- Default runtime profile: `configs/profiles/local_gpu.yaml`
- Runtime profiles: `dev_fast`, `local_gpu`, `colab_gpu`, `h100`
- W&B is mandatory for `scripts/train.py` and `scripts/evaluate.py`
- Use `logging.wandb.mode: offline` for local smoke/debug, not `enabled: false`
- `experiment.precision` now controls actual runtime precision in the shared stack:
  - `fp32`: full precision
  - `fp16`: CUDA autocast + GradScaler
  - `bf16`: CUDA autocast without GradScaler
  - CPU fallback stays in `fp32` and logs the fallback reason

## Local First

```bash
python scripts/preflight_check.py --config configs/default.yaml --io_mode normalized --check-wandb
python scripts/train.py --config configs/experiments/cifar10/resnet18/baselines/erm.yaml --profile configs/profiles/local_gpu.yaml --verbose
python scripts/evaluate.py --config configs/experiments/cifar10/resnet18/eval/baseline_erm_all_attacks.yaml --profile configs/profiles/local_gpu.yaml --verbose
```

Smoke runs before expensive GPUs:

```bash
sh scripts/qa/smoke_resnet18_dev_fast.sh
sh scripts/qa/smoke_resnet50.sh
```

## Backbone Ladder

1. ResNet-18: default, smoke path, ablations, and primary experiments.
2. ResNet-50: same pipeline, larger backbone, use the same profile switching rules.
3. ViT-B/16: optional extension, intended for `configs/profiles/h100.yaml`.

Switch backbone by changing only the experiment config:

```bash
python scripts/train.py --config configs/experiments/cifar10/resnet50/baselines/erm.yaml --profile configs/profiles/local_gpu.yaml
python scripts/train.py --config configs/experiments/cifar10/vit_b16/baselines/erm.yaml --profile configs/profiles/h100.yaml
```

## Resume And Outputs

- Resume the latest checkpoint in the canonical run directory:

```bash
python scripts/train.py --config configs/experiments/cifar10/resnet18/baselines/erm.yaml --profile configs/profiles/local_gpu.yaml --resume
```

- Each run directory now contains:
  - `resolved_config.yaml`
  - `run_manifest.json`
  - `logs/train.log` or `logs/eval.log`
  - `checkpoints/best.pt` and `checkpoints/last.pt`
  - `train/summary.json` after training
  - `eval/summary.json` after evaluation
  - `wandb/run_id.txt` and `wandb/run_url.txt` when W&B is attached
  - legacy root-level summary files are still written for compatibility

## Verification Scripts

Run these before scaling to Colab or H100:

```bash
sh scripts/qa/preflight_local.sh
sh scripts/qa/test_wandb_policy.sh
sh scripts/qa/smoke_resnet18_dev_fast.sh
sh scripts/qa/test_resume_resnet18.sh
sh scripts/qa/smoke_resnet50.sh
```

Strong-GPU-only smoke:

```bash
sh scripts/qa/preflight_h100.sh
sh scripts/qa/smoke_vit_h100.sh
```

One-command local gate:

```bash
sh scripts/qa/full_verify_local.sh
```

## Artifact Registry

- `artifacts/latest/manifest.yaml`: canonical config paths + latest important outputs
- `artifacts/latest/paths.md`: human-readable latest paths
- `artifacts/README.md`: registry policy

## Repo Layout

- `configs/experiments/`: scientific experiment configs
- `configs/profiles/`: runtime and hardware overlays
- `configs/fragments/`: reusable config pieces
- `docs/run_guide.md`: concrete commands for local, Colab, and H100
- `docs/protocol.md`: thesis workflow and W&B policy
- `docs/experiment_matrix.md`: canonical experiment map
- `legacy/`: archived configs, docs, notebooks, and exploratory materials

## Reporting

```bash
python scripts/export_results.py --root outputs
python scripts/build_report_tables.py
```
