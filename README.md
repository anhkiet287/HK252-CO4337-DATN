# ARDG Thesis Experiments

This repo is now organized as a thesis-focused experiment artifact. The primary workflow is CIFAR-10 with ResNet-18. ResNet-50 remains the scale-up path, and ViT-B/16 remains an optional extension path for stronger GPUs such as H100.

## Canonical Workflow

- Default experiment: `configs/experiments/cifar10/resnet18/baselines/erm.yaml`
- Default runtime profile: `configs/profiles/local_gpu.yaml`
- Runtime profiles: `dev_fast`, `local_gpu`, `colab_gpu`, `h100`
- W&B is mandatory for `scripts/train.py` and `scripts/evaluate.py`
- Use `logging.wandb.mode: offline` for local smoke/debug, not `enabled: false`

## Local First

```bash
python scripts/preflight_check.py --config configs/default.yaml --io_mode normalized --check-wandb
python scripts/train.py --config configs/experiments/cifar10/resnet18/baselines/erm.yaml --profile configs/profiles/local_gpu.yaml --verbose
python scripts/evaluate.py --config configs/experiments/cifar10/resnet18/eval/baseline_erm_all_attacks.yaml --profile configs/profiles/local_gpu.yaml --verbose
```

Smoke runs before expensive GPUs:

```bash
python scripts/train.py --config configs/experiments/cifar10/resnet18/baselines/erm.yaml --profile configs/profiles/dev_fast.yaml --verbose
python scripts/evaluate.py --config configs/experiments/cifar10/resnet18/eval/baseline_erm_all_attacks.yaml --profile configs/profiles/dev_fast.yaml --smoke-one-sample --max-batches 1 --verbose
python scripts/smoke_test.py
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
  - `train.log` or `eval.log`
  - checkpoint files such as `best.pt` and `last.pt`
  - `train_summary.json` after training
  - `eval_test_summary.json` after evaluation
  - `wandb_run_id.txt` when W&B is attached

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
python scripts/export_results.py --root outputs/thesis
python scripts/build_report_tables.py --input outputs/exported_results.json --output outputs/report_table.md
```
