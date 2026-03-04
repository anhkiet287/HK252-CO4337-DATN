# ARDG (Adversarial Robust Domain Generalization)

Training and evaluation pipeline for CIFAR-style robustness/domain-generalization experiments.

Current implemented training modes:
- `erm`
- `pgd_at`
- `multi_attack_erm`
- `rex`
- `groupdro`
- `groupdro_plus`

Attack backend:
- Training/eval attacks: `torchattacks`
- Optional eval-only AutoAttack: `autoattack`

## 1) Setup

From repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional extras:

```bash
pip install -e ".[wandb]"
pip install -e ".[autoattack]"
pip install -e ".[dev]"
```

Note: scripts default to `configs/default.yaml` if `--config` is omitted, but this repo does not ship that file. Always pass `--config`.

## 2) Configs By Mode

Local configs:
- `erm`: `configs/local/resnet18/erm/erm.yaml`, `configs/local/vit_b16/erm/erm.yaml`
- `pgd_at`: `configs/local/resnet18/at/pgd_at.yaml`, `configs/local/resnet50/at/pgd_at.yaml`, `configs/local/vit_b16/at/pgd_at.yaml`
- `multi_attack_erm`: `configs/local/resnet50/at/multi_attack_erm.yaml`
- `rex`: `configs/local/resnet18/at/rex.yaml`, `configs/local/vit_b16/at/rex.yaml`
- `groupdro`: `configs/local/resnet18/at/groupdro.yaml`, `configs/local/vit_b16/at/groupdro.yaml`
- `groupdro_plus`: `configs/local/resnet18/at/groupdro_plus.yaml`, `configs/local/vit_b16/at/groupdro_plus.yaml`

Colab configs:
- `erm`: `configs/colab/resnet18/erm/erm.yaml`, `configs/colab/resnet50/erm/erm.yaml`, `configs/colab/vit_b16/erm/erm.yaml`
- `pgd_at`: `configs/colab/resnet18/at/pgd_at.yaml`, `configs/colab/resnet50/at/pgd_at.yaml`, `configs/colab/vit_b16/at/pgd_at.yaml`
- `multi_attack_erm`: `configs/colab/resnet50/at/multi_attack_erm.yaml`, `configs/colab/resnet50/at/multi_attack_erm_5ep.yaml`, `configs/colab/resnet50/at/multi_attack_erm_10ep.yaml`
- `rex`: `configs/colab/resnet18/at/rex.yaml`, `configs/colab/vit_b16/at/rex.yaml`
- `groupdro`: `configs/colab/resnet18/at/groupdro.yaml`, `configs/colab/vit_b16/at/groupdro.yaml`
- `groupdro_plus`: `configs/colab/resnet18/at/groupdro_plus.yaml`, `configs/colab/vit_b16/at/groupdro_plus.yaml`

## 3) Train

Generic:

```bash
PYTHONPATH=src python3 scripts/train.py --config <CONFIG_PATH>
```

Resume latest checkpoint in run dir:

```bash
PYTHONPATH=src python3 scripts/train.py --config <CONFIG_PATH> --resume
```

Resume specific checkpoint:

```bash
PYTHONPATH=src python3 scripts/train.py --config <CONFIG_PATH> --checkpoint <CKPT_PATH>
```

Examples:

```bash
# ERM
PYTHONPATH=src python3 scripts/train.py --config configs/local/resnet18/erm/erm.yaml

# PGD-AT
PYTHONPATH=src python3 scripts/train.py --config configs/local/resnet50/at/pgd_at.yaml

# Multi-attack ERM
PYTHONPATH=src python3 scripts/train.py --config configs/local/resnet50/at/multi_attack_erm.yaml

# REx
PYTHONPATH=src python3 scripts/train.py --config configs/local/resnet18/at/rex.yaml

# GroupDRO
PYTHONPATH=src python3 scripts/train.py --config configs/local/resnet18/at/groupdro.yaml

# GroupDRO+
PYTHONPATH=src python3 scripts/train.py --config configs/local/resnet18/at/groupdro_plus.yaml
```

## 4) Evaluate

Generic:

```bash
PYTHONPATH=src python3 scripts/evaluate.py --config <CONFIG_PATH> --checkpoint <CKPT_PATH> --splits val,test
```

If `--checkpoint` is omitted, evaluator tries `best.pt` then `last.pt` in the run directory for that config.

Evaluation behavior:
- always logs clean accuracy
- runs attack suite from `attack.eval` (currently requires `pgd20`)
- runs AutoAttack only if `attack.autoattack.enabled: true`

## 5) Preflight Checks

Use preflight before training to verify normalization/attack IO consistency.

```bash
PYTHONPATH=src python3 scripts/preflight_check.py --config <CONFIG_PATH> --io_mode normalized
PYTHONPATH=src python3 scripts/preflight_check.py --config <CONFIG_PATH> --io_mode pixel
```

## 6) Attack Visualization (Clean / Adv / Perturbation)

```bash
PYTHONPATH=src python3 scripts/attack_visual_check.py \
  --config <CONFIG_PATH> \
  --checkpoint <CKPT_PATH> \
  --split test \
  --attack-source eval_pgd20 \
  --num-samples 8
```

Artifacts are saved to `<run_dir>/attack_visual_check` unless `--output-dir` is set.

## 7) Smoke Tests

Run existing smoke configs explicitly:

```bash
PYTHONPATH=src python3 scripts/smoke_test.py --configs \
  configs/smoke_test/resnet50_local_pgd_at_smoke.yaml \
  configs/smoke_test/resnet50_local_multi_attack_erm_smoke.yaml
```

## 8) Outputs And Logging

Per-run directory:
- `<output_dir>/<run_name>/best.pt`
- `<output_dir>/<run_name>/last.pt`

Default output root when `logging.output_dir` is omitted:
- local: `outputs`
- colab: `/content/drive/MyDrive/ardg/HK252-CO4337-DATN/outputs`

WandB:
- controlled by `logging.wandb.enabled`
- for resume support, run id is persisted in `<run_dir>/wandb_run_id.txt`

## 9) Core Entry Points

- Train: `scripts/train.py`
- Evaluate: `scripts/evaluate.py`
- Smoke: `scripts/smoke_test.py`
- Preflight: `scripts/preflight_check.py`
- Attack visualization: `scripts/attack_visual_check.py`

Core modules:
- Training loop: `src/ardg/training/trainer.py`
- Objectives registry/modes: `src/ardg/training/objectives/__init__.py`
- Evaluator class: `src/ardg/evaluation/evaluator.py`
- Attack builders: `src/ardg/attacks/attack_suite.py`
