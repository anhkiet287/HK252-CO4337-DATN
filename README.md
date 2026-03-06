# ARDG (Adversarial Robust Domain Generalization)

Minimal training/evaluation pipeline for CIFAR-style adversarial robustness and DG experiments.

## Current Status

Implemented training modes:
- `erm`
- `pgd_at`
- `multi_attack_erm`
- `rex`
- `groupdro`
- `groupdro_plus`

Attack backend:
- `torchattacks` for training and evaluation attacks

Main idea of current codebase:
- keep training objectives modular
- use one evaluation entry point: `scripts/evaluate.py`
- use one evaluation attack interface: `attack.eval_suite`
- keep experiments deterministic and comparable

## Execution Policy

1. Implement and debug locally first.
2. Run preflight + smoke train + smoke eval locally.
3. Run full experiments on Colab GPU.
4. Use fixed Colab output root:
   `/content/drive/MyDrive/ardg/HK252-CO4337-DATN/outputs`

## Setup

Local:

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

## Core Entry Points

- Train: `scripts/train.py`
- Evaluate: `scripts/evaluate.py`
- Preflight: `scripts/preflight_check.py`
- Attack visualize: `scripts/attack_visual_check.py`
- Smoke batch runner: `scripts/smoke_test.py`

Core modules:
- Trainer loop: `src/ardg/training/trainer.py`
- Objectives: `src/ardg/training/objectives/`
- Evaluator class: `src/ardg/evaluation/evaluator.py`
- Attack factory/suites: `src/ardg/attacks/attack_suite.py`

## Training

Generic:

```bash
python scripts/train.py --config <CONFIG_PATH>
```

Resume latest in run dir:

```bash
python scripts/train.py --config <CONFIG_PATH> --resume
```

Resume explicit checkpoint:

```bash
python scripts/train.py --config <CONFIG_PATH> --checkpoint <CKPT_PATH>
```

Verbose debug summary:

```bash
python scripts/train.py --config <CONFIG_PATH> --verbose
```

## Evaluation (Single Pipeline For All Models)

Generic:

```bash
python scripts/evaluate.py --config <CONFIG_PATH> --checkpoint <CKPT_PATH>
```

Smoke (exactly 1 test sample):

```bash
python scripts/evaluate.py --config <CONFIG_PATH> --checkpoint <CKPT_PATH> --smoke-one-sample
```

Verbose debug summary:

```bash
python scripts/evaluate.py --config <CONFIG_PATH> --checkpoint <CKPT_PATH> --verbose
```

If `--checkpoint` is omitted, evaluator tries `best.pt` then `last.pt` in run dir.

### Unified Eval Suite (Recommended)

Use this in configs (`attack.eval_suite`) for all models/modes to ensure fairness:

```yaml
attack:
  eval_suite:
    max_batches: 0
    attacks:
      - label: pgd20
        type: pgd
        eps: 0.0313725
        step_size: 0.007843
        num_steps: 20
        restarts: 5
        loss: ce
        random_start: true
      - label: autoattack
        type: autoattack
        norm: Linf
        eps: 0.0313725
        version: standard
        n_classes: 10
        verbose: false
```

## Preflight

Check normalization and attack-space consistency before training:

```bash
python scripts/preflight_check.py --config <CONFIG_PATH> --io_mode normalized
python scripts/preflight_check.py --config <CONFIG_PATH> --io_mode pixel
```

## Attack Visualization

Single source attack:

```bash
python scripts/attack_visual_check.py --config <CONFIG_PATH> --checkpoint <CKPT_PATH> --split test --attack-source train --attack from_source --num-samples 8 --strict-eps
```

Run all attacks side-by-side:

```bash
python scripts/attack_visual_check.py --config <CONFIG_PATH> --checkpoint <CKPT_PATH> --split test --all-attacks --num-samples 8 --strict-eps
```

## Determinism and Fairness Rules

Use the same settings across compared runs:
- same dataset split and seed
- same threat model (`norm`, `eps`)
- same eval suite (`attack.eval_suite`)
- same deterministic flag
- same eval sample count / max batches

Required config defaults:
- `experiment.deterministic: true`
- fixed `experiment.seed`

Evaluator deterministic controls:
- `--deterministic` / `--no-deterministic`
- `--seed <int>`
- `--max-batches <int>`
- `--max-test-samples <int>`

## W&B Organization

You can group runs by model and stage via config:

```yaml
logging:
  wandb:
    enabled: true
    project: ardg
    entity: ""
    group_template: "{model}/{stage}"   # e.g., resnet50/train, resnet50/eval
    add_default_tags: true
```

## Colab Commands (Style)

Use single-line `!python` commands.

Example train:

```bash
!python scripts/train.py --config configs/colab/resnet50/at/multi_attack_erm.yaml --verbose
```

Example eval:

```bash
!python scripts/evaluate.py --config configs/colab/resnet50/at/multi_attack_erm.yaml --checkpoint /content/drive/MyDrive/ardg/HK252-CO4337-DATN/outputs/<RUN_NAME>/best.pt --deterministic --seed 42 --verbose
```

## Minimal Maintenance Direction

Keep this repo simple and stable:
- keep one eval entrypoint (`scripts/evaluate.py`)
- keep one eval config interface (`attack.eval_suite`)
- keep objectives isolated in `training/objectives/`
- keep attack construction centralized in `attacks/attack_suite.py`
- avoid duplicating mode-specific eval scripts

Short-term cleanup targets:
- migrate remaining legacy `attack.eval` + `attack.autoattack` configs to `attack.eval_suite`
- keep smoke configs minimal and deterministic (1-sample eval when needed)
- keep debugging through `--verbose` flags instead of custom temporary scripts
