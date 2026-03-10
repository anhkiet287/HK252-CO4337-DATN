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
- keep train/eval configs separated (`.../at|erm/...` for train, `.../eval/...` for eval)
- keep experiments deterministic and comparable

## Execution Policy

1. Implement and debug locally first.
2. Run preflight + smoke train + smoke eval locally.
3. Run full experiments on Colab GPU.
4. Prefer runtime path switching via `--platform` instead of editing configs by hand.

Platform override roots:
- `--platform local` -> `/content/HK252-CO4337-DATN`
- `--platform colab` -> `/content/drive/MyDrive/HK252-CO4337-DATN`

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

## Config Layout (Train vs Eval)

For each model family, maintain separate configs:
- Train configs: `configs/<platform>/<model_family>/train/*.yaml`
- Eval configs: `configs/<platform>/<model_family>/eval/*.yaml`

ResNet50 examples:
- Train:
  - `configs/colab/resnet50/train/erm.yaml`
  - `configs/colab/resnet50/train/pgd_at.yaml`
  - `configs/colab/resnet50/train/multi_attack_erm.yaml`
- Eval:
  - `configs/colab/resnet50/eval/all_attacks.yaml` (clean + PGD20 + AutoAttack; shared across modes)

## Training (minimal commands)

Use the train configs under `configs/colab/resnet50/train/` (same structure for other platforms/models).

- ERM

  ```bash
  python scripts/train.py --config configs/colab/resnet50/train/erm.yaml --verbose
  ```

- PGD-AT

  ```bash
  python scripts/train.py --config configs/colab/resnet50/train/pgd_at.yaml --verbose
  ```

- Multi-Attack ERM (saves `best.pt`, `best_worst.pt`, `best_avg.pt`, `last.pt`)

  ```bash
  python scripts/train.py --config configs/colab/resnet50/train/multi_attack_erm.yaml --verbose
  ```

- GroupDRO (same attack domains as multi-attack ERM; saves `best.pt`, `best_worst.pt`, `best_avg.pt`, `last.pt`)

  ```bash
  python scripts/train.py --config configs/colab/resnet50/train/groupdro.yaml --verbose
  ```

Runtime platform override:

```bash
python scripts/train.py --config <CONFIG_PATH> --platform local --verbose
python scripts/train.py --config <CONFIG_PATH> --platform colab --verbose
```

Resume latest in run dir:

```bash
python scripts/train.py --config <CONFIG_PATH> --platform <local|colab> --resume
```

Resume explicit checkpoint:

```bash
python scripts/train.py --config <CONFIG_PATH> --platform <local|colab> --checkpoint <CKPT_PATH>
```

Resume and attach to an existing W&B run:

```bash
python scripts/train.py --config <CONFIG_PATH> --platform <local|colab> --resume --wandb_run_id <RUN_ID>
```

## Evaluation (single pipeline for all models)

Default unified eval suite lives in `configs/colab/resnet50/eval/all_attacks.yaml` and covers clean + PGD20 + AutoAttack. Keep the same eval config across models for fairness.

Run full eval:

```bash
python scripts/evaluate.py --config configs/colab/resnet50/eval/all_attacks.yaml \
  --platform colab \
  --checkpoint /content/drive/MyDrive/HK252-CO4337-DATN/outputs/<RUN_NAME>/best.pt \
  --deterministic --seed 42 --verbose
```

Smoke 1 sample:

```bash
python scripts/evaluate.py --config configs/colab/resnet50/eval/all_attacks.yaml \
  --platform colab \
  --checkpoint /content/drive/MyDrive/HK252-CO4337-DATN/outputs/<RUN_NAME>/best.pt \
  --smoke-one-sample
```

If `--checkpoint` is omitted, evaluator tries `best.pt` then `last.pt` in the run directory.
With `--platform`, the run directory is resolved from the corresponding platform root automatically.

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

### Train (Colab, ResNet50)
- ERM: `!python scripts/train.py --config configs/colab/resnet50/train/erm.yaml --platform colab --verbose`
- PGD-AT: `!python scripts/train.py --config configs/colab/resnet50/train/pgd_at.yaml --platform colab --verbose`
- Multi-Attack ERM: `!python scripts/train.py --config configs/colab/resnet50/train/multi_attack_erm.yaml --platform colab --verbose`
- GroupDRO: `!python scripts/train.py --config configs/colab/resnet50/train/groupdro.yaml --platform colab --verbose`

### Evaluate (Colab, unified eval suite)
Use `attack.eval_suite` configs to keep fairness; point to your checkpoint under `/content/drive/MyDrive/HK252-CO4337-DATN/outputs/<RUN_NAME>/best.pt`.
- ERM/PGD-AT/Multi-Attack (all attacks):  
  `!python scripts/evaluate.py --config configs/colab/resnet50/eval/all_attacks.yaml --platform colab --checkpoint /content/drive/MyDrive/HK252-CO4337-DATN/outputs/<RUN_NAME>/best.pt --deterministic --seed 42 --verbose`
- Smoke (1 sample): add `--smoke-one-sample`

### Train (Local)
- Activate env then:  
  `python scripts/train.py --config configs/colab/resnet50/train/pgd_at.yaml --platform local --verbose`

### Evaluate (Local)
- `python scripts/evaluate.py --config configs/colab/resnet50/eval/all_attacks.yaml --platform local --checkpoint /content/HK252-CO4337-DATN/outputs/<RUN_NAME>/best.pt --deterministic --seed 42 --verbose`

## Run Outputs

Each training run writes to `logging.output_dir` (default `outputs/<run_name>`):
- `best.pt` – by primary selection vector (default: `val/acc_worst`, then `val/acc_avg`, then `val/acc_clean`).
- `best_worst.pt` – best on `val/acc_worst` (`multi_attack_erm` and `groupdro`).
- `best_avg.pt` – best on `val/acc_avg` (`multi_attack_erm` and `groupdro`).
- `last.pt` – last epoch.
- `train_history.jsonl` / `train_summary.json` – per-epoch logs.

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
