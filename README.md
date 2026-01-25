# ARDG (Adversarial Robust Domain Generalization)

Research-oriented robust training/evaluation for multiple small image datasets using **PGD Adversarial Training (PGD-AT)** and a standardized **attack suite**.

- Primary attack backend: **TorchAttacks**
- **AutoAttack** is used for evaluation only (optional)

---

## Quick start

Run from the repo root.

### 1) Install

**Recommended (editable install for research/dev):**
```bash
pip install -e .
```

With Weights & Biases logging:
```bash
pip install -e ".[wandb]"
```

Dev tools (tests + lint, optional):
```bash
pip install -e ".[dev]"
```

We recommend `pip install -e .` so that local code changes are immediately reflected without reinstalling.

---

### 2) Train (creates splits on the fly)

```bash
python scripts/train.py --config configs/default.yaml
```

This downloads the dataset (if missing), creates a stratified split from the config seed/val_ratio, and logs metrics to console/W&B.

---

### 3) Evaluate (val + test)

```bash
python scripts/evaluate.py --config configs/default.yaml --checkpoint <path_to_checkpoint>
```

Runs clean + configured attacks (and AutoAttack if enabled) on both val/test splits and logs to console/W&B.

---

## Smoke tests (train 2 epochs + clean eval)

Run all smoke configs (CIFAR-10, MNIST, Fashion-MNIST, Color-MNIST):
```bash
python scripts/smoke_test.py
```

Run a single smoke config:
```bash
python scripts/smoke_test.py --configs configs/smoke_mnist_resnet.yaml
```

---

## Repo layout (essentials)
- `configs/default.yaml` -- single source of truth (dataset/model/train/attacks/logging)
- `configs/smoke_*.yaml` -- small smoke-test configs for quick checks
- `scripts/` -- entry points (prepare_data.py, train.py, evaluate.py, smoke_test.py)
- `src/ardg/` -- library code (models, attacks, training, evaluation, utils)
- `data/` -- dataset cache + processed split artifacts
- `outputs/` -- runtime artifacts (not committed)

---

## Config notes

Default config targets:
- CIFAR-10
- stratified validation split: 2%
- seed: 42
- training: PGD-AT (Linf)
- evaluation: clean + attack suite (+ optional AutoAttack)

All hyperparameters must be controlled via `configs/default.yaml` (no hardcoded experiment settings inside code).

---

## Testing

```bash
pytest -q
```

---

## Minimal dev rules (for clean reviews)
- `scripts/` = entry points only (no heavy logic)
- Core logic lives in `src/ardg/`
- Keep functions small and composable (avoid "god functions")
- Add/modify experiments via config, not by editing training code
