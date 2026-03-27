# ARDG
Minimal, config-driven pipeline for adversarial robustness and domain generalization on CIFAR-style data.

## Quickstart
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .           # add -e ".[wandb]" or -e ".[autoattack]" if needed
python scripts/train.py --config configs/local/resnet50/train/erm.yaml --platform local --verbose
python scripts/evaluate.py --config configs/local/resnet50/eval/all_attacks.yaml --platform local --checkpoint outputs/<RUN>/best.pt --deterministic --seed 42 --verbose
```

## What to Read
- **Training protocols (all modes, pseudocode, flowcharts, run recipes):** [`docs/training-protocols.md`](docs/training-protocols.md)
- **Configs:** `configs/<platform>/<model_family>/{train,eval}/`
- **Entrypoints:** `scripts/train.py`, `scripts/evaluate.py`, `scripts/preflight_check.py`, `scripts/smoke_test.py`
- **Code:** `src/ardg/...` (trainer, objectives, evaluator, attacks, data, models)

## Notes
- Use `--platform local|colab` instead of editing paths.
- W&B logging toggled via `logging.wandb.enabled`; resume with `--wandb_run_id`.
- Prior plans and pseudocode are archived in `docs/archive/`.
