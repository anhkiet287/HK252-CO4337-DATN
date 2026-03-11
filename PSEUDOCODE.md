# Training and Evaluation Pseudocode

This file is now the top-level index.

The detailed pseudocode has been split into the `pseudocode/` folder so each
pipeline or method can be read independently without scrolling through one
monolithic document.

## Main Entry Points

- `pseudocode/README.md`: global map of the pseudocode tree
- `pseudocode/train/README.md`: end-to-end training pipeline
- `pseudocode/eval/README.md`: end-to-end evaluation pipeline
- `pseudocode/train/method/README.md`: training objective index

## Training Docs

- `pseudocode/train/README.md`
  - CLI / resume flow
  - trainer epoch loop
  - single train step
  - checkpoint selection
- `pseudocode/train/method/ERM.md`
- `pseudocode/train/method/PGD-AT.md`
- `pseudocode/train/method/Multi-Attack.md`
- `pseudocode/train/method/DG-based/GroupDRO.md`
- `pseudocode/train/method/DG-based/GroupDRO++.md`
- `pseudocode/train/method/DG-based/REx.md`

## Evaluation Docs

- `pseudocode/eval/README.md`
  - CLI flow
  - max-batches resolution
  - evaluator core
  - attack suite resolution
  - outputs and JSON payload

## Flowchart Starters

- `pseudocode/train/training-flow.mmd`
- `pseudocode/train/groupdro-inner-loop.mmd`
- `pseudocode/eval/evaluation-flow.mmd`

## Implementation Map

- Training CLI: `scripts/train.py`
- Evaluation CLI: `scripts/evaluate.py`
- Run setup: `src/ardg/experiments/common.py`
- Trainer: `src/ardg/training/trainer.py`
- Objectives: `src/ardg/training/objectives/`
- Evaluator: `src/ardg/evaluation/evaluator.py`
- Attack suite builder: `src/ardg/attacks/attack_suite.py`

## Suggested Reading Order

1. Read `pseudocode/train/README.md` or `pseudocode/eval/README.md`.
2. Jump to the specific method file you care about.
3. Use the Mermaid `.mmd` files when you need a report or slide diagram.
