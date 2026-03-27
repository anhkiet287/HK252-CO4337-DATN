# Pseudocode Tree

This folder contains the split version of the pseudocode documentation.

## Entry Points

- `../PSEUDOCODE.md`: top-level index
- `train/README.md`: training pipeline
- `eval/README.md`: evaluation pipeline
- `train/method/README.md`: objective index

## Training

- `train/README.md`
- `train/method/ERM.md`
- `train/method/PGD-AT.md`
- `train/method/Multi-Attack.md`
- `train/method/DG-based/GroupDRO.md`
- `train/method/DG-based/GroupDRO++.md`
- `train/method/DG-based/REx.md`

## Evaluation

- `eval/README.md`

## Flowchart Starters

- `train/training-flow.mmd`: end-to-end training pipeline
- `train/groupdro-inner-loop.mmd`: GroupDRO `all_domains` inner loop
- `eval/evaluation-flow.mmd`: evaluation pipeline

## Suggested Workflow

1. Start from `train/README.md` or `eval/README.md`.
2. Open the objective-specific markdown file you need.
3. Use the Mermaid `.mmd` files when you need a diagram for slides or reports.
