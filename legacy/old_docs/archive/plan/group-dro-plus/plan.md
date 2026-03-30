# GroupDRO++ Implementation Plan

## Summary
Track and improve GroupDRO++ (`train.mode: groupdro_plus`) with clustering-based pseudo-groups, stable training, and clear validation criteria.

## Public Interface Changes
1. Keep `train.mode: groupdro_plus`.
2. Use `train.groupdro_plus` knobs: `num_clusters`, `eta`, `lambda_reg`, `gamma`.
3. Keep backward compatibility with current objective and trainer pipeline.

## Implementation Scope
1. Review `src/ardg/training/objectives/groupdro_plus.py` and `src/ardg/training/cluster_utils.py`.
2. Validate cluster stability and edge-case handling for small batch sizes.
3. Add/refresh configs for local/colab GroupDRO++ runs.
4. Add smoke config and logging checks for cluster/group metrics.

## Runtime Behavior
1. Optional adversarial preprocess (`train.adv_training=true`).
2. Build pseudo-groups via k-means on model outputs.
3. Update group weights `q` and optimize robust + regularized objective.

## Test Cases and Scenarios
1. Training runs with configured cluster count.
2. No NaN/inf in `q`, loss, or regularization term.
3. Smoke run logs group metrics and completes.
4. Regression: other modes unchanged.

## Pipeline and Workflow Summary
1. Batch -> optional adversarial preprocess.
2. Forward pass -> k-means pseudo-group IDs.
3. Group loss + weighted objective + regularization.
4. Optimize -> validate -> checkpoint.

## Assumptions and Defaults
1. Clustering runs per-batch as current implementation.
2. Group count defaults from config.
3. Attack backend remains `torchattacks` where adversarial branch is enabled.
