# Training Pipeline

This document summarizes the current training implementation from `scripts/train.py`
down to one optimizer step.

## Flowchart Starters

- `training-flow.mmd`: end-to-end training flow
- `groupdro-inner-loop.mmd`: GroupDRO all-domains inner loop

## High Level

1. Parse CLI args (`--config`, `--resume`, `--checkpoint`, `--wandb_run_id`, `--platform`, `--verbose`).
2. Resolve resume checkpoint when requested.
   - explicit `--checkpoint`, else `<run_dir>/last.pt`, then `<run_dir>/best.pt`
   - optionally recover W&B run id from checkpoint or `wandb_run_id.txt`
3. Load config and apply runtime platform override when requested.
   - `--platform local` -> `/content/HK252-CO4337-DATN/...`
   - `--platform colab` -> `/content/drive/MyDrive/HK252-CO4337-DATN/...`
   - override `dataset.data_dir` and `logging.output_dir`
4. Setup run.
   - set seed / deterministic mode
   - resolve device (cuda -> cpu fallback on failure)
   - init logger + wandb
5. Build dataloaders (train/val).
6. Build model.
7. Create trainer.
   - build optimizer + optional scheduler
   - build objective from `train.mode`
   - build validation Evaluator + validation attack suite
8. If resume checkpoint exists: `trainer.load_checkpoint(...)`.
9. For each epoch:
   - `train_one_epoch()` -> logs `train/loss_{clean|adv}`, `train/acc_{clean|adv}`, `optim/lr`, plus aggregated objective extras
   - `validate()` -> uses Evaluator + `summarize_suite` (`attack.val_suite` -> `attack.val` -> `attack.eval_suite` -> default clean+pgd20) -> prefixed `val/*` (+ objective extras)
   - compute checkpoint selection vector
   - call optional `objective.on_epoch_end(...)`
   - save `best.pt` when selection vector improves
   - always save `last.pt`
   - for `multi_attack_erm` / `groupdro`, optionally save `best_worst.pt` and `best_avg.pt`
   - update early stopping state
   - step scheduler once per epoch
10. Finish and return checkpoint paths (`best.pt`, `last.pt`).

Implementation in repo:
- `scripts/train.py` (`main`, `_resolve_resume_checkpoint`, `_extract_wandb_run_id`)
- `src/ardg/experiments/common.py` (`setup_run`, `build_loaders`)
- `src/ardg/models/factory.py` (`build_model`)
- `src/ardg/training/objectives/__init__.py` (`build_objective`)
- `src/ardg/training/trainer.py` (`Trainer.train`)
- `src/ardg/evaluation/evaluator.py`
- `src/ardg/evaluation/summary.py`

## Trainer Loop

```text
train():
  best_vector = None

  if start_epoch > epochs:
    return existing last/best paths

  for epoch in [start_epoch .. epochs]:
    train_metrics = train_one_epoch(epoch)
    log(train_metrics + {time_sec, device}, split="train")

    val_metrics = validate(epoch)
    log(val_metrics + {device}, split="val")

    val_acc = val_metrics.get("val/acc_clean", val_metrics.get("val/acc", 0.0))
    selection_names = _resolve_selection_names(cfg, val_metrics)
    metric_name = selection_names[0] if selection_names else "acc_clean"
    metric_vector = tuple(val_metrics.get(name, -inf) for name in selection_names)
    metric_value = metric_vector[0] if metric_vector else val_acc

    objective.on_epoch_end(epoch, loaders)  # optional hook

    if _is_better_checkpoint(metric_vector, best_vector, eps=ckpt_eps):
      best_vector = metric_vector
      save_checkpoint(
        "best",
        selection_metric=metric_value,
        selection_metric_name=metric_name,
        selection_vector=metric_vector,
        val_metrics=val_metrics,
      )

    save_checkpoint(
      "last",
      selection_metric=metric_value,
      selection_metric_name=metric_name,
      selection_vector=metric_vector,
      val_metrics=val_metrics,
    )

    if train_mode in {"multi_attack_erm", "multi-attack-erm", "multi_attack", "groupdro", "group_dro"}:
      maybe_save("best_worst.pt", metric=val_metrics.get("val/acc_worst"))
      maybe_save("best_avg.pt",   metric=val_metrics.get("val/acc_avg"))

    should_stop = _update_early_stopping(epoch, val_metrics)
    if scheduler is not None:
      scheduler.step()
    if should_stop:
      break
```

## Single Train Step

```text
_train_step(batch):
  batch = move_to_device(batch, device)
  batch = objective.preprocess_batch(batch, model)

  optimizer.zero_grad(set_to_none=True)
  loss, metrics = objective.compute_loss(model, batch)
  loss.backward()
  optimizer.step()

  metrics["optim/lr"] = optimizer.param_groups[0]["lr"]
  return metrics
```

Objective hook contract:
- `preprocess_batch`
- `compute_loss`
- `validate`
- `on_epoch_end`
- `state_dict`
- `load_state_dict`

## Checkpoint Selection

```text
input: val_metrics

selection_names =
  if train.selection.vector is set:
    parse explicit vector
  elif mode == groupdro and train.groupdro.selection_metric is set:
    [that metric] + fallback val/acc_avg,val/acc_clean when available
  elif val/acc_worst and val/acc_avg exist:
    ["val/acc_worst", "val/acc_avg", maybe "val/acc_clean"]
  elif val/acc_clean exists:
    ["val/acc_clean"]
  else:
    fallback to unprefixed metrics or first numeric key

vector = [val_metrics.get(name, -inf) for name in selection_names]
is_better:
  lexicographic compare(candidate_vector, best_vector, eps=ckpt_eps)
  treat missing vector entries as -inf

multi-attack/groupdro additionally save:
  best_worst.pt (val/acc_worst)
  best_avg.pt   (val/acc_avg)
```

Implementation in repo:
- `src/ardg/training/trainer.py` (`_resolve_selection_names`, `_is_better_checkpoint`)

## Training Objective Docs

- `method/ERM.md`
- `method/PGD-AT.md`
- `method/Multi-Attack.md`
- `method/DG-based/GroupDRO.md`
- `method/DG-based/GroupDRO++.md`
- `method/DG-based/REx.md`
