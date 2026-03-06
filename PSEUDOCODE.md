# Training Pipeline Pseudocode (High -> Low Level)

## 1) High Level (End-to-End)

1. Load config.
2. Setup run.
   - set seed / deterministic mode
   - resolve device (cuda/cpu)
   - init logger + wandb
3. Build dataloaders (train/val/test).
4. Build model.
5. Build objective from `train.mode`.
   - `erm` / `pgd_at` / `multi_attack_erm` / `rex` / `groupdro` / `groupdro_plus`
6. Create trainer.
7. For each epoch:
   - `train_one_epoch()`
   - `validate()`
   - choose/save best checkpoint by selection metric
   - save last checkpoint
   - scheduler step
8. Finish and return checkpoint paths (`best.pt`, `last.pt`).

## 2) Mid Level (Trainer Loop)

```text
train():
  best_vector = None

  for epoch in [start_epoch .. epochs]:
    train_metrics = train_one_epoch(epoch)
    val_metrics = validate(epoch)

    if mode == "multi_attack_erm":
      metric_name, metric_value, metric_vector = select_multi_attack_checkpoint(val_metrics)
    else:
      metric_name = "acc"
      metric_value = val_metrics["acc"]
      metric_vector = (metric_value,)

    if is_better(metric_vector, best_vector, eps=1e-6):
      best_vector = metric_vector
      save_checkpoint(
        "best",
        selection_metric=metric_value,
        selection_metric_name=metric_name,
      )

    save_checkpoint(
      "last",
      selection_metric=metric_value,
      selection_metric_name=metric_name,
    )

    scheduler.step()
```

## 3) Low Level (Single Train Step)

```text
_train_step(batch):
  batch = move_to_device(batch)
  batch = objective.preprocess_batch(batch, model)

  optimizer.zero_grad()
  loss, metrics = objective.loss(model, batch)
  loss.backward()
  optimizer.step()

  metrics["lr"] = optimizer.lr
  return metrics
```

## 4) Objective Logic by Mode

### ERM
```text
preprocess_batch: no-op
loss: CE(model(x), y)
```

### PGD-AT
```text
preprocess_batch: x_adv = PGD(model, x, y)
loss: CE(model(x_adv), y)
```

### Multi-Attack ERM
```text
preprocess_batch:
  strategy=per_batch:
    sample 1 domain for whole batch
  strategy=split_batch:
    split samples into domains in same batch
  strategy=all_domains:
    generate one attacked batch per domain

loss:
  per_batch/split_batch:
    CE on attacked batch
  all_domains:
    mean CE across all domain batches

validate:
  compute per-domain val metrics:
    val/acc_<domain>, val/loss_<domain>
  compute aggregates:
    val/acc_avg, val/acc_worst, val/worst_domain, val/acc_clean
  optional probe:
    val/pgd20_probe_acc, val/pgd20_probe_loss
```

### GroupDRO
```text
preprocess_batch:
  optional adversarial preprocessing if enabled

loss:
  compute loss per group: loss_g
  update q:
    q_g <- q_g * exp(eta * loss_g)
    normalize q
  total_loss = sum(q_g * loss_g)
```

## 5) Checkpoint Selection (Multi-Attack)

```text
input: val_metrics

primary = train.multi_attack.checkpoint_metric
tiebreakers = train.multi_attack.checkpoint_tiebreakers

defaults:
  if probe enabled and primary missing:
    primary = pgd20_probe_acc
  else if primary missing:
    primary = worst_acc

vector = [primary_value, tie_1_value, tie_2_value, ...]

is_better:
  lexicographic compare(candidate_vector, best_vector, eps=1e-6)
```

# Evaluation Pipeline Pseudocode (High -> Low Level)

## 1) High Level (End-to-End)

1. Parse CLI args (`--config`, `--checkpoint`, `--deterministic`, `--seed`, etc.).
2. Setup run with eval suffix (`run_name + "_eval"`).
3. Resolve deterministic settings:
   - seed from CLI or config
   - deterministic from CLI or config
4. Apply dataloader/runtime overrides:
   - `num_workers`
   - `max_test_samples`
   - smoke mode (`--smoke-one-sample`)
5. Resolve checkpoint path:
   - explicit `--checkpoint`, else try `best.pt`, then `last.pt`.
6. Build model and load checkpoint weights.
7. Build test loader.
8. Build evaluation attack suite from config (`attack.eval_suite`).
9. Run clean evaluation.
10. Run adversarial evaluation for each attack in suite.
11. Aggregate summary metrics and log to console + wandb.
12. Save JSON report (`eval_test_summary.json` or `--save-json`).

## 2) Mid Level (Evaluate Main Loop)

```text
evaluate_main():
  args = parse_args()
  cfg, logger, run, device = setup_run(config, run_name_suffix="eval")

  eval_seed = resolve_seed(cfg, args.seed)
  eval_det = resolve_deterministic(cfg, args.deterministic)
  set_seed(eval_seed, deterministic=eval_det)

  apply_cli_overrides(cfg, args)  # num_workers, max_test_samples, smoke

  ckpt = resolve_checkpoint(cfg, args.checkpoint)
  model = load_model_from_checkpoint(cfg, ckpt, device)
  _, _, test_loader = build_loaders(cfg)

  max_batches = resolve_max_batches(cfg, args.max_batches, args.smoke_one_sample)
  evaluator = Evaluator(model, test_loader, device, max_batches=max_batches)

  clean_metrics = evaluator.evaluate_clean()
  attacks = build_eval_suite(cfg, model)

  robust = {}
  failures = {}
  for (label, attack) in attacks:
    try:
      robust[label] = evaluator.evaluate_under_attack(attack)
    except Exception as e:
      failures[label] = str(e)

  worst_robust_acc = min(robust[*].acc, default=clean_metrics.acc)
  log test metrics + system metrics
  save payload JSON
```

## 3) Low Level (Evaluator Core)

```text
evaluate_clean():
  model.eval()
  for batch in loader (optionally capped by max_batches):
    x, y = unpack_xy(batch)
    logits = model(x)
    loss = CE(logits, y)
    accumulate loss_sum, correct_sum, n
  return {loss=loss_sum/n, acc=correct_sum/n, n_samples=n}
```

```text
evaluate_under_attack(attack):
  model.eval()
  for batch in loader (optionally capped by max_batches):
    x, y = unpack_xy(batch)
    x_adv = attack(x, y)
    logits = model(x_adv)
    loss = CE(logits, y)
    accumulate loss_sum, correct_sum, n
  return {loss=loss_sum/n, acc=correct_sum/n, n_samples=n}
```

## 4) Attack Suite Resolution

```text
build_eval_suite(cfg, model):
  if attack.eval_suite exists:
    parse suite.attacks
  else:
    fallback to legacy attack.eval + attack.autoattack

  for each attack spec:
    build via build_attack(...)
    supported types:
      clean, fgsm, fgsm_rs, pgd, pgd_ce, pgd_dlr,
      cw, deepfool, square, fab, autoattack
  return dict[label -> callable]
```

## 5) Determinism + Fairness Checklist

```text
Must keep fixed across model comparisons:
  - eval config (same attack.eval_suite)
  - seed
  - deterministic flag
  - max_batches / max_test_samples
  - threat model params (eps, norm, steps, restarts)
```

## 6) Evaluation Outputs

```text
Console:
  [INFO] clean metrics
  [ATTACK] per-attack acc/loss/runtime

WandB:
  test/acc_clean
  test/worst_robust_acc
  attack comparison bar charts (acc/loss)

JSON payload:
  {
    config, checkpoint, split, seed, deterministic,
    max_batches, clean, robust, failures, runtime_sec, worst_robust_acc
  }
```
