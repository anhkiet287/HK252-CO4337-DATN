# Training Pipeline Pseudocode (High -> Low Level)

## 1) High Level (End-to-End)

1. Load config.
2. Apply runtime platform override when requested.
   - `--platform local` -> `/content/HK252-CO4337-DATN/...`
   - `--platform colab` -> `/content/drive/MyDrive/HK252-CO4337-DATN/...`
   - override `dataset.data_dir` and `logging.output_dir`
3. Setup run.
   - set seed / deterministic mode
   - resolve device (cuda/cpu)
   - init logger + wandb
4. Build dataloaders (train/val).
5. Build model.
6. Build objective from `train.mode`.
   - `erm` / `pgd_at` / `multi_attack_erm` / `rex` / `groupdro` / `groupdro_plus`
7. Create trainer.
8. For each epoch:
   - `train_one_epoch()` -> logs `train/loss_{clean|adv}`, `train/acc_{clean|adv}`, `optim/lr`
   - `validate()` -> uses Evaluator + `summarize_suite` (uses `attack.val_suite` when present; defaults to clean+pgd20) → prefixed `val/*` (+ objective extras)
   - choose/save best checkpoint by selection metric (plus best_worst/best_avg for `multi_attack_erm` and `groupdro`)
   - save last checkpoint
   - scheduler step
9. Finish and return checkpoint paths (`best.pt`, `last.pt`).

Implementation in repo:
- CLI + orchestration: `scripts/train.py:main`
- Run setup / dataloaders: `src/ardg/experiments/common.py` (`setup_run`, `build_loaders`)
- Model construction: `src/ardg/models/factory.py` (`build_model`)
- Objective dispatch: `src/ardg/training/objectives/__init__.py` (`build_objective`)
- Training loop entry: `src/ardg/training/trainer.py` (`Trainer.train`)
- Validation summarizer: `src/ardg/evaluation/evaluator.py` + `src/ardg/evaluation/summary.py`

## 2) Mid Level (Trainer Loop)

```text
train():
  best_vector = None

  for epoch in [start_epoch .. epochs]:
    train_metrics = train_one_epoch(epoch)
    val_metrics = validate(epoch)  # returns one flat dict of prefixed val/* metrics

    selection_names = resolve_selection_vector(cfg, val_metrics)
    metric_name = selection_names[0]
    metric_vector = tuple(val_metrics[name] for name in selection_names)
    metric_value = metric_vector[0]

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

    if train_mode == "multi_attack_erm":
      maybe_save("best_worst.pt", metric=val_metrics.get("val/acc_worst"))
      maybe_save("best_avg.pt",   metric=val_metrics.get("val/acc_avg"))

    if scheduler.step_unit == "epoch":
      scheduler.step()
```

Implementation in repo:
- `src/ardg/training/trainer.py`:
  - `Trainer.train`
  - `Trainer.train_one_epoch`
  - `Trainer.validate`
  - `Trainer._is_better_checkpoint`
  - `Trainer._save_checkpoint`

## 3) Low Level (Single Train Step)

```text
_train_step(batch):
  batch = move_to_device(batch)
  batch = objective.preprocess_batch(batch, model)

  optimizer.zero_grad()
  loss, metrics = objective.compute_loss(model, batch)
  loss.backward()
  optimizer.step()

  metrics["optim/lr"] = optimizer.lr
  if scheduler.step_unit == "step":
    scheduler.step()
  return metrics
```

Implementation in repo:
- Train step: `src/ardg/training/trainer.py` (`Trainer._train_step`)
- Objective hooks contract: `src/ardg/training/objectives/base.py` (`preprocess_batch`, `compute_loss`, `state_dict`, `load_state_dict`)

## 4) Objective Logic by Mode

### ERM
```text
preprocess_batch: no-op
loss: CE(model(x), y)
metrics: loss/acc + loss_clean/acc_clean
```
Implementation in repo:
- `src/ardg/training/objectives/erm.py` (`ERM`)

### PGD-AT
```text
preprocess_batch: x_adv = PGD(model, x, y)
loss: CE(model(x_adv), y)
metrics: loss/acc + loss_adv/acc_adv
```
Implementation in repo:
- `src/ardg/training/objectives/pgd_at.py` (`PGDAT`)
- Attack builder used by objective: `src/ardg/attacks/attack_suite.py` (`build_train_attack`)

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
  suite-based via Evaluator + summarize_suite
  optional probe:
    val/pgd20_probe_acc, val/pgd20_probe_loss (objective hook)
metrics: loss/acc keyed by domain; aggregates in summarize_suite
```
Implementation in repo:
- `src/ardg/training/objectives/multi_attack_erm.py` (`MultiAttackERM`)
- Train-domain attack construction: `src/ardg/attacks/attack_suite.py` (`build_attack`)

### GroupDRO
```text
preprocess_batch:
  require strategy = all_domains
  build one attacked batch per fixed train domain from the same clean batch
  group = domain

loss:
  compute per-domain loss and acc: loss_g, acc_g
  update q:
    q_g <- q_g * exp(eta_q * detach(loss_g))
    normalize q
  total_loss = sum(q_g * loss_g)

state:
  save/load q through objective.state_dict()

metrics:
  train/loss_<group>, train/acc_<group>
  train/q_<group>, train/q_entropy
  train/avg_group_loss, train/worst_group_by_loss
  validation stays suite-based via Evaluator + summarize_suite
```
Implementation in repo:
- `src/ardg/training/objectives/groupdro.py` (`GroupDRO`)
- `src/ardg/training/objectives/groupdro_state.py` (`DomainWeightState`)

### GroupDRO++
```text
preprocess_batch:
  optional adversarial preprocessing if enabled

loss:
  run batch-wise clustering on model features/logits -> cluster_ids
  compute loss per discovered cluster: loss_g
  update q over observed clusters:
    q_g <- q_g * exp(eta * loss_g)
    normalize q
  group_weighted = sum(q_g * loss_g)
  reg = mean( CE(logits, y) * (q[cluster_ids] ^ gamma) )
  total_loss = group_weighted + lambda_reg * reg
metrics: loss/acc + loss_clean/acc_clean (or _adv)
```
Implementation in repo:
- `src/ardg/training/objectives/groupdro_plus.py` (`GroupDROPlus`)
- Cluster utility used by mode: `src/ardg/training/cluster_utils.py`

### REx
```text
preprocess_batch:
  build multi-domain attacked batch (same strategy as multi-attack)

loss:
  compute per-domain losses loss_d
  mean_loss = mean(loss_d)
  var_loss = variance(loss_d)
  total_loss = mean_loss + lambda_var * var_loss

metrics:
  loss_total
  loss_mean_domains
  loss_var_domains
  acc per domain (acc_{name} or acc_adv)
```
Implementation in repo:
- `src/ardg/training/objectives/rex.py` (`REx`)

## 5) Checkpoint Selection

```text
input: val_metrics

selection_names = train.selection.vector or ["val/acc_worst","val/acc_avg","val/acc_clean"]
vector = [val_metrics_prefixed[name] for name in selection_names]
is_better: lexicographic compare(candidate_vector, best_vector, eps=1e-6)
multi-attack/groupdro additionally save:
  best_worst.pt (val/acc_worst)
  best_avg.pt   (val/acc_avg)
```

Implementation in repo:
- Selection vector (config/default) + comparator: `src/ardg/training/trainer.py` (`resolve_selection_vector`, `_is_better_checkpoint`)

# Evaluation Pipeline Pseudocode (High -> Low Level)

## 1) High Level (End-to-End)

1. Parse CLI args (`--config`, `--checkpoint`, `--deterministic`, `--seed`, etc.).
2. Apply runtime platform override when requested (`--platform`).
3. Setup run with eval suffix (`run_name + "_eval"`).
4. Resolve deterministic settings:
   - seed from CLI or config
   - deterministic from CLI or config
5. Apply dataloader/runtime overrides:
   - `num_workers`
   - `max_test_samples`
   - smoke mode (`--smoke-one-sample`)
6. Resolve checkpoint path:
   - explicit `--checkpoint`, else try `best.pt`, then `last.pt`.
7. Build model and load checkpoint weights.
8. Build test loader.
9. Build evaluation attack suite from config (`attack.eval_suite`).
10. Run clean evaluation.
11. Run adversarial evaluation for each attack in suite.
12. Aggregate summary metrics and log to console + wandb.
13. Save JSON report (`eval_test_summary.json` or `--save-json`).

Implementation in repo:
- CLI + orchestration: `scripts/evaluate.py:main`
- Run setup / loaders / checkpoint model load: `src/ardg/experiments/common.py`
- Attack suite resolution: `src/ardg/attacks/attack_suite.py` (`build_eval_suite`)
- Evaluator runtime: `src/ardg/evaluation/evaluator.py` (`Evaluator`)

## 2) Mid Level (Evaluate Main Loop)

```text
evaluate_main():
  args = parse_args()
  cfg, logger, run, device = setup_run(
    config,
    run_name_suffix="eval",
    platform_override=args.platform,
  )

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

  worst_robust_acc = min(robust[*].acc_adv, default=clean_metrics.acc_clean)
  log test metrics + system metrics
  save payload JSON
```

Implementation in repo:
- `scripts/evaluate.py`:
  - `main`
  - `_resolve_checkpoint`
  - `_resolve_max_batches`
  - `_resolve_eval_seed`
  - `_resolve_eval_deterministic`
  - `_log_attack_comparison_chart`

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

Implementation in repo:
- `src/ardg/evaluation/evaluator.py`:
  - `Evaluator.evaluate_clean`
  - `Evaluator.evaluate_under_attack`
  - `Evaluator.evaluate_suite`

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

Implementation in repo:
- Suite parser/builders: `src/ardg/attacks/attack_suite.py`
- Concrete attack wrappers:
  - `src/ardg/attacks/fgsm.py`
  - `src/ardg/attacks/pgd.py`
  - `src/ardg/attacks/cw.py`
  - `src/ardg/attacks/deepfool.py`
  - `src/ardg/attacks/fab.py`
  - `src/ardg/attacks/square.py`
  - `src/ardg/attacks/autoattack_ta.py`

## 5) Determinism + Fairness Checklist

```text
Must keep fixed across model comparisons:
  - eval config (same attack.eval_suite)
  - seed
  - deterministic flag
  - max_batches / max_test_samples
  - threat model params (eps, norm, steps, restarts)
```

Implementation in repo:
- Seed/determinism utility: `src/ardg/utils/seed.py` (`set_seed`)
- Eval deterministic overrides: `scripts/evaluate.py`
- Train deterministic setup + runtime path override: `src/ardg/experiments/common.py` (`load_runtime_config`, `setup_run`)

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

Implementation in repo:
- Console printing + JSON write: `scripts/evaluate.py`
- Structured metric logging: `src/ardg/utils/logging.py` (`log_metrics`)
