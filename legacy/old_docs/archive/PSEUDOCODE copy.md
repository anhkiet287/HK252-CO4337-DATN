# Training Pipeline Pseudocode (High -> Low Level)

Flowchart starters:
- `pseudocode/training-flow.mmd`
- `pseudocode/groupdro-inner-loop.mmd`
- `pseudocode/evaluation-flow.mmd`

## 1) High Level (End-to-End)

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
   - `validate()` -> uses Evaluator + `summarize_suite` (`attack.val_suite` -> `attack.val` -> `attack.eval_suite` -> default clean+pgd20) → prefixed `val/*` (+ objective extras)
   - compute checkpoint selection vector
   - call optional `objective.on_epoch_end(...)`
   - save `best.pt` when selection vector improves
   - always save `last.pt`
   - for `multi_attack_erm` / `groupdro`, optionally save `best_worst.pt` and `best_avg.pt`
   - update early stopping state
   - step scheduler once per epoch
10. Finish and return checkpoint paths (`best.pt`, `last.pt`).

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

  if start_epoch > epochs:
    return existing last/best paths

  for epoch in [start_epoch .. epochs]:
    train_metrics = train_one_epoch(epoch)
    log(train_metrics + {time_sec, device}, split="train")

    val_metrics = validate(epoch)  # returns one flat dict of prefixed val/* metrics
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
  batch = move_to_device(batch, device)
  batch = objective.preprocess_batch(batch, model)

  optimizer.zero_grad(set_to_none=True)
  loss, metrics = objective.compute_loss(model, batch)
  loss.backward()
  optimizer.step()

  metrics["optim/lr"] = optimizer.param_groups[0]["lr"]
  return metrics
```

Implementation in repo:
- Train step: `src/ardg/training/trainer.py` (`Trainer._train_step`)
- Objective hooks contract: `src/ardg/training/objectives/base.py` (`preprocess_batch`, `compute_loss`, `validate`, `on_epoch_end`, `state_dict`, `load_state_dict`)

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

### Shared Attack-Domain Setup (Multi-Attack ERM + GroupDRO)
```text
resolve fixed train domains from:
  preferred: attack.train_domains
  legacy:    attack.multi_train.domains

normalize domains:
  optionally inject clean when include_clean=True
  require unique domain names
  infer/enforce one shared threat model (eps, norm) across non-clean domains
  build one attack callable per domain
```
Implementation in repo:
- `src/ardg/training/objectives/multi_attack_erm.py` (`resolve_attack_domains_cfg`, `AttackDomainObjective`)

### Multi-Attack ERM
```text
preprocess_batch:
  strategy=per_batch:
    sample 1 domain for whole batch
  strategy=split_batch:
    sample one domain per sample in the same batch
  strategy=all_domains:
    generate one attacked batch per domain

loss:
  per_batch/split_batch:
    CE on attacked batch
  all_domains:
    compute CE per domain batch
    total_loss = mean(domain_losses)
    acc = mean(domain_accuracies)

validate:
  loop over val loader and over fixed train domains
  compute:
    val/loss_<domain>, val/acc_<domain>
    val/acc_avg, val/acc_worst, val/worst_domain
  if clean domain exists:
    also expose val/acc_clean
  optional probe:
    val/pgd20_probe_acc, val/pgd20_probe_loss

metrics:
  per_batch/split_batch:
    loss/acc + loss_adv/acc_adv + domain counts
  all_domains:
    per-domain loss_<name>/acc_<name>
    loss_total/acc_total
    avg_group_loss, worst_group_by_loss
```
Implementation in repo:
- `src/ardg/training/objectives/multi_attack_erm.py` (`MultiAttackERM`)
- Train-domain attack construction: `src/ardg/attacks/attack_suite.py` (`build_attack`)

### GroupDRO
```text
preprocess_batch:
  require train.domain_strategy or train.groupdro.domain_strategy = all_domains
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
  objective state saves/restores q via DomainWeightState
```
Implementation in repo:
- `src/ardg/training/objectives/groupdro.py` (`GroupDRO`)
- `src/ardg/training/objectives/groupdro_state.py` (`DomainWeightState`)

#### GroupDRO Toy Trace (CIFAR-10, batch_size = 2)

```text
Assume train domains = [clean, fgsm_rs, pgd_ce, pgd_dlr]
Assume one CIFAR-10 batch from DataLoader:

  batch = (x, y)
  x.shape = [2, 3, 32, 32]
  y.shape = [2]
  y = [3, 8]

Step 1: move_to_device(batch, cuda)
  batch stays structurally the same:
    (
      x.cuda(),   # [2, 3, 32, 32]
      y.cuda(),   # [2]
    )

Step 2: preprocess_batch(batch, model)
  GroupDRO always calls build_all_domains_batch(...)
  batch becomes:

    {
      "x": x_clean,          # [2, 3, 32, 32]
      "y": y,                # [2]
      "x_domains": [
        x_clean,             # [2, 3, 32, 32]
        x_fgsm_rs,           # [2, 3, 32, 32]
        x_pgd_ce,            # [2, 3, 32, 32]
        x_pgd_dlr,           # [2, 3, 32, 32]
      ],
      "domain_names": ["clean", "fgsm_rs", "pgd_ce", "pgd_dlr"],
      "domain_name": "all_domains",
      "domain_batch_counts": {
        "clean": 2,
        "fgsm_rs": 2,
        "pgd_ce": 2,
        "pgd_dlr": 2,
      },
    }

  Interpretation:
    - still the same 2 source samples
    - each sample is expanded into 4 domain views
    - total views processed by the objective = 2 * 4 = 8

Step 3: compute_loss(model, batch)
  labels = data["y"] = [3, 8]

  forward on each domain batch:
    clean   -> loss_clean   = 0.20, acc_clean   = 1.00
    fgsm_rs -> loss_fgsm_rs = 0.80, acc_fgsm_rs = 0.50
    pgd_ce  -> loss_pgd_ce  = 1.10, acc_pgd_ce  = 0.00
    pgd_dlr -> loss_pgd_dlr = 1.50, acc_pgd_dlr = 0.50

  therefore:
    loss_g = [0.20, 0.80, 1.10, 1.50]

Step 4: q initialization and update
  if q is not initialized yet:
    q_old = [0.25, 0.25, 0.25, 0.25]

  with eta_q = 0.02:
    q_g <- q_g * exp(eta_q * detach(loss_g))
    normalize q

  numerically:
    exp(0.02 * loss_g)
      = [exp(0.004), exp(0.016), exp(0.022), exp(0.030)]
      ≈ [1.0040, 1.0161, 1.0222, 1.0305]

    q_new ≈ [0.2468, 0.2498, 0.2513, 0.2521]

  interpretation:
    - hardest domain (pgd_dlr) gets largest q
    - easiest domain (clean) gets smallest q

Step 5: weighted total loss
  loss_total = sum(q_g * loss_g)
             ≈ 0.2468*0.20 + 0.2498*0.80 + 0.2513*1.10 + 0.2521*1.50
             ≈ 0.904

Step 6: optimizer step
  backward(loss_total)
  optimizer.step()

Result:
  the model is updated using a weighted combination of all 4 attack-domains,
  with slightly larger emphasis on domains that currently have larger loss.
```

### GroupDRO++
```text
preprocess_batch:
  optional single-attack adversarial preprocessing if train.adv_training is enabled

loss:
  run batch-wise clustering on logits -> cluster_ids
  compute loss per discovered cluster: loss_g
  update q over observed clusters:
    q_g <- q_g * exp(eta * loss_g)
    normalize q
  group_weighted = sum(q_g * loss_g)
  reg = mean( CE(logits, y) * (q[cluster_ids] ^ gamma) )
  total_loss = group_weighted + lambda_reg * reg
metrics:
  loss/acc + loss_clean/acc_clean (or _adv)
  q_max, q_min, reg
  loss_g0, loss_g1, loss_g2 for the first observed clusters
```
Implementation in repo:
- `src/ardg/training/objectives/groupdro_plus.py` (`GroupDROPlus`)
- Cluster utility used by mode: `src/ardg/training/cluster_utils.py`

### REx
```text
preprocess_batch:
  if train.adv_training and one train attack is configured:
    replace x with one adversarially attacked batch
  else:
    no-op

loss:
  logits = model(x)
  base_loss = CE(logits, y)
  permute batch indices
  split the permutation into num_splits pseudo-environments
  penalty = variance( CE(logits[idx], y[idx]) over non-empty splits )
  total_loss = base_loss + lambda_rex * penalty

metrics:
  loss/acc + loss_clean/acc_clean (or _adv)
  rex_penalty
```
Implementation in repo:
- `src/ardg/training/objectives/rex.py` (`REx`)

## 5) Checkpoint Selection

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
- Selection vector (config/default) + comparator: `src/ardg/training/trainer.py` (`_resolve_selection_names`, `_is_better_checkpoint`)

# Evaluation Pipeline Pseudocode (High -> Low Level)

## 1) High Level (End-to-End)

1. Parse CLI args (`--config`, `--checkpoint`, `--max-batches`, `--num-workers`, `--max-test-samples`, `--smoke-one-sample`, `--seed`, `--deterministic`, `--save-json`, `--platform`, `--verbose`).
2. Apply runtime platform override when requested (`--platform`).
3. Setup run with eval suffix (`run_name + "_eval"`).
4. Resolve deterministic settings:
   - seed from CLI or config
   - deterministic from CLI or config
5. Apply dataloader/runtime overrides:
   - `num_workers` (force `0` when deterministic and not explicitly set)
   - `max_test_samples`
   - smoke mode (`--smoke-one-sample`)
6. Resolve checkpoint path:
   - explicit `--checkpoint`
   - else try `<eval_run_dir>/best.pt`, `<eval_run_dir>/last.pt`
   - then fall back to the original training run dir when `run_name` ends with `_eval`
7. Build model and load checkpoint weights.
8. Build test loader.
9. Resolve `max_batches` from CLI, else `attack.eval_suite.max_batches`, else `attack.eval.max_batches`.
10. Run clean evaluation.
11. Build evaluation attack suite from config (`attack.eval_suite`, else legacy `attack.eval` + `attack.autoattack`).
12. Run adversarial evaluation for each attack in suite and track per-attack runtime / failures.
13. Aggregate summary metrics and log to console + wandb.
14. Save JSON report (`eval_test_summary.json` or `--save-json`).

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

  cli_max_batches = args.max_batches
  if args.smoke_one_sample and cli_max_batches is None:
    cli_max_batches = 1
  max_batches = resolve_max_batches(cfg, cli_max_batches)
  evaluator = Evaluator(model, test_loader, device, max_batches=max_batches)

  clean = evaluator.evaluate_clean()
  attacks = build_eval_suite(cfg, model)

  robust = {}
  failures = {}
  for (label, attack) in attacks.items():
    try:
      metrics = evaluator.evaluate_under_attack(attack)
      metrics["runtime_sec"] = per_attack_elapsed_time
      robust[label] = metrics
    except Exception as exc:
      failures[label] = str(exc)

  summary = summarize_suite({"clean": clean, **robust}, prefix="test")
  log test metrics + attack charts + system metrics
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
    if disabled:
      return {}
    parse a list, a single-attack mapping, or suite.attacks
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
  [INFO] checkpoint / seed / deterministic / max_test_samples
  [INFO] clean metrics
  [ATTACK] per-attack acc/loss/runtime
  [ERROR] failed attacks (if any)
  [INFO] saved=<json_path>

WandB:
  test/acc_clean
  test/loss_clean
  test/acc_<attack>, test/loss_<attack>
  test/acc_avg, test/acc_worst, test/worst_domain
  test/runtime_sec
  test/attack_compare_table
  attack comparison bar charts (acc/loss)
  sys/runtime_sec, sys/n_samples_eval, sys/torch_version, sys/num_attacks, optional sys/gpu_name

JSON payload:
  {
    config, checkpoint, split, seed, deterministic,
    max_batches, clean, robust, summary, failures, runtime_sec, worst_robust_acc
  }
```

Implementation in repo:
- Console printing + JSON write: `scripts/evaluate.py`
- Structured metric logging: `src/ardg/utils/logging.py` (`log_metrics`)
