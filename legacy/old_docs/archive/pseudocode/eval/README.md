# Evaluation Pipeline

This document summarizes the current test-time evaluation path driven by
`scripts/evaluate.py`.

## Flowchart Starter

- `evaluation-flow.mmd`

## High Level

1. Parse CLI args (`--config`, `--checkpoint`, `--max-batches`, `--num-workers`, `--max-test-samples`, `--smoke-one-sample`, `--seed`, `--deterministic`, `--save-json`, `--platform`, `--verbose`).
2. Apply runtime platform override when requested (`--platform`).
3. Setup run with eval suffix (`run_name + "_eval"`).
4. Resolve deterministic settings.
   - seed from CLI or config
   - deterministic from CLI or config
5. Apply dataloader/runtime overrides.
   - `num_workers` (force `0` when deterministic and not explicitly set)
   - `max_test_samples`
   - smoke mode (`--smoke-one-sample`)
6. Resolve checkpoint path.
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
- `scripts/evaluate.py`
- `src/ardg/experiments/common.py`
- `src/ardg/evaluation/evaluator.py`
- `src/ardg/evaluation/summary.py`
- `src/ardg/attacks/attack_suite.py`

## Main Loop

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

## Evaluator Core

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
- `src/ardg/evaluation/evaluator.py`

## Attack Suite Resolution

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

## Outputs

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
