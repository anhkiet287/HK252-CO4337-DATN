# Lightweight Multi-Attack Training Plan (v1)

## Summary
Implement `multi_attack_erm` with low-overhead defaults for your deadline: default `per_batch` domain sampling, default checkpoint metric `val/pgd20_probe_acc` (cheap probe), and minimal trainer/attack API expansion. Existing modes (`erm`, `pgd_at`, `rex`, `groupdro`, `groupdro_plus`) remain backward compatible.

## Public Interface Changes
1. Add `train.mode: multi_attack_erm`.
2. Add `train.multi_attack.strategy` with values `per_batch | split_batch | all_domains` and default `per_batch`.
3. Add `train.multi_attack.aggregation` with supported value `mean` (strictly validated).
4. Add `train.multi_attack.include_clean` (default `true`; if clean already exists in domain list, dedupe).
5. Add `attack.multi_train.norm`, `attack.multi_train.eps`, and `attack.multi_train.domains` (list of attack-domain specs).
6. Add `val.probe` block for cheap robust validation (`enabled`, `type`, `norm`, `eps`, `steps`, `alpha`, `restarts`, `max_batches`).
7. Add logged metrics `train/domain_name`, `train/domain_count/*`, `val/pgd20_probe_acc`, and `val/pgd20_probe_loss`.
8. Define checkpoint rule: in `multi_attack_erm`, best checkpoint uses `val/pgd20_probe_acc` if present, else `val/acc`; all other modes keep `val/acc`.

## Implementation Scope (Minimal Touch Points)
1. Create `src/ardg/training/objectives/multi_attack_erm.py` to parse config, validate domain setup, build attacks once, run domain assignment strategy, compute CE loss, and expose optional probe-validation metrics.
2. Update `src/ardg/training/objectives/__init__.py` to register `multi_attack_erm`.
3. Extend `src/ardg/attacks/attack_suite.py` with one minimal API: `build_attack(spec, model, dataset_name, shared_norm=None, shared_eps=None)`; keep existing train/val/eval builders unchanged.
4. Extend `src/ardg/attacks/fgsm.py` so `random_start=true` maps to RS-FGSM behavior (`torchattacks.RFGSM`), else regular FGSM.
5. Extend `src/ardg/attacks/pgd.py` to support loss selection for domain specs (`ce` and `dlr`; `dlr` via `UPGD(loss='dlr')`).
6. Add optional `validate(...)` no-op hook in `src/ardg/training/objectives/base.py`.
7. Make minimal change in `src/ardg/training/trainer.py`: keep current clean validation, merge objective hook metrics if present, and switch checkpoint metric only for `multi_attack_erm`.
8. Add example configs for local/colab multi-attack ResNet50 and one smoke config under `configs/smoke_test/`.

## Runtime Behavior
1. `per_batch`: sample one domain uniformly per batch and attack whole batch once.
2. `split_batch`: assign each sample to one domain in the same batch and attack per chunk.
3. `all_domains`: generate all domains per batch and average losses; keep as debug/ablation only.
4. Validation always computes clean metrics; robust probe runs only when `val.probe.enabled` and only for `max_batches`.
5. Domain consistency checks fail fast on duplicate names, invalid strategy/aggregation, or mixed threat-model params.

## Target Config Shape
```yaml
train:
  mode: multi_attack_erm
  multi_attack:
    strategy: per_batch
    include_clean: true
    aggregation: mean

attack:
  multi_train:
    norm: Linf
    eps: 0.0313725
    domains:
      - {name: clean, type: clean}
      - {name: fgsm_rs, type: fgsm, alpha: 0.0313725, random_start: true}
      - {name: pgd_ce,  type: pgd, steps: 10, alpha: 0.007843, loss: ce,  random_start: true}
      - {name: pgd_dlr, type: pgd, steps: 10, alpha: 0.007843, loss: dlr, random_start: true}

val:
  probe:
    enabled: true
    type: pgd
    norm: Linf
    eps: 0.0313725
    steps: 20
    alpha: 0.007843
    restarts: 1
    max_batches: 10
```

## Test Cases and Scenarios
1. Config validation test: rejects duplicate domain names and invalid strategy/aggregation.
2. Threat-model validation test: rejects per-domain `eps`/`norm` mismatches against shared `attack.multi_train`.
3. Attack sanity test: each domain obeys pixel-space Linf <= eps on one batch.
4. Integration smoke test: 2-3 epochs with `multi_attack_erm` logs domain counts and `val/pgd20_probe_acc`.
5. Checkpoint behavior test: best checkpoint in `multi_attack_erm` tracks probe metric, not clean `val/acc`.
6. Regression test: existing `erm` and `pgd_at` configs run unchanged.

## Pipeline and Workflow Summary
1. Data and normalization pipeline stays unchanged.
2. Model factory stays unchanged.
3. Objective layer gets one new mode (`multi_attack_erm`) that maps attacks to domains.
4. Trainer gets a tiny validation-extension path and multi-attack checkpoint rule.
5. Evaluation scripts stay intact; training-time robust model selection uses cheap PGD20 probe.

## Assumptions and Defaults
1. Attack backend remains `torchattacks` only.
2. `fgsm_rs` means RS-FGSM 1-step behavior.
3. `pgd_dlr` uses `UPGD(loss='dlr')`.
4. Default domain strategy is `per_batch`.
5. Default robust checkpoint metric is `val/pgd20_probe_acc` with `max_batches=10`.
6. Full worst-domain validation is intentionally deferred and disabled by default.
