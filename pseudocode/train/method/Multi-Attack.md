# Multi-Attack ERM

Implementation in repo:
- `src/ardg/training/objectives/multi_attack_erm.py` (`MultiAttackERM`)
- `src/ardg/attacks/attack_suite.py` (`build_attack`)

## Shared Fixed Attack-Domain Setup

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

This shared setup is also reused by `GroupDRO`.

## Preprocess Strategies

```text
strategy=per_batch:
  sample 1 domain for the whole batch
  attack every sample with that domain
  set domain_name = selected_domain
  set domain_batch_counts = {selected_domain: batch_size}

strategy=split_batch:
  sample one domain id per sample
  attack sub-batches under each selected domain
  set domain_name = "mixed"
  set domain_batch_counts = {selected_domain: count_in_batch}

strategy=all_domains:
  build one attacked copy of the batch per domain
  set:
    x_domains = [x_domain_1, ..., x_domain_k]
    domain_names = [name_1, ..., name_k]
    domain_name = "all_domains"
    domain_batch_counts = {name_i: batch_size}
```

## Loss

```text
compute_loss(model, batch):
  data = as_xy_dict(batch)
  y = data["y"]

  if x_domains exists:
    for each domain batch:
      logits_d = model(x_domain_d)
      loss_d = CE(logits_d, y)
      acc_d = mean(argmax(logits_d) == y)

    total_loss = mean(loss_d)
    total_acc = mean(acc_d)
    metrics include:
      loss_<domain>, acc_<domain>
      loss_total, acc_total
      avg_group_loss, worst_group_by_loss
      loss_adv, acc_adv
    return total_loss, metrics

  else:
    logits = model(data["x"])
    loss = CE(logits, y)
    acc = mean(argmax(logits) == y)
    metrics include:
      loss, acc
      loss_adv, acc_adv
      loss_total, acc_total
      domain_batch_counts
    return loss, metrics
```

## Validation Hook

```text
validate(model, loader):
  for each val batch:
    for each fixed train domain:
      if domain == clean:
        attacked = clean images
      else:
        attacked = attack(images, labels)

      logits = model(attacked)
      accumulate domain loss, domain correct, domain seen

  emit:
    loss_<domain>, acc_<domain>
    acc_avg, acc_worst, worst_domain
    acc_clean (if clean domain exists)

  optional probe:
    run a separate PGD probe attack for a capped number of batches
    emit pgd20_probe_loss, pgd20_probe_acc
```

## Notes

- `aggregation="mean"` is the only supported aggregation mode.
- `include_clean` controls whether the clean domain is inserted into the fixed domain list.
- The trainer still runs the generic validation suite; this objective adds extra `val/*` metrics through `objective.validate(...)`.
