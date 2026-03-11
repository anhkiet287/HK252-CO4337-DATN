# GroupDRO++

Implementation in repo:
- `src/ardg/training/objectives/groupdro_plus.py` (`GroupDROPlus`)
- `src/ardg/training/cluster_utils.py` (`run_kmeans`)
- `src/ardg/attacks/attack_suite.py` (`build_train_attack`)

## Core Idea

`GroupDRO++` does not rely on fixed attack domains. It discovers groups inside
the current batch by clustering logits, then applies a GroupDRO-style weighting
plus a regularization term.

## Pseudocode

```text
__init__(cfg, model):
  num_clusters = train.groupdro_plus.num_clusters
  eta = train.groupdro_plus.eta
  lambda_reg = train.groupdro_plus.lambda_reg
  gamma = train.groupdro_plus.gamma

  if train.adv_training:
    attack = build_train_attack(cfg, model)
  else:
    attack = None

preprocess_batch(batch, model):
  if attack is None:
    return batch

  x, y = unpack_xy(batch)
  model.eval()
  x_adv = attack(x, y)
  model.train()
  x_adv = detach(x_adv)

  if batch is a dict:
    return batch with x replaced by x_adv
  return (x_adv, y)

compute_loss(model, batch):
  x, y = unpack_xy(batch)
  logits = model(x)

  with no grad:
    cluster_ids = run_kmeans(logits.detach(), num_clusters)

  maybe initialize q over the observed number of clusters

  for each cluster g:
    if cluster g is non-empty:
      loss_g[g] = CE(logits[cluster_ids == g], y[cluster_ids == g])
      counts[g] = number of samples in cluster g

  update q on observed clusters:
    q[g] <- q[g] * exp(eta * detach(loss_g[g]))
    normalize q

  group_weighted = sum(loss_g * q)
  reg = mean( CE(logits, y) * (q[cluster_ids] ^ gamma) )
  total_loss = group_weighted + lambda_reg * reg

  metrics include:
    loss, acc
    loss_clean/acc_clean or loss_adv/acc_adv
    q_max, q_min, reg
    loss_g0, loss_g1, loss_g2 for the first clusters
```

## Notes

- Clustering is batch-wise and uses logits, not a separate stored embedding bank.
- If adversarial preprocessing is enabled, the clustering runs on adversarially perturbed inputs.
- `state_dict()` only stores `q`.
