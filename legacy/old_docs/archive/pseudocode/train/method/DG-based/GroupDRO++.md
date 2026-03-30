# GroupDRO++

Implementation in repo:
- `src/ardg/training/objectives/groupdro_plus.py` (`GroupDROPlus`)
- `src/ardg/training/cluster_utils.py` (`run_kmeans`)
- `src/ardg/attacks/attack_suite.py` (`build_train_attack`)

## Core Idea

`GroupDRO++` does not rely on fixed attack domains. It clusters logits and then
applies a GroupDRO-style weighting plus a regularization term.

The current implementation supports two clustering modes:
- `cluster_mode: batch`
  - run k-means on the current batch logits every training step
- `cluster_mode: epoch`
  - cluster the full train loader at `on_train_start(...)` and again every
    `epoch_cluster.every_epochs`
  - keep the resulting global centers fixed within each epoch
  - assign each step's logits to the nearest saved center

## Pseudocode

```text
__init__(cfg, model):
  num_clusters = train.groupdro_plus.num_clusters
  eta = train.groupdro_plus.eta
  lambda_reg = train.groupdro_plus.lambda_reg
  gamma = train.groupdro_plus.gamma
  cluster_mode = train.groupdro_plus.cluster_mode  # batch | epoch
  kmeans_iters = train.groupdro_plus.kmeans_iters

  if cluster_mode == epoch:
    epoch_cluster.every_epochs
    epoch_cluster.source  # clean | adv
    epoch_cluster.max_batches

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

  if cluster_mode == batch:
    with no grad:
      cluster_ids = run_kmeans(logits.detach(), num_clusters)
  else:
    cluster_ids = assign_to_centers(logits.detach(), saved_epoch_centers)

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

## Epoch Mode Refresh

```text
on_train_start(loaders):
  if cluster_mode == epoch and centers are missing:
    cluster the train loader once to initialize centers

on_epoch_end(epoch, loaders):
  if cluster_mode == epoch and epoch % every_epochs == 0:
    replay the train loader
    collect logits for the configured source (clean or adv)
    run global k-means once
    store cluster centers for the next epoch
```

## Notes

- Clustering uses logits, not a separate stored embedding bank.
- If adversarial preprocessing is enabled, the clustering runs on adversarially perturbed inputs.
- `state_dict()` stores `q` and, in epoch mode, the saved global centers/counts.

## Tradeoff

- `batch` mode:
  - lowest extra compute
  - adapts immediately to each batch
  - noisier pseudo-groups because clusters are local to one batch
- `epoch` mode:
  - higher compute because it adds an extra pass over the train loader at each refresh
  - if `epoch_cluster.source=adv`, cost is much higher again because attacks are generated during reclustering
  - expected to be more stable because all batches in one epoch share the same global centers

## Optional Snapshot Logging

You can periodically save the clustering space used by `GroupDROPlus` through
`train.groupdro_plus.snapshot`.

```yaml
train:
  groupdro_plus:
    snapshot:
      enabled: true
      every_epochs: 5
      max_batches: 2
      splits: [train, val]
      save_pca: true
      output_subdir: cluster_snapshots
```

Behavior:
- Runs in `on_epoch_end(...)`
- Replays the current model on selected loader batches
- Saves one `.npz` file per batch under:
  - `<run_dir>/cluster_snapshots/<split>/epoch_0005_batch_0001.npz`
- Stored arrays include:
  - `embeddings`
  - `projection_2d` when `save_pca=true`
  - `cluster_ids`
  - `labels`
  - `centers`
  - `counts`
  - `q`
