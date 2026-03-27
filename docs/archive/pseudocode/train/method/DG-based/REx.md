# REx

Implementation in repo:
- `src/ardg/training/objectives/rex.py` (`REx`)
- `src/ardg/attacks/attack_suite.py` (`build_train_attack`)

## Core Idea

The current `REx` implementation does not build multiple fixed domains. It
computes one batch loss, randomly partitions the batch into pseudo-environments,
and penalizes the variance of per-split losses.

## Pseudocode

```text
__init__(cfg, model):
  lambda_rex = train.rex.lambda
  num_splits = train.rex.num_splits

  if train.adv_training and model is not None:
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
  base_loss = CE(logits, y)

  perm = random permutation of batch indices
  splits = chunk(perm, num_splits)

  per_env_losses = []
  for idx in splits:
    if idx is non-empty:
      per_env_losses.append(CE(logits[idx], y[idx]))

  if <= 1 non-empty split:
    penalty = 0
  else:
    penalty = variance(per_env_losses, unbiased=False)

  total_loss = base_loss + lambda_rex * penalty
  acc = mean(argmax(logits) == y)

  metrics include:
    loss, acc
    loss_clean/acc_clean or loss_adv/acc_adv
    rex_penalty
```

## Notes

- This is a pseudo-environment implementation based on random batch splits.
- When `train.adv_training` is enabled, REx operates on one adversarially perturbed batch rather than a multi-domain batch.
