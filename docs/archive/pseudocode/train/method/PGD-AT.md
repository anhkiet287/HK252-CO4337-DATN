# PGD-AT

Implementation in repo:
- `src/ardg/training/objectives/pgd_at.py` (`PGDAT`)
- `src/ardg/attacks/attack_suite.py` (`build_train_attack`)

## Pseudocode

```text
__init__(cfg, model):
  attack = build_train_attack(cfg, model)

preprocess_batch(batch, model):
  x, y = unpack_xy(batch)
  model.eval()
  x_adv = attack(x, y)
  model.train()
  x_adv = detach(x_adv)

  if batch is a dict:
    return batch with x replaced by x_adv
  return (x_adv, y)

compute_loss(model, batch):
  x_adv, y = unpack_xy(batch)
  logits = model(x_adv)
  loss = CE(logits, y)
  acc = mean(argmax(logits) == y)

  metrics = {
    loss,
    acc,
    loss_adv,
    acc_adv,
    correct,
    batch_size,
  }
  return loss, metrics
```

## Notes

- The attack is generated during `preprocess_batch`, not inside `compute_loss`.
- This path expects exactly one configured train attack.
