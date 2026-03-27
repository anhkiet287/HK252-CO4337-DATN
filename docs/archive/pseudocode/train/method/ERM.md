# ERM

Implementation in repo:
- `src/ardg/training/objectives/erm.py` (`ERM`)

## Pseudocode

```text
preprocess_batch:
  no-op

compute_loss(model, batch):
  x, y = unpack_xy(batch)
  logits = model(x)
  loss = CE(logits, y)
  acc = mean(argmax(logits) == y)

  metrics = {
    loss,
    acc,
    loss_clean,
    acc_clean,
    correct,
    batch_size,
  }
  return loss, metrics
```

## Notes

- This is the baseline objective.
- Trainer-level epoch aggregation will expose `train/loss_clean` and `train/acc_clean`.
