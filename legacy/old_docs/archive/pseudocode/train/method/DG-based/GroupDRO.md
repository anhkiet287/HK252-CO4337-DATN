# GroupDRO

Implementation in repo:
- `src/ardg/training/objectives/groupdro.py` (`GroupDRO`)
- `src/ardg/training/objectives/groupdro_state.py` (`DomainWeightState`)
- `src/ardg/training/objectives/multi_attack_erm.py` (`AttackDomainObjective`)

## Core Idea

`GroupDRO` reuses the fixed attack-domain setup from `MultiAttackERM`, but it
forces `all_domains` and replaces mean aggregation with a learned domain-weight
vector `q`.

## Pseudocode

```text
__init__(cfg, model):
  require model is not None
  require train.domain_strategy or train.groupdro.domain_strategy == "all_domains"

  include_clean = train.groupdro.include_clean
  resolve fixed domains
  build one attack callable per domain

  eta_q = train.groupdro.eta_q or train.groupdro.eta
  init_q = train.groupdro.init_q
  q_state = DomainWeightState(domain_names, eta_q, init_q)

preprocess_batch(batch, model):
  return build_all_domains_batch(batch, model)

compute_loss(model, batch):
  data = as_xy_dict(batch)
  if "x_domains" not in data:
    data = build_all_domains_batch(data, model)

  y = data["y"]
  domain_names = data.get("domain_names", self.domain_names)

  for each domain batch:
    logits_g = model(x_domain_g)
    loss_g = CE(logits_g, y)
    acc_g = mean(argmax(logits_g) == y)

  q = q_state.update(loss_g)
  total_loss = sum(q_g * loss_g)
  mean_acc = mean(acc_g)

  metrics include:
    loss_<group>, acc_<group>
    q_<group>, q_entropy
    avg_group_loss
    worst_group_by_loss
    group_batch_counts
    loss_adv, acc_adv

  return total_loss, metrics

state_dict():
  return {"q_state": q_state.state_dict()}

load_state_dict(state):
  restore q_state
  also accept legacy {"q": tensor} checkpoints
```

## Validation Behavior

Validation stays trainer-driven:
- generic suite evaluation via `Evaluator`
- flattening via `summarize_suite`
- no custom `objective.validate(...)` hook in `GroupDRO`

## Toy Trace (CIFAR-10, batch_size = 2)

```text
Assume train domains = [clean, fgsm_rs, pgd_ce, pgd_dlr]
Assume one CIFAR-10 batch:
  x.shape = [2, 3, 32, 32]
  y = [3, 8]

Step 1: move_to_device(batch, cuda)
  batch -> (x.cuda(), y.cuda())

Step 2: preprocess_batch(batch, model)
  build_all_domains_batch(...)
  batch becomes:
    {
      "x": x_clean,
      "y": y,
      "x_domains": [x_clean, x_fgsm_rs, x_pgd_ce, x_pgd_dlr],
      "domain_names": ["clean", "fgsm_rs", "pgd_ce", "pgd_dlr"],
      "domain_name": "all_domains",
      "domain_batch_counts": {
        "clean": 2,
        "fgsm_rs": 2,
        "pgd_ce": 2,
        "pgd_dlr": 2,
      },
    }

Step 3: compute losses
  clean   -> loss = 0.20, acc = 1.00
  fgsm_rs -> loss = 0.80, acc = 0.50
  pgd_ce  -> loss = 1.10, acc = 0.00
  pgd_dlr -> loss = 1.50, acc = 0.50

  loss_g = [0.20, 0.80, 1.10, 1.50]

Step 4: update q
  q_old = [0.25, 0.25, 0.25, 0.25]
  q_g <- q_g * exp(eta_q * detach(loss_g))
  normalize q

  with eta_q = 0.02:
    q_new ~= [0.2468, 0.2498, 0.2513, 0.2521]

Step 5: weighted total loss
  loss_total = sum(q_g * loss_g) ~= 0.904

Step 6: optimizer step
  backward(loss_total)
  optimizer.step()
```

## Notes

- Harder domains get higher `q` mass after each update.
- `DomainWeightState` owns normalization, initialization, entropy, and checkpoint restore order.
