"""GroupDRO objective over fixed attack domains."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from ardg.training.objectives.multi_attack_erm import AttackDomainObjective
from ardg.training.objectives.groupdro_state import (
    DomainWeightState,
    q_entropy,
    weighted_group_loss,
)
from ardg.utils.batch import as_xy_dict


class GroupDRO(AttackDomainObjective):
    """GroupDRO v1 = all-domains multi-attack training with q-weighted aggregation."""

    def __init__(self, cfg: Dict[str, Any], model: Any | None = None) -> None:
        if model is None:
            raise ValueError("GroupDRO requires a model to build fixed attack domains.")

        train_cfg = cfg.get("train", {})
        gd_cfg = train_cfg.get("groupdro", {})
        strategy = str(train_cfg.get("domain_strategy", gd_cfg.get("domain_strategy", "all_domains"))).lower()
        if strategy != "all_domains":
            raise ValueError(
                f"GroupDRO v1 only supports train.domain_strategy='all_domains', got {strategy!r}."
            )

        self.cfg = cfg
        self.domain_strategy = strategy
        self.include_clean = bool(gd_cfg.get("include_clean", True))
        super().__init__(cfg, model, include_clean=self.include_clean)

        self.eta_q = float(gd_cfg.get("eta_q", gd_cfg.get("eta", 0.02)))
        self.init_q = str(gd_cfg.get("init_q", "uniform"))

        # Initialize q_state with the domain names, eta_q, and init_q.
        self.q_state = DomainWeightState(self.domain_names, eta_q=self.eta_q, init_q=self.init_q)

    @property
    def q(self) -> torch.Tensor | None:
        return self.q_state.q

    def preprocess_batch(self, batch: Any, model: Any) -> Any:
        return self.build_all_domains_batch(batch, model)

    def compute_loss(self, model: Any, batch: Any) -> Tuple[torch.Tensor, Dict[str, Any]]:
        data = as_xy_dict(batch) # Ensure data is in dict format with "x" and "y" keys, shape (batch_size, ...).
        # for example: cifar10, batch size 64, x shape [64,3,32,32], y shape [64] then
        # x: (batch_size, ...) tensor of input features, [64,3,32,32] .
        # y: (batch_size, ...) tensor of labels, e.g [64]

        # If x_domains is not already in the batch, build it using the model and the input data.
        # however for groupdro we expect x_domains as default as all domain strategy is used.
        if "x_domains" not in data:
            data = self.build_all_domains_batch(data, model)

        labels = data["y"] # shape [64]
        batch_size = int(labels.size(0)) # shape () scalar tensor with value 64
        domain_names = data.get("domain_names", self.domain_names) # list of domain names, e.g. ["clean", "attack1", "attack2"].

        # Compute losses for each domain and update q weights accordingly.
        stats = self._compute_all_domain_stats(model, data["x_domains"], labels, domain_names) 

        # stats is a dict with keys:
        # "domain_losses": tensor of shape [num_domains] with the average loss for each domain, e.g. [0.5, 1.0, 1.5]
        # "mean_acc": scalar tensor with the mean accuracy across all domains, e.g. 0.75
        # "avg_group_loss": scalar tensor with the average loss across all domains, e.g. 1.0
        # "worst_group_by_loss": string with the name of the domain with the highest loss, e.g. "attack2"
        # "metrics": dict with any additional metrics computed for the batch, e.g. {"acc_clean": 0.8, "acc_attack1": 0.7, "acc_attack2": 0.75}
        loss_g = stats["domain_losses"] 

        # Update q weights based on the domain losses and compute the total loss as a weighted sum of the domain losses.
        q = self.q_state.update(loss_g) 

        # Compute the total loss as a weighted sum of the domain losses using the updated q weights.
        total_loss = weighted_group_loss(q, loss_g)
        mean_acc = stats["mean_acc"]

        metrics: Dict[str, Any] = {
            "loss": float(total_loss.item()),
            "acc": mean_acc,
            "loss_adv": float(total_loss.item()),
            "acc_adv": mean_acc,
            "loss_total": float(total_loss.item()),
            "acc_total": mean_acc,
            "correct": mean_acc * batch_size,
            "batch_size": batch_size,
            "domain_name": str(data.get("domain_name", "all_domains")),
            "domain_batch_counts": data.get("domain_batch_counts", {}),
            "group_batch_counts": data.get("domain_batch_counts", {}),
            "avg_group_loss": stats["avg_group_loss"],
            "worst_group_by_loss": stats["worst_group_by_loss"],
            "q_entropy": float(q_entropy(q).item()),
        }
        # Add any additional metrics computed for the batch to the metrics dict.
        metrics.update(stats["metrics"]) 

        # Add the current q values for each domain to the metrics dict for logging and analysis.
        for domain_name, q_value in zip(self.domain_names, q.tolist()):
            metrics[f"q_{domain_name}"] = float(q_value)

        # Return the total loss and the metrics dict for logging and analysis.
        return total_loss, metrics

    def state_dict(self) -> Dict[str, Any]:
        return {
            "q_state": self.q_state.state_dict(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if "q_state" in state and isinstance(state["q_state"], dict):
            self.q_state.load_state_dict(state["q_state"])
            return
        # Backward compatibility with earlier checkpoints that saved {"q": tensor}.
        if "q" in state:
            self.q_state.load_state_dict({"q": state.get("q"), "domain_names": list(self.domain_names)})
