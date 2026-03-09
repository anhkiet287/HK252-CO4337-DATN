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
        self.q_state = DomainWeightState(self.domain_names, eta_q=self.eta_q, init_q=self.init_q)

    @property
    def q(self) -> torch.Tensor | None:
        return self.q_state.q

    def preprocess_batch(self, batch: Any, model: Any) -> Any:
        return self.build_all_domains_batch(batch, model)

    def compute_loss(self, model: Any, batch: Any) -> Tuple[torch.Tensor, Dict[str, Any]]:
        data = as_xy_dict(batch)
        if "x_domains" not in data:
            data = self.build_all_domains_batch(data, model)

        labels = data["y"]
        batch_size = int(labels.size(0))
        domain_names = data.get("domain_names", self.domain_names)
        stats = self._compute_all_domain_stats(model, data["x_domains"], labels, domain_names)
        loss_g = stats["domain_losses"]
        q = self.q_state.update(loss_g)
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
        metrics.update(stats["metrics"])

        for domain_name, q_value in zip(self.domain_names, q.tolist()):
            metrics[f"q_{domain_name}"] = float(q_value)

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
