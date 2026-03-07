"""GroupDRO++ draft: per-batch automated group discovery via clustering."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from ardg.training.cluster_utils import run_kmeans
from ardg.training.losses import compute_loss
from ardg.training.objectives.base import Objective
from ardg.attacks.attack_suite import build_train_attack
from ardg.utils.batch import unpack_xy


class GroupDROPlus(Objective):
    """GroupDRO with on-the-fly clustering (batch-wise approximation)."""

    def __init__(self, cfg: Dict[str, Any], model: Any) -> None:
        gd_cfg = cfg.get("train", {}).get("groupdro_plus", {})
        self.num_clusters = int(gd_cfg.get("num_clusters", 4))
        self.eta = float(gd_cfg.get("eta", 0.02))
        self.lambda_reg = float(gd_cfg.get("lambda_reg", 0.5))
        self.gamma = float(gd_cfg.get("gamma", 1.0))
        self.q: torch.Tensor | None = None
        self.attack = None
        if cfg["train"].get("adv_training", False):
            self.attack = build_train_attack(cfg, model)

    def _maybe_init_q(self, num_groups: int, device: torch.device) -> None:
        if self.q is None or self.q.numel() != num_groups:
            self.q = torch.ones(num_groups, device=device) / float(num_groups)

    def compute_loss(self, model: Any, batch: Any) -> Tuple[torch.Tensor, Dict[str, float]]:
        images, labels = unpack_xy(batch)
        if self.attack is not None:
            model.eval()
            images = self.attack(images, labels).detach()
            model.train()
        device = labels.device

        logits = model(images)
        with torch.no_grad():
            cluster_ids = run_kmeans(logits.detach(), self.num_clusters)
        num_groups = int(cluster_ids.max().item()) + 1
        self._maybe_init_q(num_groups, device)

        loss_g = torch.zeros(num_groups, device=device)
        counts = torch.zeros(num_groups, device=device)
        for g in range(num_groups):
            mask = cluster_ids == g
            if mask.any():
                lg = compute_loss(logits[mask], labels[mask])
                loss_g[g] = lg
                counts[g] = mask.sum()
        observed = counts > 0
        if observed.any():
            self.q[observed] = self.q[observed] * torch.exp(self.eta * loss_g[observed].detach())
        self.q = self.q / self.q.sum()

        group_weighted = (loss_g * self.q).sum()
        reg = (compute_loss(logits, labels) * (self.q[cluster_ids] ** self.gamma)).mean()
        total_loss = group_weighted + self.lambda_reg * reg

        correct = (logits.argmax(dim=1) == labels).sum().item()
        is_adv = self.attack is not None
        metrics = {
            "loss": float(total_loss.item()),
            "acc": correct / max(labels.size(0), 1),
            ("loss_adv" if is_adv else "loss_clean"): float(total_loss.item()),
            ("acc_adv" if is_adv else "acc_clean"): correct / max(labels.size(0), 1),
            "correct": correct,
            "batch_size": labels.size(0),
            "q_max": float(self.q.max().item()),
            "q_min": float(self.q.min().item()),
            "reg": float(reg.item()),
        }
        for g in range(min(3, num_groups)):
            metrics[f"loss_g{g}"] = float(loss_g[g].item())
        return total_loss, metrics

    def state_dict(self) -> Dict[str, Any]:
        return {"q": self.q} if self.q is not None else {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        q = state.get("q")
        if q is not None:
            self.q = q
