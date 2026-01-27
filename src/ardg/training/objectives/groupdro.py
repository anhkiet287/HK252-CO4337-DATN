"""GroupDRO objective (requires group ids)."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from ardg.training.losses import compute_loss
from ardg.training.objectives.base import Objective


class GroupDRO(Objective):
    """Implements multiplicative weight updates over groups."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        gd_cfg = cfg.get("train", {}).get("groupdro", {})
        self.eta = float(gd_cfg.get("eta", 0.02))
        self.q: torch.Tensor | None = None

    def _maybe_init_q(self, num_groups: int, device: torch.device) -> None:
        if self.q is None or self.q.numel() != num_groups:
            self.q = torch.ones(num_groups, device=device) / float(num_groups)

    def loss(self, model: Any, batch: Any) -> Tuple[torch.Tensor, Dict[str, float]]:
        images, labels, groups = _unpack_batch(batch)
        device = labels.device
        num_groups = int(groups.max().item()) + 1
        self._maybe_init_q(num_groups, device)

        logits = model(images)
        losses = []
        loss_g = torch.zeros(num_groups, device=device)
        counts = torch.zeros(num_groups, device=device)
        for g in range(num_groups):
            mask = groups == g
            if mask.any():
                l = compute_loss(logits[mask], labels[mask])
                losses.append(l)
                loss_g[g] = l
                counts[g] = mask.sum()
        # Update q with observed groups only
        observed = counts > 0
        if observed.any():
            self.q[observed] = self.q[observed] * torch.exp(self.eta * loss_g[observed].detach())
        self.q = self.q / self.q.sum()

        total = (loss_g * self.q).sum()
        correct = (logits.argmax(dim=1) == labels).sum().item()
        metrics = {
            "loss": float(total.item()),
            "acc": correct / max(labels.size(0), 1),
            "correct": correct,
            "batch_size": labels.size(0),
            "q_max": float(self.q.max().item()),
            "q_min": float(self.q.min().item()),
        }
        # Log first few group losses
        for g in range(min(3, num_groups)):
            metrics[f"loss_g{g}"] = float(loss_g[g].item())
        return total, metrics


def _unpack_batch(batch: Any):
    if isinstance(batch, dict):
        g = batch.get("g", batch.get("y"))
        if g is None:
            raise ValueError("GroupDRO expects group ids; provide batch['g'] or fallback to labels.")
        return batch["x"], batch["y"], g
    if len(batch) != 3:
        images, labels = batch
        return images, labels, labels
    return batch
