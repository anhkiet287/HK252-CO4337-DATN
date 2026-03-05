"""Risk Extrapolation (REx) draft objective."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from ardg.training.losses import compute_loss
from ardg.training.objectives.base import Objective
from ardg.attacks.attack_suite import build_train_attack
from ardg.utils.batch import unpack_xy


class REx(Objective):
    """REx with per-batch environment splits."""

    def __init__(self, cfg: Dict[str, Any], model: Any | None = None) -> None:
        rex_cfg = cfg.get("train", {}).get("rex", {})
        self.lambda_rex = float(rex_cfg.get("lambda", 1.0))
        self.num_splits = int(rex_cfg.get("num_splits", 2))
        self.attack = None
        if cfg["train"].get("adv_training", False) and model is not None:
            self.attack = build_train_attack(cfg, model)

    def preprocess_batch(self, batch: Any, model: Any) -> Any:
        if self.attack is None:
            return batch
        images, labels = unpack_xy(batch)
        model.eval()
        adv = self.attack(images, labels)
        model.train()
        adv = adv.detach()
        if isinstance(batch, dict):
            newb = dict(batch)
            newb["x"] = adv
            return newb
        return adv, labels

    def loss(self, model: Any, batch: Any) -> Tuple[torch.Tensor, Dict[str, float]]:
        images, labels = unpack_xy(batch)
        logits = model(images)
        base_loss = compute_loss(logits, labels)

        # Randomly partition batch into num_splits pseudo-environments.
        perm = torch.randperm(labels.size(0), device=labels.device)
        per_env_losses = []
        splits = torch.chunk(perm, self.num_splits)
        for idx in splits:
            if idx.numel() == 0:
                continue
            env_logits = logits[idx]
            env_labels = labels[idx]
            per_env_losses.append(compute_loss(env_logits, env_labels))

        if len(per_env_losses) <= 1:
            penalty = torch.zeros_like(base_loss)
        else:
            stacked = torch.stack(per_env_losses)
            penalty = stacked.var(unbiased=False)

        loss = base_loss + self.lambda_rex * penalty
        correct = (logits.argmax(dim=1) == labels).sum().item()
        metrics = {
            "loss": float(loss.item()),
            "acc": correct / max(labels.size(0), 1),
            "rex_penalty": float(penalty.item()),
            "correct": correct,
            "batch_size": labels.size(0),
        }
        return loss, metrics
