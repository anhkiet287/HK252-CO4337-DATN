"""PGD Adversarial Training objective (matches existing behavior)."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from ardg.attacks.attack_suite import build_train_attack
from ardg.training.losses import compute_loss
from ardg.training.objectives.base import Objective
from ardg.utils.batch import unpack_xy


class PGDAT(Objective):
    """PGD adversarial training."""

    def __init__(self, cfg: Dict[str, Any], model: Any) -> None:
        self.cfg = cfg
        self.attack = build_train_attack(cfg, model)

    def preprocess_batch(self, batch: Any, model: Any) -> Any:
        images, labels = unpack_xy(batch)
        was_training = bool(model.training)
        model.eval()
        with self.full_precision_context():
            adv = self.attack(images, labels)
        model.train(was_training)
        adv = adv.detach()
        if isinstance(batch, dict):
            new_batch = dict(batch)
            new_batch["x"] = adv
            return new_batch
        return adv, labels

    def compute_loss(self, model: Any, batch: Any) -> Tuple[torch.Tensor, Dict[str, float]]:
        images, labels = unpack_xy(batch)
        with self.autocast_context():
            logits = model(images)
            loss = compute_loss(logits, labels)
        correct = (logits.argmax(dim=1) == labels).sum().item()
        metrics = {
            "loss": float(loss.item()),
            "acc": correct / max(labels.size(0), 1),
            "loss_adv": float(loss.item()),
            "acc_adv": correct / max(labels.size(0), 1),
            "correct": correct,
            "batch_size": labels.size(0),
        }
        return loss, metrics
