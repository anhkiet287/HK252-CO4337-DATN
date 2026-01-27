"""ERM objective (baseline)."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from ardg.training.losses import compute_loss
from ardg.training.objectives.base import Objective


class ERM(Objective):
    """Standard ERM with cross-entropy."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg

    def loss(self, model: Any, batch: Any) -> Tuple[torch.Tensor, Dict[str, float]]:
        images, labels = _unpack_batch(batch)
        logits = model(images)
        loss = compute_loss(logits, labels)
        correct = (logits.argmax(dim=1) == labels).sum().item()
        metrics = {
            "loss": float(loss.item()),
            "acc": correct / max(labels.size(0), 1),
            "correct": correct,
            "batch_size": labels.size(0),
        }
        return loss, metrics


def _unpack_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """Support both tuple and dict batches."""
    if isinstance(batch, dict):
        return batch["x"], batch["y"]
    images, labels = batch
    return images, labels
