"""Evaluation helpers for clean and adversarial metrics."""

from typing import Any, Dict

import torch
from torch import nn

from ardg.training.losses import compute_loss


@torch.no_grad()
def evaluate_clean(model: Any, loader: Any, device: str) -> Dict[str, float]:
    """Evaluate a classifier on clean inputs."""
    model.eval()
    device_t = torch.device(device)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    for images, labels in loader:
        images = images.to(device_t)
        labels = labels.to(device_t)
        logits = model(images)
        loss = compute_loss(logits, labels)
        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_seen += images.size(0)
    return {
        "loss": total_loss / max(total_seen, 1),
        "acc": total_correct / max(total_seen, 1),
        "n_samples": total_seen,
    }


def evaluate_under_attack(model: Any, loader: Any, attack: Any, device: str) -> Dict[str, float]:
    """Evaluate a classifier under a given adversarial attack."""
    model.eval()
    device_t = torch.device(device)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    for images, labels in loader:
        images = images.to(device_t)
        labels = labels.to(device_t)
        adv = attack(images, labels)
        logits = model(adv)
        loss = compute_loss(logits, labels)
        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_seen += images.size(0)
    return {
        "acc": total_correct / max(total_seen, 1),
        "loss": total_loss / max(total_seen, 1),
        "n_samples": total_seen,
    }


def evaluate_suite(model: Any, loader: Any, attacks: Dict[str, Any], device: str) -> Dict[str, Dict[str, float]]:
    """Evaluate a classifier across an attack suite."""
    return {name: evaluate_under_attack(model, loader, atk, device) for name, atk in attacks.items()}
