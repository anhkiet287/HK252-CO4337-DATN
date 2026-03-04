"""PGD attack wrapper."""

from typing import Any, Dict, TYPE_CHECKING

import torch
import torch.nn.functional as F

from ardg.attacks.torchattacks_utils import configure_normalization, require_torchattacks

if TYPE_CHECKING:
    from torch import nn


def build_pgd_attack(cfg: Dict[str, Any], model: "nn.Module") -> Any:
    """Build a PGD attack via TorchAttacks with pixel-space epsilon/alpha."""
    eps = float(cfg["eps"])
    alpha = float(cfg.get("step_size", cfg.get("alpha", 2.0 / 255.0)))
    steps = int(cfg.get("num_steps", cfg.get("steps", 10)))
    restarts = max(1, int(cfg.get("restarts", 1)))
    random_start = bool(cfg.get("random_start", True))
    dataset_name = cfg.get("dataset_name", "cifar10")
    torchattacks = require_torchattacks()
    attack = torchattacks.PGD(
        model,
        eps=eps,
        alpha=alpha,
        steps=steps,
        random_start=random_start,
    )
    attack = configure_normalization(attack, dataset_name)
    if restarts <= 1:
        return attack

    class _RestartedPGD:
        def __init__(self, atk: Any, mdl: "nn.Module", runs: int) -> None:
            self.attack = atk
            self.model = mdl
            self.restarts = runs

        def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            best_adv = images.detach().clone()
            best_loss = torch.full((images.size(0),), float("-inf"), device=images.device)
            for _ in range(self.restarts):
                adv = self.attack(images, labels).detach()
                with torch.no_grad():
                    logits = self.model(adv)
                    per_sample = F.cross_entropy(logits, labels, reduction="none")
                better = per_sample > best_loss
                best_loss = torch.where(better, per_sample, best_loss)
                best_adv = torch.where(better.view(-1, 1, 1, 1), adv, best_adv)
            return best_adv

    return _RestartedPGD(attack, model, restarts)
