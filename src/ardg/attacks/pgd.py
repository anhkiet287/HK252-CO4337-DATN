"""PGD attack wrapper."""

from typing import Any, Dict, TYPE_CHECKING

import torch
import torch.nn.functional as F

from ardg.attacks.torchattacks_utils import configure_normalization, require_torchattacks

if TYPE_CHECKING:
    from torch import nn


def _dlr_loss_per_sample(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute untargeted DLR loss per sample."""
    if logits.ndim != 2:
        raise ValueError("Expected logits with shape (B, C).")
    if logits.size(1) < 3:
        return F.cross_entropy(logits, labels, reduction="none")
    sorted_logits, sorted_idx = logits.sort(dim=1, descending=True)
    z1 = sorted_logits[:, 0]
    z2 = sorted_logits[:, 1]
    z3 = sorted_logits[:, 2]
    zy = logits.gather(1, labels.view(-1, 1)).squeeze(1)
    top_is_y = sorted_idx[:, 0].eq(labels)
    z_other = torch.where(top_is_y, z2, z1)
    return -(zy - z_other) / (z1 - z3 + 1e-12)


def build_pgd_attack(cfg: Dict[str, Any], model: "nn.Module") -> Any:
    """Build a PGD attack via TorchAttacks with pixel-space epsilon/alpha."""
    eps = float(cfg["eps"])
    alpha = float(cfg.get("step_size", cfg.get("alpha", 2.0 / 255.0)))
    steps = int(cfg.get("num_steps", cfg.get("steps", 10)))
    restarts = max(1, int(cfg.get("restarts", 1)))
    random_start = bool(cfg.get("random_start", True))
    dataset_name = cfg.get("dataset_name", "cifar10")
    loss_name = str(cfg.get("loss", "ce")).lower()
    torchattacks = require_torchattacks()
    if loss_name not in {"ce", "dlr"}:
        raise ValueError(f"Unsupported PGD loss: {loss_name!r}. Use 'ce' or 'dlr'.")
    if loss_name == "ce":
        attack = torchattacks.PGD(
            model,
            eps=eps,
            alpha=alpha,
            steps=steps,
            random_start=random_start,
        )
    else:
        attack = torchattacks.UPGD(
            model,
            eps=eps,
            alpha=alpha,
            steps=steps,
            random_start=random_start,
            loss="dlr",
        )
    attack = configure_normalization(attack, dataset_name)
    if restarts <= 1:
        return attack

    class _RestartedPGD:
        def __init__(self, atk: Any, mdl: "nn.Module", runs: int, loss_type: str) -> None:
            self.attack = atk
            self.model = mdl
            self.restarts = runs
            self.loss_type = loss_type

        def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            best_adv = images.detach().clone()
            best_loss = torch.full((images.size(0),), float("-inf"), device=images.device)
            for _ in range(self.restarts):
                adv = self.attack(images, labels).detach()
                with torch.no_grad():
                    logits = self.model(adv)
                    if self.loss_type == "dlr":
                        per_sample = _dlr_loss_per_sample(logits, labels)
                    else:
                        per_sample = F.cross_entropy(logits, labels, reduction="none")
                better = per_sample > best_loss
                best_loss = torch.where(better, per_sample, best_loss)
                best_adv = torch.where(better.view(-1, 1, 1, 1), adv, best_adv)
            return best_adv

    return _RestartedPGD(attack, model, restarts, loss_name)
