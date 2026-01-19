"""PGD attack wrapper."""

from typing import Any, Dict, TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch import nn


def build_pgd_attack(cfg: Dict[str, Any], model: "nn.Module") -> Any:
    """Build a PGD attack."""
    eps = float(cfg["eps"])
    alpha = float(cfg["step_size"])
    steps = int(cfg["num_steps"])

    class _PGD:
        def __init__(self, mdl: "nn.Module") -> None:
            self.model = mdl

        def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            x = images.detach()
            x_adv = x + torch.empty_like(x).uniform_(-eps, eps)
            for _ in range(steps):
                x_adv.requires_grad_(True)
                logits = self.model(x_adv)
                loss = F.cross_entropy(logits, labels)
                grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
                x_adv = x_adv.detach() + alpha * torch.sign(grad.detach())
                delta = torch.clamp(x_adv - x, min=-eps, max=eps)
                x_adv = (x + delta).detach()
            return x_adv

    return _PGD(model)
