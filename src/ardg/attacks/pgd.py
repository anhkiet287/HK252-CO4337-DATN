"""PGD attack wrapper."""

from typing import Any, Dict, TYPE_CHECKING

import torch
import torch.nn.functional as F

from ardg.data.transforms import get_dataset_stats

if TYPE_CHECKING:
    from torch import nn


def build_pgd_attack(cfg: Dict[str, Any], model: "nn.Module") -> Any:
    """Build a PGD attack (assumes inputs are normalized)."""
    eps_pix = float(cfg["eps"])
    alpha_pix = float(cfg["step_size"])
    steps = int(cfg["num_steps"])
    dataset_name = cfg.get("dataset_name", "cifar10")

    # Scale pixel-space eps/alpha into normalized space per channel.
    mean, std = get_dataset_stats(dataset_name)
    device = next(model.parameters()).device
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)
    eps = eps_pix / std_t
    alpha = alpha_pix / std_t

    class _PGD:
        def __init__(self, mdl: "nn.Module") -> None:
            self.model = mdl

        def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            x = images.detach()
            # Random start in Linf ball (per-channel eps tensor)
            noise = torch.empty_like(x).uniform_(-1.0, 1.0) * eps
            x_adv = x + noise
            for _ in range(steps):
                x_adv.requires_grad_(True)
                logits = self.model(x_adv)
                loss = F.cross_entropy(logits, labels)
                grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
                x_adv = x_adv.detach() + alpha * torch.sign(grad.detach())
                delta = x_adv - x
                # Per-channel clamp using eps tensor
                delta = torch.max(torch.min(delta, eps), -eps)
                x_adv = (x + delta).detach()
            return x_adv

    return _PGD(model)
