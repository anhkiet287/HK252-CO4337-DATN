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
    restarts = max(1, int(cfg.get("restarts", 1)))
    random_start = bool(cfg.get("random_start", True))
    dataset_name = cfg.get("dataset_name", "cifar10")

    # Scale pixel-space eps/alpha into normalized space per channel.
    mean, std = get_dataset_stats(dataset_name)

    class _PGD:
        def __init__(self, mdl: "nn.Module") -> None:
            self.model = mdl

        def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            x = images.detach()
            device = x.device
            mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
            std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)
            eps = eps_pix / std_t
            alpha = alpha_pix / std_t
            lo = (0.0 - mean_t) / std_t
            hi = (1.0 - mean_t) / std_t

            best_adv = x.clone()
            best_loss = torch.full((x.size(0),), float("-inf"), device=device)

            for _ in range(restarts):
                if random_start:
                    noise = torch.empty_like(x).uniform_(-1.0, 1.0) * eps
                    x_adv = torch.max(torch.min(x + noise, hi), lo).detach()
                else:
                    x_adv = x.clone().detach()

                for _ in range(steps):
                    x_adv.requires_grad_(True)
                    logits = self.model(x_adv)
                    loss = F.cross_entropy(logits, labels, reduction="mean")
                    grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
                    x_adv = x_adv.detach() + alpha * torch.sign(grad.detach())
                    delta = x_adv - x
                    delta = torch.max(torch.min(delta, eps), -eps)
                    x_adv = torch.max(torch.min(x + delta, hi), lo).detach()

                with torch.no_grad():
                    logits = self.model(x_adv)
                    per_sample = F.cross_entropy(logits, labels, reduction="none")
                    better = per_sample > best_loss
                    best_loss = torch.where(better, per_sample, best_loss)
                    best_adv = torch.where(better.view(-1, 1, 1, 1), x_adv, best_adv)

            return best_adv

    return _PGD(model)
