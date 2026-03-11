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
    norm_name = str(cfg.get("norm", "Linf")).lower()
    torchattacks = require_torchattacks()
    if loss_name not in {"ce", "dlr"}:
        raise ValueError(f"Unsupported PGD loss: {loss_name!r}. Use 'ce' or 'dlr'.")
    if norm_name in {"l2", "2"}:
        from torchattacks.attack import Attack

        class _L2PGD(Attack):
            def __init__(self) -> None:
                super().__init__("L2PGD", model)
                self.eps = eps
                self.alpha = alpha
                self.steps = steps
                self.restarts = restarts
                self.random_start = random_start
                self.loss_type = loss_name
                self.eps_for_division = float(cfg.get("eps_for_division", 1e-10))

            def _per_sample_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
                if self.loss_type == "dlr":
                    return _dlr_loss_per_sample(logits, labels)
                return F.cross_entropy(logits, labels, reduction="none")

            def _run_once(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
                adv_images = images.clone().detach()
                batch_size = len(images)

                if self.random_start:
                    delta = torch.empty_like(adv_images).normal_()
                    d_flat = delta.view(batch_size, -1)
                    n = d_flat.norm(p=2, dim=1).view(batch_size, 1, 1, 1)
                    r = torch.zeros_like(n).uniform_(0, 1)
                    delta *= r / (n + self.eps_for_division) * self.eps
                    adv_images = torch.clamp(adv_images + delta, min=0, max=1).detach()

                for _ in range(self.steps):
                    adv_images.requires_grad = True
                    logits = self.get_logits(adv_images)
                    cost = self._per_sample_loss(logits, labels).sum()
                    grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]
                    grad_norms = torch.norm(grad.view(batch_size, -1), p=2, dim=1) + self.eps_for_division
                    grad = grad / grad_norms.view(batch_size, 1, 1, 1)
                    adv_images = adv_images.detach() + self.alpha * grad

                    delta = adv_images - images
                    delta_norms = torch.norm(delta.view(batch_size, -1), p=2, dim=1) + self.eps_for_division
                    factor = torch.minimum(self.eps / delta_norms, torch.ones_like(delta_norms))
                    delta = delta * factor.view(-1, 1, 1, 1)
                    adv_images = torch.clamp(images + delta, min=0, max=1).detach()

                return adv_images

            def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
                images = images.clone().detach().to(self.device)
                labels = labels.clone().detach().to(self.device)

                best_adv = images.detach().clone()
                best_loss = torch.full((images.size(0),), float("-inf"), device=images.device)
                for _ in range(self.restarts):
                    adv_images = self._run_once(images, labels)
                    with torch.no_grad():
                        logits = self.get_logits(adv_images)
                        per_sample = self._per_sample_loss(logits, labels)
                    better = per_sample > best_loss
                    best_loss = torch.where(better, per_sample, best_loss)
                    best_adv = torch.where(better.view(-1, 1, 1, 1), adv_images, best_adv)
                return best_adv

        attack = _L2PGD()
        return configure_normalization(attack, dataset_name)

    if norm_name not in {"linf", "inf"}:
        raise ValueError(f"Unsupported PGD norm: {norm_name!r}. Use 'Linf' or 'L2'.")

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
