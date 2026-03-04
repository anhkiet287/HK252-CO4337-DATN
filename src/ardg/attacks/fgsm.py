"""FGSM attack wrapper."""

from typing import Any, Dict, TYPE_CHECKING

from ardg.attacks.torchattacks_utils import configure_normalization, require_torchattacks

if TYPE_CHECKING:
    from torch import nn


def build_fgsm_attack(cfg: Dict[str, Any], model: "nn.Module") -> Any:
    """Build an FGSM attack via TorchAttacks."""
    eps = float(cfg.get("eps", 8.0 / 255.0))
    alpha = float(cfg.get("alpha", cfg.get("step_size", eps)))
    random_start = bool(cfg.get("random_start", False))
    dataset_name = cfg.get("dataset_name", "cifar10")
    torchattacks = require_torchattacks()
    if random_start:
        try:
            attack = torchattacks.RFGSM(model, eps=eps, alpha=alpha, steps=1)
        except TypeError:
            attack = torchattacks.RFGSM(model, eps=eps, alpha=alpha)
    else:
        attack = torchattacks.FGSM(model, eps=eps)
    return configure_normalization(attack, dataset_name)
