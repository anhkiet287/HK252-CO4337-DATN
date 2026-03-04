"""FGSM attack wrapper."""

from typing import Any, Dict, TYPE_CHECKING

from ardg.attacks.torchattacks_utils import configure_normalization, require_torchattacks

if TYPE_CHECKING:
    from torch import nn


def build_fgsm_attack(cfg: Dict[str, Any], model: "nn.Module") -> Any:
    """Build an FGSM attack via TorchAttacks."""
    eps = float(cfg.get("eps", 8.0 / 255.0))
    dataset_name = cfg.get("dataset_name", "cifar10")
    torchattacks = require_torchattacks()
    attack = torchattacks.FGSM(model, eps=eps)
    return configure_normalization(attack, dataset_name)
