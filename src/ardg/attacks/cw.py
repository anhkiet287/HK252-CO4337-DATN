"""Carlini-Wagner attack wrapper."""

from typing import Any, Dict, TYPE_CHECKING

from ardg.attacks.torchattacks_utils import configure_normalization, require_torchattacks

if TYPE_CHECKING:
    from torch import nn


def build_cw_attack(cfg: Dict[str, Any], model: "nn.Module") -> Any:
    """Build a CW attack via TorchAttacks."""
    c = float(cfg.get("c", 1.0))
    kappa = float(cfg.get("kappa", 0.0))
    steps = int(cfg.get("num_steps", cfg.get("steps", 50)))
    lr = float(cfg.get("lr", cfg.get("learning_rate", 0.01)))
    dataset_name = cfg.get("dataset_name", "cifar10")
    torchattacks = require_torchattacks()
    attack = torchattacks.CW(model, c=c, kappa=kappa, steps=steps, lr=lr)
    return configure_normalization(attack, dataset_name)
